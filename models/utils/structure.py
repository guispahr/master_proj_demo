import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict

from models.utils.serialization import encode, decode
from models.utils import offset2batch, batch2offset


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust grid_size to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        # A single-voxel cloud (e.g. a 1-point test fragment) has grid_coord.max()==0
        # -> bit_length()==0. Serialization needs >=1 bit; depth==0 also mis-slices the
        # gray code ([..., -0:] keeps all 64 bits) and crashes the hilbert reshape.
        depth = max(depth, 1)
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust grid_size to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        # spconv flattens (b, z, y, x) indices with int32. A chunk with outlier
        # coordinates inflates grid_coord -> the spatial-shape product can exceed
        # 2**31, which surfaces downstream as a *silent CUDA "illegal memory access"*
        # (often misattributed to the next kernel). Fail loud and early instead.
        # Uses the already-synced sparse_shape list -> no extra device sync.
        _shape_prod = 1
        for _s in sparse_shape:
            _shape_prod *= int(_s)
        if _shape_prod >= 2**31:
            raise ValueError(
                f"sparsify: spatial_shape {sparse_shape} (prod={_shape_prod:.3e}) exceeds "
                f"spconv's int32 index limit. A chunk almost certainly has outlier "
                f"coordinates - sanitize coords at data-load time (see sanitize_points)."
            )
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn."
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
