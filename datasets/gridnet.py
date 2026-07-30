from pathlib import Path
import json
import warnings

import numpy as np
import cv2
import time
from torch.utils.data import DataLoader

from datasets.builder import DATASETS, build_transforms, Compose
from datasets.defaults import DefaultDataset
from datasets.utils import collate_fn

CLASSES = [
    ("pylon",      "#bebebe"),  #  0  light gray (metal)
    ("conductor",  "#ff8c00"),  #  1  dark orange (cables)
    ("structural", "#8b4513"),  #  2  saddle brown
    ("insulator",  "#ff00ff"),  #  3  magenta
    ("high_veg",   "#228b22"),  #  4  forest green
    ("low_veg",    "#7cfc00"),  #  5  lawn green
    ("herb",       "#daa520"),  #  6  goldenrod
    ("gravel",     "#d2b48c"),  #  7  tan
    ("impervious", "#808080"),  #  8  gray
    ("water",      "#1e90ff"),  #  9  dodger blue
    ("building",   "#dc143c"),  # 10  crimson
    ("unlabeled",  "#000000"),  # 11  black
]
names  = [c[0] for c in CLASSES]
colors = [c[1] for c in CLASSES]

@DATASETS.register_module("GridNet-Pair-Dataset")
class GridNetPairDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "strength",
        "segment",
        "image_coord",
        "image_mask",
    ]
    CLASS_NAMES = names
    CLASS_COLORS = colors

    def __init__(
        self,
        data_root: Path,
        split: Path = "train",
        transform: Compose = None,
        with_normals: bool = False,
        with_image: bool = True,
        test_mode: bool = False,
        test_cfg: dict | None = None,
        cache: bool = False,
        ignore_index: int = -1,
        loop: int = 1,
    ):
        """
        Dataset for pre-chunked GridNet point clouds.

        On-disk layout (per chunk):
            chunk_dir/xyz.npy           (N, 3)
            chunk_dir/rgb.npy           (N, 3)
            chunk_dir/intensity.npy     (N,)
            chunk_dir/labels.npy        (N,)   — only present on train/val
            chunk_dir/image_coord.npz   (key 'coord' → (N, max_cam, 2))
            chunk_dir/meta.json         (with `cam_names`)
            chunk_dir/../images/{cam}.JPG

        Parameters
        ----------
        data_root : preprocessed-root directory containing `manifest.json`
                    and `<split>/<zone>/chunk_XXXXX/` subfolders.
        split     : "train" | "val" | "test"
        transform : a `Compose` object built by `build_transforms`.
        with_normals : if True, expose a zero `normal` (N, 3) channel.
        with_image   : if True, load camera JPGs and `image_coord`/`image_mask`.

        Other kwargs (`test_mode`, `test_cfg`, `cache`, `ignore_index`, `loop`)
        pass straight through to `DefaultDataset` for Pointcept API parity.
        """
        self.with_normals = with_normals
        self.with_image = with_image
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            ignore_index=ignore_index,
            loop=loop,
        )

    @property
    def chunk_paths(self):
        """Backwards-compatible alias for `self.data_list` used by the tester."""
        return self.data_list

    def get_data_list(self):
        chunk_root = Path(self.data_root)
        with open(chunk_root / "manifest.json", "r") as f:
            manifest = json.load(f)
        paths_dict_list = manifest[self.split]
        chunk_paths = [
            str(chunk_root / self.split / d["zone"] / Path(d["chunk_dir"]).name)
            for d in paths_dict_list
        ]
        if not chunk_paths:
            raise RuntimeError(f"No chunk files found in {chunk_root / self.split}")
        return chunk_paths

    def get_data(self, idx):

        chunk_path = Path(self.data_list[idx % len(self.data_list)])
        # Read meta data
        with open(chunk_path/"meta.json", "r") as f:
            meta = json.load(f)

        # Load coords with checks
        coord = np.load(chunk_path / "xyz.npy", mmap_mode="r")

        if coord.dtype == np.int16:
            # Stored in int16 for memory effciency :)
            coord = coord.astype(np.float32) / 100.0
        elif coord.dtype == np.float32:
            pass  # already good
        else:
            raise TypeError(f"Unsupported dtype: {coord.dtype}")

        # Load color
        color = np.load(chunk_path / "rgb.npy")#, mmap_mode="r")

        # Load intensity
        intensity = np.load(chunk_path / "intensity.npy")#, mmap_mode="r")

        if intensity.dtype == np.uint8:
            # Stored in int16 for memory effciency :)
            intensity = intensity.astype(np.float32) * 255
        elif intensity.dtype == np.float32:
            pass  # already good
        else:
            raise TypeError(f"Unsupported dtype: {intensity.dtype}")

        # Load segment (absent for test set)
        segment = None
        if (chunk_path / "labels.npy").exists():
            segment = np.load(chunk_path / "labels.npy")#, mmap_mode="r")

        data_dict = {
            "coord": coord,
            "color": color,
            "strength": intensity,
            "name": chunk_path.name,
        }
        if segment is not None:
            data_dict["segment"] = segment

        if self.with_normals:
            data_dict["normal"]  = np.zeros_like(color)   # (N, max_cam)

        if self.with_image:
            coord_npz = chunk_path / "image_coord.npz"
            if not coord_npz.exists():
                warnings.warn(
                    f"image_coord.npz missing for {chunk_path}; "
                    f"treating chunk as image-free (zero cameras)."
                )
                N = coord.shape[0]
                data_dict["image"]       = []
                data_dict["image_coord"] = np.zeros((N, 0, 2), dtype=np.int16)
                data_dict["image_mask"]  = np.zeros((N, 0), dtype=bool)
                return data_dict
            image_coord_full = np.load(coord_npz)["coord"]   # (N, max_cam, 2)
            image_mask_full  = image_coord_full[..., 0] >= 0                       # (N, max_cam)
            images_folder    = chunk_path.parent / "images"
            cam_names = [c for c in meta["cam_names"] if c]                        # filtered: matches image_coord's CAM dim

            images: list = []
            valid_cols: list[int] = []
            for j, cam in enumerate(cam_names):
                if not image_mask_full[:, j].any():                                # no point projects into this cam
                    continue
                raw = cv2.imread(str(images_folder / f"{cam}.JPG"))
                if raw is None:                                                     # JPG missing on disk
                    continue
                images.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                valid_cols.append(j)

            if valid_cols:
                cols = np.asarray(valid_cols, dtype=np.int64)
                image_coord = image_coord_full[:, cols]                            # (N, CAM_active, 2)
                image_mask  = image_mask_full [:, cols]                            # (N, CAM_active)
            else:
                image_coord = image_coord_full[:, :0]
                image_mask  = image_mask_full [:, :0]

            data_dict["image"]       = images        # list of length CAM_active (variable, may be 0)
            data_dict["image_coord"] = image_coord
            data_dict["image_mask"]  = image_mask

        return data_dict

    def _control_number_of_images(self):
        num_cam_per_chunk = []
        for path in self.chunk_paths:
            chunk_path = Path(path)
            with open(chunk_path/"meta.json", "r") as f:
                meta = json.load(f)
            num_cam_per_chunk.append(sum(1 for c in meta["cam_names"] if c))
        num_cam_per_chunk = np.array(num_cam_per_chunk)
        unique_number = np.unique(num_cam_per_chunk)
        if len(unique_number) != 1:
            print(len(unique_number), " unique numbers")
            print(unique_number)
        if len(unique_number) == 1:
            print("All chunks have the same number of image pairs")

# if __name__ == "__main__":

#     # weights = np.array([12.987, 22.727, 90.909, 181.818, 0.733, 2.674, 0.121, 1.337, 11.364, 454.545, 22.727])
#     # weigths = weights/weights.mean()
#     # print(weights)
#     # sqrt_weights = weights**0.5
#     # print(sqrt_weights)
#     # sqrt_weights = np.clip(np.sqrt(weights), 0, 20)
#     # print(sqrt_weights)

#     # path = "/mnt/c/Users/guill/OneDrive/Bureau/Personal_projects/PDM_tests/GridNet-HD/chunks_test"
#     # dataset = GridNetPairDataset(path, "train", None, True)
#     # dataset._control_number_of_images()
    
#     # for sample in dataset:
#     #     if sample["coord"].shape[0]>=100000:
#     #         print(sample["coord"].shape[0])
#     #         vizualize_point_cloud_with_colors(sample["coord"], sample["color"]/255)

#     # sample0 = dataset[100]
    
#     # vizualize_point_cloud_with_colors(sample0["coord"], sample0["color"]/255)
#     # # print(dataset[0])
#     # import matplotlib.pyplot as plt
#     # print(sample0["image_coord"].shape) 
#     # print(sample0["image_coord"].swapaxes(0, 1)[0].max(0)) # (W x H)
#     # plt.imshow(sample0["image"][0])
#     # plt.axis("off")
#     # plt.show()
#     from utils.visualization import vizualize_point_cloud_with_colors
#     from utils.profiling import batch_memory
#     from datasets.transforms import *

#     path = "/mnt/c/Users/guill/OneDrive/Bureau/Personal_projects/PDM_tests/GridNet-HD"

#     cfg = {
#             "Update": {
#                 "keys_dict" : {
#                     "index_valid_keys": [
#                         "coord",
#                         "color",
#                         "intensity",
#                         "segment",
#                         "grid_coord"
#                     ]
#                 }
#             },
#             "GridSample": {
#                 "grid_size" : 0.05,
#                 "hash_type" : "fnv",
#                 "mode" : "train",
#                 "return_inverse" : False,
#                 "return_grid_coord" : True,
#                 "return_min_coord" : False,
#                 "return_displacement" : False,
#                 "project_displacement" : False,
#             },
#             # "SmartPointSampler":{
#             #     "max_points": 200000,
#             #     "class_weights": [12.987, 22.727, 30., 30., 0.733, 2.674, 0.121, 1.337, 11.364, 30., 22.727],
#             #     "ignore_index": 11,
#             #     "ignore_kept_weight" : 0.05
#             # },
#             "ToTensor": {},
            
#             "Collect": {
#                 "keys" : ["coord", "grid_coord", "segment", "color", "image", "image_coord", "image_mask"],
#                 "feat_keys":["coord", "color"]
#             }
            
#         }
    
    
    
#     transforms = build_transforms(cfg)
#     path = "/mnt/c/Users/guill/OneDrive/Bureau/Personal_projects/PDM_tests/GridNet-HD/preprocessed"
#     # path = "/mnt/scratch/students/gspahr/GridNet_preprocessed"
#     dataset = GridNetPairDataset(path, "train", transforms, False)

#     # See utils/analyze_chunks.py for chunk/coverage/image-pass statistics.
#     quit()


#     dataset._control_number_of_images()
    
#     for sample in dataset:
#         print(sample["coord"].shape[0])
#         # sample["coord"] -= sample["coord"].mean(0)
#         # print(sample["coord"].min(0), sample["coord"].max(0))
#         # print(sample["coord"].max(0).values - sample["coord"].min(0).values)
#         # if sample["coord"].shape[0]>=200000:
#         #     print(sample["coord"].shape[0])
#         #     print(torch.unique(sample["segment"], return_counts =True))
#         #     vizualize_point_cloud_with_colors(sample["coord"], sample["color"]/255)
#     quit()

#     dataset = ChunkGridNetSemanticPointCloud(path+"/chunks", 
#                                              "train",
#                                              transform = transforms)
    
#     print(len(dataset))
#     print(dataset[-1])
    

#     # quit()

#     # label_counts, percentages, weights = dataset._get_label_distribution()
#     # print("Counts:")
#     # for i in range(label_counts.shape[0]):
#     #     print(f"Label {i}: Counts {label_counts[i]}, percentage {percentages[i]}, weight {weights[i]}")

#     # dataset = ChunkGridNetSemanticPointCloud(path+"/chunks", 
#     #                                         "val",
#     #                                         grid_size=0.05,
#     #                                         max_points=100000)
#     # label_counts, percentages, weights = dataset._get_label_distribution()
#     # print("Counts:")
#     # for i in range(label_counts.shape[0]):
#     #     print(f"Label {i}: Counts {label_counts[i]}, percentage {percentages[i]}, weight {weights[i]}")
    

#     batch_size=1
    
#     loader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         collate_fn=collate_fn, 
#         num_workers=batch_size
#     )

#     t1= time.time()
#     # Get one batch
#     batch = next(iter(loader))
#     print(f"Time to load a batch of size {batch_size} : {time.time()-t1}")
#     print(f"Batch memory: {batch_memory(batch):.2f} MB")

#     print(batch["coord"])
    
#     # vizualize_point_cloud_with_colors(batch["coord"].numpy(), batch["feat"].numpy())
#     timings = []
#     start = time.time()
#     for i, batch in enumerate(loader):
#         now = time.time()
#         diff = now - start
#         print(f"Batch {i} loading time: {now - start:.4f} s")

#         # your training / inference code here
#         timings.append(diff)
#         start = time.time()
#         print(f"Number of points in the batch: {batch['coord'].shape[0]}")

#         if i>50:
#             break
    
#     mean_time = np.mean(timings)
#     print("Mean time to load a batch:", mean_time)

#     for i, batch in enumerate(loader):
#         print(f"Number of points in the batch: {batch['coord'].shape[0]}")
#         coord = batch["coord"] - batch["coord"].min(0)[0]
#         vizualize_point_cloud_with_colors(coord, batch["color"].numpy()/255)
        # quit()
