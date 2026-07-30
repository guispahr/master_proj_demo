import torch
from einops import rearrange

_DEBUG = True  # set False to silence image-shape debug prints


def pack_cameras(data_dict: dict) -> None:
    """
    Replaces (B, CAM, C, H, W) image tensors with a flat (N_active_cams, C, H, W)
    form, keeping only cameras that see at least one point in each batch item.

    Operates in-place.  After this call:
        image       (B, CAM, C, H, W)  ->  (N_active_cams, C, H, W)
        image_mask  (N, CAM)           ->  (N, CAM_active_max)
        image_coord (N, CAM, 2)        ->  (N, CAM_active_max, 2)
        cam_offset  (B,)               added - same semantics as point 'offset'

    Safe to call from collate_fn (CPU tensors) or from model forward (GPU tensors).
    """
    offset = data_dict["offset"].tolist()
    B = len(offset)
    splits = [offset[0]] + [offset[i] - offset[i - 1] for i in range(1, B)]

    mask_full  = data_dict["image_mask"]   # (N, CAM)
    coord_full = data_dict["image_coord"]  # (N, CAM, 2)
    images     = data_dict["image"]        # (B, CAM, C, H, W)
    device     = mask_full.device

    # Optional per-(point, camera) depth (model K's aux-depth target) - packed
    # alongside image_mask so its camera axis stays aligned after activation.
    depth_full = data_dict.get("image_depth", None)  # (N, CAM) or None

    packed_imgs, new_masks, new_coords, new_depths = [], [], [], []
    cam_offset_list = []
    n_cams, n_pts = 0, 0

    for b in range(B):
        n_b      = splits[b]
        mask_b   = mask_full[n_pts : n_pts + n_b]        # (N_b, CAM)
        active_b = mask_b.any(dim=0)                      # (CAM,)
        idx_b    = active_b.nonzero(as_tuple=True)[0]    # (n_active_b,)

        packed_imgs.append(images[b, idx_b])              # (n_active_b, C, H, W)
        new_masks.append(mask_b[:, idx_b])                # (N_b, n_active_b)
        new_coords.append(coord_full[n_pts : n_pts + n_b, idx_b])  # (N_b, n_active_b, 2)
        if depth_full is not None:
            new_depths.append(depth_full[n_pts : n_pts + n_b, idx_b])  # (N_b, n_active_b)

        n_cams += idx_b.shape[0]
        n_pts  += n_b
        cam_offset_list.append(n_cams)

    # Pad mask / coord to the widest active-camera count so they stack into a tensor
    CAM_max = max(m.shape[1] for m in new_masks)
    N       = mask_full.shape[0]
    mask_out  = torch.zeros(N, CAM_max, dtype=torch.bool, device=device)
    coord_out = torch.zeros(N, CAM_max, 2, dtype=coord_full.dtype, device=device)
    depth_out = (
        torch.zeros(N, CAM_max, dtype=depth_full.dtype, device=device)
        if depth_full is not None else None
    )

    n_pts = 0
    for b in range(B):
        n_b, n_act = splits[b], new_masks[b].shape[1]
        mask_out [n_pts : n_pts + n_b, :n_act] = new_masks[b]
        coord_out[n_pts : n_pts + n_b, :n_act] = new_coords[b]
        if depth_out is not None:
            depth_out[n_pts : n_pts + n_b, :n_act] = new_depths[b]
        n_pts += n_b

    data_dict["image"]       = torch.cat(packed_imgs, dim=0)  # (N_active_cams, C, H, W)
    data_dict["image_mask"]  = mask_out                        # (N, CAM_active_max)
    data_dict["image_coord"] = coord_out                       # (N, CAM_active_max, 2)
    if depth_out is not None:
        data_dict["image_depth"] = depth_out                   # (N, CAM_active_max)
    data_dict["cam_offset"]  = torch.tensor(
        cam_offset_list, dtype=torch.long, device=device
    )                                                          # (B,)


def get_image_feat_packed(img_enc, data_dict, cam_batch: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Like get_image_feat but expects data_dict["image"] already packed as
    (N_active_cams, C, H, W) - as produced by pack_cameras.

    Returns feat_map (N_active_cams, C, H_f, W_f), cls_token (N_active_cams, C), patch_size.

    cam_batch (None = disabled, the default) feeds the packed cameras through
    the ViT in groups of at most cam_batch and concatenates. The ViT treats
    cameras as an independent batch dim, so this is bit-identical to one big
    forward, but it bounds peak activation memory. Training/validation leave this
    None (one forward, as before); only the tester sets it, because a single
    full-resolution test chunk can project to far more cameras than any training
    step packed - running a giant ViT on the whole stack at 10362 in fp16 can then
    exhaust memory mid-kernel, which flash-attention surfaces as "illegal memory
    access" rather than a clean OOM.
    """
    images = data_dict["image"]   # (N_cams, C, H, W)
    if cam_batch is None or images.shape[0] <= cam_batch:
        feat_map, cls_token = img_enc(images)
        return feat_map, cls_token, img_enc.patch_size

    feat_chunks, cls_chunks = [], []
    for i in range(0, images.shape[0], cam_batch):
        fm, ct = img_enc(images[i : i + cam_batch])
        feat_chunks.append(fm)
        cls_chunks.append(ct)
    return torch.cat(feat_chunks, dim=0), torch.cat(cls_chunks, dim=0), img_enc.patch_size

# def get_image_feat_packed(img_enc, data_dict) -> tuple[torch.Tensor, torch.Tensor, int]:
#     """
#     Like get_image_feat but expects data_dict["image"] already packed as
#     (N_active_cams, C, H, W) - as produced by pack_cameras.

#     Returns feat_map (N_active_cams, C, H_f, W_f), cls_token (N_active_cams, C), patch_size.
#     """
#     ps   = img_enc.patch_size
#     image = data_dict["image"]
#     if image.shape[0] == 0:
#         # No cameras in this batch (chunk with no camera coverage).
#         # Skip DINO entirely and return empty feature maps; the model's
#         # cross-attention will find 0 active cameras for every scene and
#         # fall back to the missing-feature embedding for every point.
#         C_out = img_enc.output_channels
#         return image.new_zeros(0, C_out, 1, 1), image.new_zeros(0, C_out), ps
#     feat_map, cls_token = img_enc(image)   # already flat
#     return feat_map, cls_token, ps



def get_image_feat(img_enc, data_dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    feat_map, cls_token = img_enc(
        rearrange(data_dict["image"], "b cam c h w -> (b cam) c h w")
    )
    feat_map = rearrange(
        feat_map,
        "(b cam) c h w -> b cam c h w",
        b=data_dict["image"].shape[0],
    )
    cls_token = rearrange(
        cls_token,
        "(b cam) c -> b cam c",
        b=data_dict["image"].shape[0],
    )
    return feat_map, cls_token, img_enc.patch_size


# def assign_image_feat(
#     data_dict: dict,
#     image_feat: torch.Tensor,  # (B, CAM, C, H, W)
#     patch_size: int,
#     missing_feature_embedding: torch.Tensor | None = None,  # (C,)
# ) -> torch.Tensor:  # (N, C)
#     """
#     data_dict["image_coord"]: (N, CAM, 2)
#     data_dict["image_mask"]: (N, CAM)
#     data_dict["offset"]: (B,)
#     """

#     point_image_feat = []

#     offset = (
#         data_dict["unmix3d_offset"]
#         if "unmix3d_offset" in data_dict.keys()
#         else data_dict["offset"]
#     ).tolist()
#     splits = [o - offset[i - 1] if i > 0 else o for i, o in enumerate(offset)]
#     image_coord = torch.split(
#         data_dict["image_coord"] / patch_size, splits
#     )  # [(N_i, CAM, 2)]
#     image_mask = torch.split(data_dict["image_mask"], splits)  # [(N_i, CAM, 2)]

#     for i, image_feat_i in enumerate(image_feat):
#         N_i = image_coord[i].shape[0]
#         C = image_feat_i.shape[1]
#         point_image_feat_i = (
#             missing_feature_embedding.expand(N_i, -1).clone()
#             if missing_feature_embedding is not None
#             else image_feat_i.new_zeros(N_i, C)
#         )

#         for j, image_feat_ij in enumerate(image_feat_i):
#             image_coord_ij = image_coord[i][:, j]
#             image_mask_ij = image_mask[i][:, j]

#             # temporary fix; should be handled a bit more elegantly. after augmentations
#             # and scaling, the coords can be out of bounds due to rounding errors.
#             H, W = image_feat_ij.shape[-2:]
#             image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 1] < H)
#             image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 0] < W)

#             point_image_feat_i[image_mask_ij] = rearrange(
#                 image_feat_ij, "c h w -> h w c"
#             )[
#                 image_coord_ij[image_mask_ij][:, 1].int(),
#                 image_coord_ij[image_mask_ij][:, 0].int(),
#             ]

#         point_image_feat.append(point_image_feat_i)

#     return torch.cat(point_image_feat)

def assign_image_feat(
    data_dict: dict,
    image_feat: torch.Tensor,  # (B, CAM, C, H, W)
    patch_size: int,
    missing_feature_embedding: torch.Tensor | None = None,  # (C,)
) -> torch.Tensor:  # (N, C)
    """
    data_dict["image_coord"]: (N, CAM, 2)
    data_dict["image_mask"]: (N, CAM)
    data_dict["offset"]: (B,)
    """

    point_image_feat = []

    offset = (
        data_dict["unmix3d_offset"]
        if "unmix3d_offset" in data_dict.keys()
        else data_dict["offset"]
    ).tolist()
    splits = [o - offset[i - 1] if i > 0 else o for i, o in enumerate(offset)]
    image_coord = torch.split(
        data_dict["image_coord"] / patch_size, splits
    )  # [(N_i, CAM, 2)]
    image_mask = torch.split(data_dict["image_mask"], splits)  # [(N_i, CAM)]

    for i, image_feat_i in enumerate(image_feat):
        N_i = image_coord[i].shape[0]
        C = image_feat_i.shape[1]
        point_image_feat_i = (
            missing_feature_embedding.expand(N_i, -1).clone()
            if missing_feature_embedding is not None
            else image_feat_i.new_zeros(N_i, C)
        )

        for j, image_feat_ij in enumerate(image_feat_i):
            image_coord_ij = image_coord[i][:, j]
            image_mask_ij = image_mask[i][:, j]

            # temporary fix; should be handled a bit more elegantly. after augmentations
            # and scaling, the coords can be out of bounds due to rounding errors.
            H, W = image_feat_ij.shape[-2:]
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 1] < H)
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 0] < W)

            point_image_feat_i[image_mask_ij] = image_feat_ij.permute(1, 2, 0)[
                image_coord_ij[image_mask_ij][:, 1].int(),
                image_coord_ij[image_mask_ij][:, 0].int(),
            ]

        point_image_feat.append(point_image_feat_i)

    return torch.cat(point_image_feat)


def assign_image_feat_multi(
    data_dict: dict,
    image_feat: torch.Tensor,  # (B, CAM, C, H, W)
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # (N, CAM, C), (N, CAM) bool
    """
    Returns per-point per-camera patch features and a visibility mask.
    Unlike assign_image_feat, no camera is selected: all visible cameras are kept,
    so points can later fuse them with attention.
    """
    point_image_feats = []
    point_image_masks = []

    offset = (
        data_dict["unmix3d_offset"]
        if "unmix3d_offset" in data_dict.keys()
        else data_dict["offset"]
    ).tolist()
    splits = [o - offset[i - 1] if i > 0 else o for i, o in enumerate(offset)]
    image_coord = torch.split(
        data_dict["image_coord"] / patch_size, splits
    )  # [(N_i, CAM, 2)]
    image_mask = torch.split(data_dict["image_mask"], splits)  # [(N_i, CAM)]

    for i, image_feat_i in enumerate(image_feat):
        N_i = image_coord[i].shape[0]
        CAM = image_feat_i.shape[0]
        C = image_feat_i.shape[1]

        point_feat_i = image_feat_i.new_zeros(N_i, CAM, C)
        point_mask_i = image_mask[i].clone()  # (N_i, CAM)

        for j, image_feat_ij in enumerate(image_feat_i):
            image_coord_ij = image_coord[i][:, j]
            image_mask_ij = image_mask[i][:, j]

            H, W = image_feat_ij.shape[-2:]
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 1] < H)
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 0] < W)

            point_mask_i[:, j] = image_mask_ij
            point_feat_i[image_mask_ij, j] = image_feat_ij.permute(1, 2, 0)[
                image_coord_ij[image_mask_ij][:, 1].int(),
                image_coord_ij[image_mask_ij][:, 0].int(),
            ]

        point_image_feats.append(point_feat_i)
        point_image_masks.append(point_mask_i)

    return torch.cat(point_image_feats, dim=0), torch.cat(point_image_masks, dim=0)


def mix3d_cls_token(
    data_dict: dict,
    cls_token: torch.Tensor,  # (B, CAM, C)
) -> torch.Tensor:  # (B', CAM, C)
    """
    reshapes unmix3d cls tokens to mix3d cls tokens if necessary
    """
    if "unmix3d_offset" in data_dict.keys():
        assert len(data_dict["unmix3d_offset"]) / len(data_dict["offset"]) == 2
        return rearrange(cls_token, "(b mix) cam c -> b (mix cam) c", mix=2)
    else:
        return cls_token
