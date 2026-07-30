from omegaconf import OmegaConf

from utils.registry import Registry
from .transforms import TRANSFORMS, Compose

DATASETS = Registry("datasets")


def build_transforms(cfg):
    transforms = []
    # cfg can be a dict {Name: params, ...} or a list [{Name: params}, ...].
    # List form allows duplicate transform names (e.g. two SphereCrop in sequence).
    items = cfg if isinstance(cfg, list) else [cfg]
    for entry in items:
        for name, params in entry.items():
            if params is None:
                params = {}
            elif not isinstance(params, dict):
                params = {"value": params}
            transform = TRANSFORMS.build({"type": name, **params})
            transforms.append(transform)
    return Compose(transforms)


def build_dataset(cfg, mode):
    cfg_key = "training" if mode == "train" else mode
    cfg_section = cfg[cfg_key]
    transform_params = OmegaConf.to_container(cfg_section.transform, resolve=True)
    transform = build_transforms(transform_params)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg["data_root"] = cfg.data.data_root
    if hasattr(cfg_section, "split"):
        split_val = cfg_section.split
        dataset_cfg["split"] = OmegaConf.to_container(split_val) if OmegaConf.is_config(split_val) else split_val
    else:
        dataset_cfg["split"] = mode
    dataset_cfg["transform"] = transform

    if mode == "test":
        test_cfg_raw = cfg.test.test_cfg if (hasattr(cfg, "test") and hasattr(cfg.test, "test_cfg")) else None
        test_cfg = OmegaConf.to_container(test_cfg_raw, resolve=True) if test_cfg_raw is not None else None
        dataset_cfg["test_mode"] = test_cfg is not None
        dataset_cfg["test_cfg"] = test_cfg

    return DATASETS.build(dataset_cfg)
