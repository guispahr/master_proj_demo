import re
import torch

from utils.registry import Registry

OPTIMIZERS = Registry("optimizers")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")
OPTIMIZERS.register_module(module=torch.optim.Adam,  name="Adam")
OPTIMIZERS.register_module(module=torch.optim.SGD,   name="SGD")


def build_param_groups(model, base_lr, param_dicts_cfg):
    """Split model parameters into groups based on keyword/regex matching.

    Each entry in param_dicts_cfg can override lr, weight_decay, or momentum
    for parameters whose name matches the given keyword (regex).
    Order matters: the first matching keyword wins.
    """
    regexes = [(re.compile(d["keyword"]), d) for d in param_dicts_cfg]

    groups = [{"params": [], "lr": base_lr, "_label": "base"}]
    for _, d in regexes:
        group = {"params": [], "lr": d["lr"], "_label": d["keyword"]}
        for key in ("weight_decay", "momentum"):
            if key in d:
                group[key] = d[key]
        groups.append(group)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        idx = next((i + 1 for i, (rx, _) in enumerate(regexes) if rx.search(name)), 0)
        groups[idx]["params"].append(param)

    # Drop empty groups — label is stored in each group so indices don't need to stay aligned
    groups = [g for g in groups if g["params"]]

    for i, g in enumerate(groups):
        label  = g.pop("_label")
        extras = "  ".join(f"{k}={v}" for k, v in g.items() if k not in ("params", "lr"))
        print(f"  param group {i}: [{label}]  lr={g['lr']:.2e}  n={len(g['params'])}  {extras}")

    return groups


def build_optimizer(model, optim_cfg, base_lr, param_dicts_cfg=None):
    """Build an optimizer, optionally with per-group learning rates.

    Args:
        model:            the model whose parameters to optimise
        optim_cfg:        config namespace with at least 'optimizer' field
        base_lr:          peak LR for the base (unmatched) parameter group
        param_dicts_cfg:  list of dicts with 'keyword', 'lr', and optional
                          'weight_decay' / 'momentum' overrides
    """
    if param_dicts_cfg:
        params = build_param_groups(model, base_lr, param_dicts_cfg)
    else:
        params = model.parameters()

    optimizer_name = optim_cfg.optimizer.upper()
    weight_decay   = getattr(optim_cfg, "weight_decay", 0.01)

    # Case-insensitive lookup (config may use 'adamw', registry key is 'AdamW')
    opt_cls = next(
        (v for k, v in OPTIMIZERS._module_dict.items() if k.upper() == optimizer_name),
        None,
    )
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer '{optim_cfg.optimizer}'. "
                         f"Registered: {list(OPTIMIZERS._module_dict)}")

    kwargs = {"lr": base_lr, "weight_decay": weight_decay}
    if optimizer_name == "SGD":
        kwargs["momentum"] = getattr(optim_cfg, "momentum", 0.9)

    return opt_cls(params, **kwargs)
