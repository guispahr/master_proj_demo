from losses import LOSSES


class LossManager:

    def __init__(self, loss_configs):
        self.losses = []
        for cfg in loss_configs:
            cfg = cfg.copy()
            weight = cfg.pop("loss_weight", 1.0)
            name = cfg.get("type")
            loss_fn = LOSSES.build(cfg)
            self.losses.append((name, loss_fn, weight))

    def __call__(self, outputs, targets=None):
        """
        outputs : model output — either a plain logits tensor (Base-Segmentor)
                  or a Point dict with 'logits', 'coord', 'segment', … (Point-Segmentor).
        targets : optional label tensor (N,).  When outputs is a Point dict and
                  'segment' is present, targets is extracted automatically so the
                  caller does not need to pass it separately.
        """
        # Auto-extract targets from the output dict when not provided explicitly.
        if targets is None and isinstance(outputs, dict) and "segment" in outputs:
            targets = outputs["segment"]

        total_loss = 0
        loss_dict  = {}

        for name, loss_fn, weight in self.losses:
            needs_dict = getattr(loss_fn, "NEEDS_POINT_DICT", False)

            if needs_dict:
                # Loss handles everything internally (targets already in the dict).
                value = loss_fn(outputs)
            else:
                # Standard loss: extract logits tensor, pass with targets.
                inp = outputs.get("logits", outputs) if isinstance(outputs, dict) else outputs
                value = loss_fn(inp, targets)

            loss_dict[name] = value.item()
            total_loss += weight * value

        return total_loss, loss_dict
