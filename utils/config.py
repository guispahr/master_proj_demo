from types import SimpleNamespace

class Config:
    """Unified config wrapper supporting dict, OmegaConf, and namespaces."""

    def __init__(self, cfg):

        try:
            from omegaconf import OmegaConf
        except ImportError:
            OmegaConf = None

        if OmegaConf is not None and OmegaConf.is_config(cfg):
            # resolve=True expands ${interpolations}; raises clearly if a key is missing
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            self.args = self._to_namespace(cfg)

        elif isinstance(cfg, SimpleNamespace):
            self.args = cfg

        else:
            self.args = cfg or SimpleNamespace()

    def _to_namespace(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._to_namespace(x) for x in d]
        return d

    def __getattr__(self, item):
        return getattr(self.args, item)