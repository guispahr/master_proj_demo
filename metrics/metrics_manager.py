import torch


class MetricManager:

    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics:
            if hasattr(m, "reset"):
                m.reset()

    @staticmethod
    def _detach(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs.detach()
        if isinstance(outputs, dict):
            return {k: MetricManager._detach(v) for k, v in outputs.items()}
        return outputs

    def update(self, outputs, batch):
        outputs = self._detach(outputs)
        for m in self.metrics:
            m.update(outputs, batch)

    def compute(self):
        results = {}
        for m in self.metrics:
            results.update(m.compute())
        return results