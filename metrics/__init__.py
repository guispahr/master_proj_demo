from utils.registry import Registry

METRICS = Registry("metrics")

from .metrics import SegmentationMetrics

METRICS.register_module(module=SegmentationMetrics)
