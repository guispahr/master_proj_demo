from utils.registry import Registry

SCHEDULERS = Registry("schedulers")

from .linearwarmcosinescheduler import LinearWarmupCosineScheduler
from .onecyclelr import OneCycleLRScheduler

SCHEDULERS.register_module(module=LinearWarmupCosineScheduler)
SCHEDULERS.register_module(module=OneCycleLRScheduler)
