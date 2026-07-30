from utils.registry import Registry

LOGGERS = Registry("loggers")

from .loggers import BaseLogger, ConsoleLogger, CSVLogger, WandbLogger, ProgressBarLogger

LOGGERS.register_module(module=ConsoleLogger)
LOGGERS.register_module(module=CSVLogger)
LOGGERS.register_module(module=WandbLogger)
LOGGERS.register_module(module=ProgressBarLogger)
