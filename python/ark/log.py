# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from .core import LogLevel, log
from .error import *
from .error import __all__ as error_all

__all__ = [*error_all, "DEBUG", "INFO", "WARN"]


def DEBUG(msg: str) -> None:
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    log(LogLevel.DEBUG, info.filename, info.lineno, msg)


def INFO(msg: str) -> None:
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    log(LogLevel.INFO, info.filename, info.lineno, msg)


def WARN(msg: str) -> None:
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    log(LogLevel.WARN, info.filename, info.lineno, msg)
