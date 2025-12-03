# -*- coding: utf-8 -*-
# pImage.__init__.py

__version__ = "3.0.5"
__all__ = ["interact", "mosaics", "pillow", "pillow_draw"]

from .image import *
from .converters import *
from .transformations import *
from .measurements import *
from .blend_modes import *
from . import interact
from . import mosaics

try:
    import PIL.Image as pillow
    import PIL.ImageDraw as pillow_draw
except ImportError:
    pillow = None
    pillow_draw = None

import warnings


def __getattr__(name):
    import importlib

    if name in ["readers", "writers", "converters"]:
        warnings.warn(
            f"Accessing 'pImage.{name}' is deprecated. Please use 'pImage.video.{name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module(f".video.{name}", __package__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
