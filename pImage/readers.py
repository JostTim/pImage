# -*- coding: utf-8 -*-

from .video.readers import *
from .video.readers import _readers_factory

# For retrocompatibility:
import warnings


def _warn_deprecated(name):
    warnings.warn(
        f"You are importing '{name}' from 'pImage.readers'. This location is deprecated. "
        "Please import from 'pImage.video.readers' instead.",
        DeprecationWarning,
        stacklevel=3,
    )

def __getattr__(name):
    _warn_deprecated(name)
    # Try to get the attribute from the real module
    from .video import readers as _video_readers_mod

    try:
        return getattr(_video_readers_mod, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
