"""
Created on Tue Oct 12 18:54:37 2021
@author: Timothe
"""

from .video.converters import *

import warnings


def _warn_deprecated(name):
    warnings.warn(
        f"You are importing '{name}' from 'pImage.converters'. This location is deprecated. "
        "Please import from 'pImage.video.converters' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name):
    _warn_deprecated(name)
    # Try to get the attribute from the real module
    from .video import converters as _video_converters_mod

    try:
        return getattr(_video_converters_mod, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
