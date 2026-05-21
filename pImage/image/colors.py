import matplotlib.colors as mcolors
from matplotlib.axes import Axes
import numpy as np
from dataclasses import dataclass, field
from typing import Sized

__all__ = ["Colormap", "colorize_grayscale_image", "colorize_grayscale_image_stack", "get_default_stack_colors"]


@dataclass
class Colormap:
    """Returns a 256x3 array mapping grayscale [0,255] to an RGB color gradient from black to the given color."""

    color: str
    lut: np.ndarray = field(init=False)

    def __post_init__(self):
        rgb = np.array(mcolors.to_rgb(self.color))
        ramp = np.linspace(0, 1, 256)[:, None] * rgb[None, :]
        self.lut = (ramp * 255).astype(np.uint8)

    def maps(self, image: np.ndarray) -> np.ndarray:
        return self.lut[image]


STACK_COLORS = [
    ["white"],  # 1 image
    ["lime", "magenta"],  # 2 images
    ["red", "lime", "blue"],  # 3 images
    ["orangered", "lime", "blue", "deeppink"],  # 4 images (max readable)
]


def get_default_stack_colors(iterable: Sized) -> list[str]:
    """
    Return a list of n visually distinct color names (matplotlib-compatible).
    For n <= 10, uses a hand-picked palette. For larger n, uses a matplotlib colormap.
    """
    if not len(iterable):
        return []
    if len(iterable) > len(STACK_COLORS):
        raise ValueError(
            f"More than supported {len(STACK_COLORS)} colors required by the length of the iterable ({len(iterable)})"
        )
    return STACK_COLORS[len(iterable) - 1]


def colorize_grayscale_image(image, color_name: str):
    """
    Map a single-channel 8-bit image to RGB using the given color name.
    """
    colormap = Colormap(color_name)
    return colormap.maps(image)


def colorize_grayscale_image_stack(
    images: list | np.ndarray,
    colors: list[str] | None = None,
    auto_expose: bool = True,
    ax: Axes | None = None,
    default_image_size=(512, 512),
):
    """
    images: list/array of grayscale images (all same shape)
    colors: list of color names (matplotlib color names or hex)
    auto_expose: if True, rescale each image to 8-bit using its min/max
    Returns: RGB image (H, W, 3), dtype=uint8
    """

    if colors is None:
        colors = get_default_stack_colors(images)
    images: list[np.ndarray] = [np.asarray(img) for img in images]
    assert len(images) == len(colors), "The length of colors don't match the length of images"

    if not len(images):
        # in case no image, we use an assumption that the black image resulting should be 512 * 512 (default image size)
        h, w = default_image_size
    else:
        h, w = images[0].shape
    rgb_stack = np.zeros((h, w, 3), dtype=np.float32)
    for img, color in zip(images, colors):
        img_proc = np.nan_to_num(img, nan=0.0)
        if auto_expose:
            # Use only finite values for percentile computation
            finite_vals = img_proc[np.isfinite(img_proc)]
            if finite_vals.size > 0:
                vmin, vmax = (
                    np.percentile(finite_vals, 1),
                    np.percentile(finite_vals, 99),
                )
                if vmax > vmin:
                    img8_f = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255)
                else:
                    # all finite values in the array are equal
                    img8_f = np.clip(img_proc, 0, 255)
            else:
                img8_f = np.zeros_like(img_proc, dtype=np.float32)
        else:
            img8_f = np.clip(img, 0, 255)
        img8 = np.nan_to_num(img8_f, nan=0.0).astype(np.uint8)
        rgb = colorize_grayscale_image(img8, color).astype(np.float32)
        rgb_stack += rgb
    rgb_stack = np.clip(rgb_stack, 0, 255).astype(np.uint8)
    if ax is not None:
        ax.imshow(rgb_stack, zorder=1, interpolation="none")
    return rgb_stack
