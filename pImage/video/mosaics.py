import os
import math
import numpy as np
import cv2
import tempfile, secrets
import warnings
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple, Any, Callable, TypeVar, List
from abc import ABC, abstractmethod

try:
    import zarr
    from zarr import Array as ZarrArray
except ImportError:
    zarr = None

from .readers import DefaultReader

Y = TypeVar("Y", bound=int)
X = TypeVar("X", bound=int)


class InterpolationModes(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    INTER_MAX = cv2.INTER_MAX
    FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    INVERSE_MAP = cv2.WARP_INVERSE_MAP
    RELATIVE_MAP = cv2.WARP_RELATIVE_MAP

    @classmethod
    def _missing_(cls, value):
        """Method making able to do InterpolationModes("Linear") and get the value, for example."""
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


def ensure_frame_has_color_layers(
    frame: np.ndarray[tuple[Y, X], np.dtype[Any]], color=True
) -> np.ndarray[tuple[Y, X, Literal[4]], np.dtype[np.uint8]]:
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[-1] == 1):
        # Convert grayscale to 3-channel
        frame = np.repeat(frame[..., np.newaxis], 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 3:
        # in this case, the data is already 2D + 3 layers color, just passing this check
        pass
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        # In case it's an RGBA (alpha) layers array, then we remove the alpha layer,
        # as we can't have alpha for most video formats
        frame = frame[..., :3]
    else:
        raise ValueError(f"Unsupported array shape : {frame.shape}")
    return frame.astype(np.uint8)


class VignetteObject(ABC):
    """
    Unified API for video/image/frame sources.
    Supports: np.ndarray, readers.DefaultReader, VignetteBuilder, VariableFrameProvider, and optionally zarr arrays.
    Handles static frames, memory/disk buffering, and color conversion.
    """

    object_type: str = "not_set"
    static_mode: bool = False
    object: Any

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Returns the Y,X shape of the array"""

    @abstractmethod
    def get_max_frame_index(self) -> int | None:
        """Returns the maximum frame index, if able to get it, or None otherwise"""

    @property
    @abstractmethod
    def has_color_layers(self) -> bool:
        """Returns true if the object contains colored arrays"""

    @abstractmethod
    def get_frame(self, frame_id: int) -> np.ndarray:
        """Returns the frame corresponding to the provided index"""

    def __init__(self, object: Any, parent: "VignetteBuilder | None" = None):  # , **variables):
        self.parent: "VignetteBuilder | None" = parent
        self.object = object

    @property
    def duration(self) -> int | None:
        """Duration in frames, None if the duration is infinite (example, in static_mode, of function type)"""
        return None if self.static_mode else self.get_max_frame_index()

    def set_static(self, static_mode):
        self.static_mode = static_mode

    def close(self):
        pass

    def __str__(self):
        return f"{type(self).__name__} Type: {self.object_type}"

    def __repr__(self):
        return self.__str__()

    @property
    def bg_color(self):
        return getattr(self, "bg_color", self.parent.bg_color)

    @classmethod
    def auto_type_from_array(cls, item: Any, parent: "VignetteBuilder", **kwargs) -> "VignetteObject":
        match item:
            case np.ndarray():
                if parent.buffer_mode == "disk" and zarr is not None:
                    selected_cls = ZarrVignetteObject
                else:
                    selected_cls = ArrayVignetteObject
            case ZarrArray():
                selected_cls = ZarrVignetteObject
            case DefaultReader():
                selected_cls = ReaderVignetteObject
            case VignetteBuilder():
                selected_cls = BuilderVignetteObject
            case callable():
                selected_cls = FunctionVignetteObject
            case _:
                raise TypeError(f"Unsupported type {item.__class__.__name__}")
        return selected_cls(item, parent, **kwargs)


class ArrayBasedVignetteObject(VignetteObject):
    time_dimension: int

    def __init__(self, object: Any, parent: "VignetteBuilder | None" = None, /, time_dimension: int = 0):
        self.time_dimension = time_dimension
        super().__init__(object, parent)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(dim for i, dim in enumerate(self.object.shape) if i != self.time_dimension)[:2]

    @property
    def has_color_layers(self) -> bool:
        return len(tuple(dim for i, dim in enumerate(self.object.shape) if i != self.time_dimension)) == 3

    def get_frame_slice(self, frame_id: int) -> Tuple[int, ...]:
        index = [slice(None)] * self.object.ndim
        index[self.time_dimension] = frame_id
        return tuple(index)

    def get_max_frame_index(self) -> int:
        return self.object.shape[self.time_dimension]

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self.static_mode:
            # if static mode, we just return the array, if 2d, it will be made 3D (color layers),
            # and if 3D, we use colors as provided by the user.
            frame = self.object
        elif 0 <= frame_id <= self.get_max_frame_index():
            frame = self.object[self.get_frame_slice(frame_id)]
        else:
            frame = np.ones(self.shape, dtype=np.uint8) * self.bg_color
        return ensure_frame_has_color_layers(frame)


class ArrayVignetteObject(ArrayBasedVignetteObject):
    object_type = "array"
    object: np.ndarray


class ZarrVignetteObject(ArrayBasedVignetteObject):
    object_type = "zarr"
    object: "ZarrArray"
    zarr_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        object: Any,
        parent: "VignetteBuilder | None" = None,
        /,
        time_dimension: int = -1,
        zarr_kwargs: dict[str, Any] = {},
    ):
        if not isinstance(object, ZarrArray):
            object = self.make_array_into_zarr(object, **zarr_kwargs)
        self.zarr_kwargs = zarr_kwargs
        super().__init__(object, parent, time_dimension=time_dimension)

    def close(self):
        # Remove zarr buffer directory
        zarr_path = self.zarr_kwargs.get("zarr_path")
        if not zarr_path:
            return
        zarr_dir = Path(zarr_path).parent
        try:
            import shutil

            shutil.rmtree(zarr_dir)
        except (ImportError, OSError):
            pass

    @classmethod
    def make_array_into_zarr(cls, array: np.ndarray, **zarr_kwargs):
        # Buffer array to disk using zarr
        tmpdir = tempfile.mkdtemp()
        random_hex = secrets.token_hex(3)
        zarr_path = os.path.join(tmpdir, f"buffer_{random_hex}.zarr")
        zarr_kwargs.update(dict(store=zarr_path, mode="w", shape=array.shape, dtype=array.dtype))
        zarr_buffer = zarr.open(**zarr_kwargs)
        zarr_kwargs["zarr_path"] = zarr_path
        zarr_buffer[:] = array
        return zarr_buffer


class ReaderVignetteObject(VignetteObject):
    object_type = "reader"
    object: "DefaultReader"

    def get_max_frame_index(self) -> int:
        return self.object.frames_number - 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.height, self.width)

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self.static_mode:
            # if static mode, we just return the array, if 2d, it will be made 3D (color layers),
            # and if 3D, we use colors as provided by the user.
            frame = self.object.frame(0)
        elif 0 <= frame_id <= self.get_max_frame_index():
            frame = self.object.frame(frame_id)
        else:
            frame = np.ones(self.shape, dtype=np.uint8) * self.bg_color
        return ensure_frame_has_color_layers(frame)

    @property
    def has_color_layers(self) -> bool:
        return self.object.color

    def close(self):
        self.object.close()


class FunctionVignetteObject(VignetteObject):
    object_type = "function"
    object: "Callable[[int], np.ndarray]"
    maximum_index: int | None

    def __init__(self, object: Any, parent: "VignetteBuilder | None" = None, /, maximum_index=None):
        self.maximum_index = maximum_index
        super().__init__(object, parent)

    @property
    def shape(self) -> Tuple[int, ...]:
        if hasattr(self, "_shape"):
            self._shape = self.get_frame(0).shape[:2]
        return self._shape

    @property
    def has_color_layers(self) -> bool:
        if hasattr(self, "_has_color_layers"):
            self._has_color_layers = len(self.get_frame(0).shape) == 3
        return self._has_color_layers

    def get_frame(self, frame_id: int) -> np.ndarray:
        return self.object(frame_id)

    def get_max_frame_index(self) -> int | None:
        return self.maximum_index


class BuilderVignetteObject(VignetteObject):
    object_type = "builder"
    object: "VignetteBuilder"

    def __init__(self, object: "VignetteBuilder", parent: "VignetteBuilder | None" = None):
        super().__init__(object, parent)
        object.set_parent(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        self.object.ensure_background_ready()
        return (
            self.object.background_height,
            self.object.background_width,
        )

    @property
    def has_color_layers(self) -> bool:
        return True

    def get_max_frame_index(self):
        return self.object.get_total_duration() - 1

    def get_frame(self, frame_id: int) -> np.ndarray:
        if 0 <= frame_id <= self.object.get_total_duration():
            return self.object.frame(frame_id)
        return self.object.get_background()


class VignetteBuilder:
    """
    Video mosaic/composite builder.
    Supports RAM/disk buffering, grid/snappy layouts, and efficient frame access.
    """

    background_height: int
    background_width: int
    v_objects: List[VignetteObject]
    time_offsets: List[int]
    sampling_rate_multipliers: List[float]
    post_transforms: "List[Callable[[VignetteBuilder, np.ndarray], np.ndarray] | None]"
    parent = None

    def __init__(self, layout_style="grid", buffer_mode="ram", zarr_kwargs=None, **kwargs):
        self.v_objects = []
        self.time_offsets = []
        self.sampling_rate_multipliers = []
        self.post_transforms = []
        self.layout_args: tuple[Any, ...] = ()
        self.layout_kwargs: dict[str, Any] = {}
        self.buffer_mode = buffer_mode
        self.zarr_kwargs = zarr_kwargs or {}

        if layout_style is not None:
            self.set_layout(layout_style, **kwargs)

        self.set_max_size(**kwargs)
        self.set_border(**kwargs)
        self.set_padding(**kwargs)
        self.set_bg_color(**kwargs)
        self.set_target_aspect_ratio(**kwargs)
        self.set_first_object(**kwargs)
        self.set_fit_to(**kwargs)
        self.set_resize_algorithm(**kwargs)

        self._layout_ready = False
        self._duration_ready = False

    # --- Setters ---

    def set_parent(self, parent: BuilderVignetteObject):
        self.parent = parent

    def set_layout(self, layout_style, *args, **kwargs):
        self.layout = layout_style
        self.layout_args = args
        self.layout_kwargs = kwargs
        self._layout_ready = False

    def set_max_size(self, max_size=1000, **extras):
        self.maxwidth = self.maxheight = max_size
        self._layout_ready = False

    def set_border(self, border=0, **extras):
        self.border = border
        self._layout_ready = False

    def set_padding(self, padding=0, **extras):
        self.padding = padding
        self._layout_ready = False

    def set_bg_color(self, bg_color=0, **extras):
        self.bg_color = bg_color

    def set_resize_algorithm(self, resize_algorithm=InterpolationModes.AREA.value, **extras):
        try:
            resize_algorithm = InterpolationModes(resize_algorithm).value
        except ValueError:
            pass
        self.resize_algorithm = resize_algorithm

    def set_first_object(self, first_object=0, **extras):
        self._f_o = first_object
        self.layout_kwargs.update({"first_object": first_object})
        self._layout_ready = False

    def set_fit_to(self, fit_to=0, **extras):
        self._fit_to = fit_to
        self.layout_kwargs.update({"fit_to": fit_to})
        self._layout_ready = False

    def set_target_aspect_ratio(self, target_ar=16 / 9, **extras):
        self.target_aspect_ratio = target_ar
        self.layout_kwargs.update({"target_ar": target_ar})

    # --- Deprecated setters ---

    def fit_to(self, fit_to):
        warnings.warn(
            "fit_to is deprecated, use set_fit_to instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_fit_to(fit_to)

    # def add_border(self, width):
    #     warnings.warn(
    #         "add_border is deprecated, use set_border instead",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     self.set_border(width)

    # def add_padding(self, thickness):
    #     warnings.warn(
    #         "add_padding is deprecated, use set_padding instead",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     self.set_padding(thickness)

    # --- Main API ---

    def add_video(
        self,
        obj,
        array_mode=None,
        sampling_rate_multiplier=1,
        time_offset: int = 0,
        transform_func=None,
        buffer_mode=None,
        static_mode=False,
        **kwargs,
    ):
        """
        Add a video/image/array/provider/builder to the mosaic.
        buffer_mode: 'ram', 'disk' (overrides builder default)
        """
        array_mode = kwargs.pop("array_mode", array_mode)
        buffer_mode = buffer_mode or self.buffer_mode

        # Handle array_mode for static/2D/3D/4D arrays
        if array_mode == "np_bwt":
            obj = np.repeat(obj[..., np.newaxis], 3, axis=-1)
        if array_mode == "np_bw":
            obj = np.repeat(obj[:, :, np.newaxis], 2, axis=2)
            obj = np.repeat(obj[..., np.newaxis], 3, axis=-1)
            static_mode = True
        if array_mode == "np_col":
            obj = np.repeat(obj[:, :, np.newaxis, :], 2, axis=2)
            static_mode = True

        self.sampling_rate_multipliers.append(sampling_rate_multiplier)
        self.time_offsets.append(time_offset)
        self.post_transforms.append(transform_func)

        vobj = VignetteObject.auto_type_from_array(item=obj, parent=self, **kwargs)
        vobj.set_static(static_mode)
        self.v_objects.append(vobj)

        self._layout_ready = False
        self._duration_ready = False

    def frame(self, index):
        background = self.get_background()
        if self.layout == "grid":
            return self._grid_frame_getter(background, index)
        elif self.layout == "snappy":
            return self._snappy_frame_getter(background, index)
        else:
            raise ValueError("Unknown layout style")

    def frames(self):
        total_time = self.get_total_duration()
        for time_index in range(total_time):
            yield self.frame(time_index)

    def close(self):
        for array in self.v_objects:
            try:
                array.close()
            except Exception:
                pass

    def get_total_duration(self):
        if not self._duration_ready:
            self._calculate_time_offsets()
        return self._total_duration

    # --- Internal layout methods ---

    def get_layout_factory(self):
        if self.layout == "grid":
            return self._apply_grid_layout
        elif self.layout == "snappy":
            return self._apply_snappy_layout
        else:
            raise ValueError("Unknown layout style")

    def _apply_grid_layout(self, columns=None, lines=None, target_ar=None, **extras):
        video_count = len(self.v_objects)
        ratios = []

        if target_ar is not None:
            self.set_target_aspect_ratio(target_ar)

        if columns is not None or lines is not None:
            if columns is not None and lines is None:
                self.columns = columns
                self.lines = math.ceil(video_count / self.columns)
            elif lines is not None and columns is None:
                self.lines = lines
                self.columns = math.ceil(video_count / self.lines)
            else:
                self.columns = columns
                self.lines = lines
                if self.columns * self.lines < video_count:
                    raise ValueError(
                        f"Cannot set {self.columns} columns and {self.lines} lines for {video_count} videos"
                    )
        else:
            for columns in range(1, video_count + 1):
                lines = math.ceil(video_count / columns)
                aspectratio = (self.v_objects[0].shape[1] * columns) / (self.v_objects[0].shape[0] * lines)
                ratios.append(abs(aspectratio / self.target_aspect_ratio - 1))
            self.columns = ratios.index(min(ratios)) + 1
            self.lines = math.ceil(video_count / self.columns)

    def _apply_snappy_layout(self, alignment="hori", first_object=None, fit_to=None, **extras):
        if len(self.v_objects) > 2:
            raise ValueError("Cannot snap more than two objects with one vignette builder. Nest builders for more.")
        if first_object is not None:
            self.set_first_object(first_object)
        if fit_to is not None:
            self.set_fit_to(fit_to)
        if alignment in ("hori", "horizontal"):
            self.lines = 1
            self.columns = 2
        elif alignment in ("vert", "vertical"):
            self.lines = 2
            self.columns = 1
        else:
            raise ValueError("Unknown alignment for snappy layout")

    def get_frame_time_index(self, object_index, frame_index):
        sampling_rate_multiplier = self.sampling_rate_multipliers[object_index]
        time_offset = self.get_time_offset(object_index)
        return int(np.round((frame_index + time_offset) / sampling_rate_multiplier))

    def get_frame_location(self, index):
        col = index % self.columns
        lin = index // self.columns
        return col, lin

    def get_frame_coordinates(self, index):
        col, lin = self.get_frame_location(index)
        x = self.frames_xorigin + (lin * self.padding) + (lin * self.frameheight)
        y = self.frames_yorigin + (col * self.padding) + (col * self.framewidth)
        return x, y, x + self.frameheight, y + self.framewidth

    def _create_snappy_bg(self):
        adjust_index = self._fit_to
        if self.lines == 1:
            real_height = self.v_objects[adjust_index].shape[0]
            self.frameheight = real_height
            scale_multiplier = self.frameheight / self.v_objects[1 - adjust_index].shape[0]
            self.framewidth = math.ceil(scale_multiplier * self.v_objects[1 - adjust_index].shape[1])
            self.background_height = self.frameheight
            self.background_width = self.framewidth + self.v_objects[adjust_index].shape[1]
        elif self.columns == 1:
            real_width = self.v_objects[adjust_index].shape[1]
            self.framewidth = real_width
            scale_multiplier = self.framewidth / self.v_objects[1 - adjust_index].shape[1]
            self.frameheight = math.ceil(scale_multiplier * self.v_objects[1 - adjust_index].shape[0])
            self.background_width = self.framewidth
            self.background_height = self.frameheight + self.v_objects[adjust_index].shape[0]
        else:
            raise ValueError("At least one dimension must be equal to 1")
        self.frames_xorigin = self.frames_yorigin = 0
        self._make_spacings()
        self._layout_ready = True

    def _create_grid_bg(self):
        real_width = self.columns * self.v_objects[0].shape[1]
        real_height = self.lines * self.v_objects[0].shape[0]
        self.frames_xorigin = self.frames_yorigin = 0
        if real_width > self.maxwidth or real_height > self.maxheight:
            if real_width / self.maxwidth > real_height / self.maxheight:
                width = self.maxwidth
                height = math.ceil(real_height / (real_width / self.maxwidth))
            else:
                height = self.maxheight
                width = math.ceil(real_width / (real_height / self.maxheight))
        else:
            width = real_width
            height = real_height
        self.framewidth = math.ceil(width / self.columns)
        self.frameheight = math.ceil(height / self.lines)
        self.background_width = self.framewidth * self.columns
        self.background_height = self.frameheight * self.lines
        self._make_spacings()
        self._layout_ready = True

    def _make_spacings(self):
        self.background_height += (self.lines - 1) * self.padding
        self.background_width += (self.columns - 1) * self.padding
        self.background_height += 2 * self.border
        self.background_width += 2 * self.border
        self.frames_xorigin += self.border
        self.frames_yorigin += self.border

    def get_background_factory(self):
        self.get_layout_factory()(*self.layout_args, **self.layout_kwargs)
        if self.layout == "snappy":
            return self._create_snappy_bg
        elif self.layout == "grid":
            return self._create_grid_bg
        else:
            raise ValueError("layout type not understood")

    def ensure_background_ready(self):
        if not self._layout_ready:
            self.get_background_factory()()

    def get_background(self):
        self.ensure_background_ready()
        return np.ones((self.background_height, self.background_width, 3), dtype=np.uint8) * self.bg_color

    def _calculate_time_offsets(self):
        min_value: int = min(self.time_offsets)
        self._positive_offsets: list[int] = [offset + min_value for offset in self.time_offsets]
        videos_ends = [
            self.v_objects[i].duration + self._positive_offsets[i]
            for i in range(len(self.v_objects))
            if self.v_objects[i].duration is not None
        ]
        if not videos_ends:
            videos_ends = [1]
        self._total_duration = max(videos_ends)
        self._duration_ready = True

    def get_time_offset(self, object_index):
        if not self._duration_ready:
            self._calculate_time_offsets()
        return -self._positive_offsets[object_index]

    def _get_object_frame(self, object_index, frame_index):
        return self.v_objects[object_index].get_frame(self.get_frame_time_index(object_index, frame_index))

    def _snappy_frame_getter(self, frame, index):
        f_o = self._f_o
        x, y = self.frames_xorigin, self.frames_yorigin
        if self._fit_to != f_o:
            patch = cv2.resize(
                self._get_object_frame(f_o, index),
                (self.framewidth, self.frameheight),
                interpolation=self.resize_algorithm,
            )
        else:
            patch = self._get_object_frame(f_o, index)
        ex, ey = x + patch.shape[0], y + patch.shape[1]
        frame[x:ex, y:ey, :] = self._process_patch(patch, f_o)
        col, lin = self.get_frame_location(1 - f_o)
        x2 = x + (patch.shape[0] * lin) + (lin * self.padding)
        y2 = y + (patch.shape[1] * col) + (col * self.padding)
        if self._fit_to == f_o:
            patch2 = cv2.resize(
                self._get_object_frame(1 - f_o, index),
                (self.framewidth, self.frameheight),
                interpolation=self.resize_algorithm,
            )
        else:
            patch2 = self._get_object_frame(1 - f_o, index)
        ex2, ey2 = x2 + patch2.shape[0], y2 + patch2.shape[1]
        frame[x2:ex2, y2:ey2, :] = self._process_patch(patch2, 1 - f_o)
        return frame

    def _process_patch(self, patch: np.array, index: int):
        transform_function = self.post_transforms[index]
        if transform_function is None:
            return patch
        return transform_function(self, patch)

    def _grid_frame_getter(self, frame, index):
        resize_arrays = []
        for i in range(len(self.v_objects)):
            _fullsizevig = self._get_object_frame(i, index)
            resize_arrays.append(
                cv2.resize(
                    _fullsizevig,
                    (self.framewidth, self.frameheight),
                    interpolation=self.resize_algorithm,
                )
            )
        for i, patch in enumerate(resize_arrays):
            x, y, ex, ey = self.get_frame_coordinates(i)
            patch = self._process_patch(patch, i)
            frame[x:ex, y:ey, :] = patch
        return frame


def make_title_band(array, title, height=50, bg_color=0, **kwargs):
    from transformations import annotate_image

    title_band = np.ones((height, array.shape[1], 3), dtype=array.dtype)
    title_band = annotate_image(title_band * bg_color, title, **kwargs)
    return title_band


# Example usage (RAM vs disk buffering):
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate 30 videos, each 1000x1000, 10 frames
    videos = [np.random.randint(0, 255, (10, 1000, 1000, 3), dtype=np.uint8) for _ in range(30)]

    # RAM mode (fast, high memory)
    vb_ram = VignetteBuilder(buffer_mode="ram")
    for vid in videos:
        vb_ram.add_video(vid)
    vb_ram.set_layout("grid")
    frame = next(vb_ram.frames())
    plt.imshow(frame)
    plt.title("RAM mode")
    plt.show()
    vb_ram.close()

    # Disk mode (low memory, slower, requires zarr)
    if zarr is not None:
        vb_disk = VignetteBuilder(buffer_mode="disk")
        for vid in videos:
            vb_disk.add_video(vid)
        vb_disk.set_layout("grid")
        frame = next(vb_disk.frames())
        plt.imshow(frame)
        plt.title("Disk mode (zarr)")
        plt.show()
        vb_disk.close()
