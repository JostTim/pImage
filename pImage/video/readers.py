# -*- coding: utf-8 -*-
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from ffmpeg.ffmpeg import FFmpeg
from tqdm import tqdm

__all__ = [
    "VideoReader",
    "DefaultReader",
    "OpenCVReader",
    "FFmpegReader",
    "_readers_factory",
]


def _readers_factory(file_path: str, **kwargs) -> "type[DefaultReader]":
    use_ffmpeg = kwargs.get("use_ffmpeg", False)
    extension = os.path.splitext(file_path)[1]
    if use_ffmpeg:
        base = FFmpegReader
    else:
        base = OpenCVReader
    if extension == ".seq":
        try:
            from . import hiris

            return hiris.HirisReader
        except ImportError:
            raise ImportError("hiris.py not available in library folder")
    elif extension == ".avi":

        class AviReader(base):
            pass

        return AviReader
    elif extension == ".mp4":

        class MP4Reader(base):
            pass

        return MP4Reader
    else:

        class UnknownReader(base):
            pass

        return UnknownReader
        # raise NotImplementedError("File extension/CODEC not supported yet")


class VideoReader:
    # do not inherit from this class. It only returns other classes factories.
    def __new__(cls, path, **kwargs):
        from ..transformations import TransformingReader, available_transforms

        if set(kwargs.keys()).intersection(available_transforms):
            return TransformingReader(path, **kwargs)
        selected_reader_class = _readers_factory(path, **kwargs)
        return selected_reader_class(path, **kwargs)


class DefaultReader(ABC):
    path: Path
    color: bool

    ############## Methods that needs to be overriden :

    def __init__(self, path: str | Path, color: bool = False):
        self.path = Path(path)
        self.color = color

    @abstractmethod
    def _get_frame(self, frame_id: int) -> np.ndarray:
        raise NotImplementedError
        # make it return the specific frame

    @abstractmethod
    def _get_all(self):
        raise NotImplementedError
        # make it a yielder for all frames in object (no need for indexing)

    @abstractmethod
    def _get_frames_number(self):
        raise NotImplementedError
        # returns the frames number implementing any calculation needed

    ############## Methods to override if relevant :
    def open(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    ############## Commong methods that need no reimplementation :
    @property
    def frames_number(self):
        try:
            self._frames_number
        except AttributeError:
            self._frames_number = self._get_frames_number()
        finally:
            return self._frames_number

    def sequence(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = self.frames_number

        self._check_frame_id(start)
        self._check_frame_id(stop - 1)

        for i in tqdm(range(start, stop), delay=1):
            yield self.frame(i)

    def _check_frame_id(self, frame_id):
        if frame_id < 0:
            raise ValueError("Cannot get negative frame ids")
        if self.frames_number is not None and frame_id > self.frames_number - 1:
            if os.path.isfile(self.path):
                raise IOError(f"File supplied as input does not exist : {self.path}")
            raise ValueError("Not enough frames in reader")

    def frames(self):
        yield from self._get_all()

    def frame(self, frame_id):
        self._check_frame_id(frame_id)
        return self._get_frame(frame_id)

    def __getitem__(self, index):

        try:
            time_start, time_stop = index[2].start, index[2].stop
        except (TypeError, AttributeError):
            _slice = slice(index[2], index[2] + 1)
            time_start, time_stop = _slice.start, _slice.stop
        return np.squeeze(np.moveaxis(np.array(list(self.sequence(time_start, time_stop))), 0, 2)[(index[0], index[1])])

    @property
    def width(self):
        try:
            self._width
        except AttributeError:
            shape = self.frame(0).shape
            self._height = shape[1]
            self._width = shape[0]
        finally:
            return self._width

    @property
    def height(self):
        try:
            self._height
        except AttributeError:
            shape = self.frame(0).shape
            self._height = shape[1]
            self._width = shape[0]
        finally:
            return self._height

    @property
    def shape(self):
        return (self.width, self.height, self.frames_number)


class OpenCVReader(DefaultReader):
    _internal_index: int
    _stored_frame: np.ndarray | None

    def __init__(self, path: str | Path, color: bool = False):
        if isinstance(cv2, ImportError):
            raise ImportError("OpenCV2 cannot be imported sucessfully or is not installed")
        super().__init__(path, color)
        self._internal_index = 0
        self._stored_frame = None

    def open(self):
        if not hasattr(self, "file_handle"):
            if self.color:
                self.file_handle = cv2.VideoCapture(self.path)  # ,cv2.IMREAD_COLOR )
            else:
                self.file_handle = cv2.VideoCapture(self.path, cv2.IMREAD_GRAYSCALE)
        return self.file_handle

    def close(self):
        if hasattr(self, "file_handle"):
            self.file_handle.release()

    def _get_frames_number(self):
        self.open()
        frameno = int(self.file_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        return frameno if frameno > 0 else None

    def _get_frame(self, frame_id):
        self.open()
        if (
            self._internal_index == frame_id and self._stored_frame is not None
        ):  # if we ask again for the same frame, just give it back
            return self._stored_frame
        elif self._internal_index + 1 == frame_id:
            # if we ask for next frame than before just use the fact that internal index in VideoCapture.
            # read auto increases after a call
            pass
        else:  # if we ask for a specific frame set VideoCapture internal index to get the right frame
            self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        success, temp_frame = self.file_handle.read()
        if not success:
            raise IOError("out of the frames available for this file")
        if self.color:
            self._stored_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        else:
            self._stored_frame = temp_frame[:, :, 0]

        self._internal_index = frame_id
        return self._stored_frame.copy()
        # return a copy to avoid the issues related to giving back a frame that could have beend modified
        # externally without being aware of it

    def _get_all(self):
        self.open()
        self._internal_index = -1
        self._stored_frame = None
        self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            success, temp_frame = self.file_handle.read()
            if not success:
                break
            if self.color:
                yield cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
            else:
                yield temp_frame[:, :, 0]


class FFmpegReader(DefaultReader):
    # This reader is based on video time (e.g. based on framerate metadata and ffmped seek)
    # It is usefull for long videos that are somewhat corrupted in the sense that opencv reader
    # dont' get all  the frames with cv2.CAP_PROP_FRAME_COUNT and you wish to access a part of
    # the video that lie beyond this point in the video. (this corruption often occurw when saving
    # an avi file and writer is not properly closed.)

    pix_fmt: str

    def __init__(
        self,
        path: str | Path,
        color: bool = False,
        pix_fmt: Literal["gray"] = "gray",
    ):
        super().__init__(path, color)
        self.pix_fmt = pix_fmt

    def get_framerate(self) -> float:
        ffprobe = FFmpeg(executable="ffprobe").input(
            self.path,
            print_format="json",
            show_streams=None,
        )
        media_infos = json.loads(ffprobe.execute())
        return float(media_infos["streams"][0]["r_frame_rate"].split("/")[0])

    def ffmpeg_probe(self) -> dict:
        if not hasattr(self, "_ffmpeg_probe"):
            ffprobe = FFmpeg(executable="ffprobe").input(
                self.path,
                print_format="json",
                show_streams=None,
            )
            self._ffmpeg_probe = cast(dict, json.loads(ffprobe.execute()))
        return self._ffmpeg_probe

    def _get_time_from_frameno(self, frame_no) -> str:
        return str(timedelta(seconds=frame_no / self.get_framerate()))

    def _get_frameno_from_time(self, time_str) -> int:
        # imprecise at the number of frames per second

        frame_seconds = (datetime.strptime(time_str, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds()
        return int(frame_seconds * self.get_framerate())

    def _get_height_ffmpeg(self):
        return self.ffmpeg_probe()["streams"][0]["height"]

    def _get_width_ffmpeg(self):
        return self.ffmpeg_probe()["streams"][0]["width"]

    def _get_frames_number(self):
        return np.inf

    def sequence(self, start: int | str | None = None, stop=None):
        # In str format: 'HH:MM:SS'
        if isinstance(start, int):
            start = self._get_time_from_frameno(start)
        if isinstance(stop, str):
            frame_no = self._get_frameno_from_time(stop)
            frame_nb = frame_no - self._get_frameno_from_time(start)
        else:
            frame_nb = stop

        frame_nb
        width, height = self._get_width_ffmpeg(), self._get_height_ffmpeg()
        buffer, _ = (
            FFmpeg()
            .input(self.path, ss=start)
            .option("scale", [width, -1])
            .option("ss", start)
            # ffmpeg.input(self.path, ss=start)
            # .filter("scale", width, -1)
            .output("pipe:1", format="rawvideo", pix_fmt=self.pix_fmt, vframes=frame_nb)
            .execute(capture_stdout=True, capture_stderr=True)
        )
        frames = np.frombuffer(buffer, np.uint8).reshape(frame_nb, height, width)
        for frame_index in range(frames.shape[0]):
            yield frames[frame_index]

    def _get_frame(self, frame_id):
        return next(self.sequence(frame_id, 1))

    def _get_all(self):
        raise NotImplementedError(
            "Cannot know total duration from ffmpeg reader. "
            "Use sequence to get sequence betwen start and stop duration."
        )
