# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:28:33 2020

@author: Timothe
"""

import os, sys
import numpy as np
import re, configparser
from typing import Any

from .readers import DefaultReader


class HirisReader(DefaultReader):
    def __init__(self, path):
        def files_match(input_folder, regexp):
            def bool_regexp(input_line, regex):
                matches = re.finditer(regex, input_line, re.MULTILINE)
                for matchnum, match in enumerate(matches, start=1):
                    return True
                return False

            file_list = os.listdir(input_folder)
            return [os.path.join(input_folder, file) for file in file_list if bool_regexp(file, regexp)]

        super().__init__(path)
        filename = os.path.splitext(os.path.split(path)[1])[0]

        self.seqfile = HirisSeqFile(path)
        self.dir = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
        binfiles_names = sorted(files_match(self.dir, rf"^{filename}.*\.bin$"))
        self.binfiles = [HirisBinFile(_file_path, self.seqfile) for _file_path in binfiles_names]

    def _get_frame(self, frame_id):
        index, frame_offset = self.find_frame_binfile(frame_id)
        return self.binfiles[index].get_frame(slice(frame_offset, frame_offset + 1, 1))[0]

    def _get_all(self):
        for binfile in self.binfiles:
            yield from binfile.frames()

    def _get_frames_number(self):
        return sum([binfile.frames_number for binfile in self.binfiles])

    def find_frame_binfile(self, frame_id):
        # TODO : make this more efficient so that time is constant whatever the index
        frame_count = 0
        for index, binfile in enumerate(self.binfiles):
            if frame_count <= frame_id < (frame_count + binfile.frames_number):
                return index, frame_id - frame_count
            frame_count += binfile.frames_number
        raise EOFError


class HirisSeqFile(dict):
    def __init__(self, file_path):
        """Usually has keys :
        'Sequence name',
        'CommentSize',
        'Comment',
        'Number of files',
        'Mode', 'Format',
        'Agregat',
        'Width',
        'Height',
        'BytesPerPixel',
        'BitsPerPixel',
        'Image dimension',
        'Bin directory',
        'Bin repertoire',
        'Bin File',
        'Start Time',
        'TrigType',
        'TrigTime',
        'TimeTrig',
        'FramePerSecond',
        under [Sequence Settings] section.
        """
        self.path = file_path
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(file_path)
        _tempdict: dict[str, Any] = {
            key: cfg.get(
                "Sequence Settings",
                key,
            )
            for key in cfg["Sequence Settings"].keys()
        }
        height = _tempdict.get("Height")
        width: Any | None = _tempdict.get("Width")
        bytes_per_pixel = _tempdict.get("BytesPerPixel")
        if height is None or width is None or bytes_per_pixel is None:
            raise ValueError("height, width or bytes_per_pixel are None ! Please investigate.")
        _tempdict["FrameBinSize"] = int(height) * int(width) * int(bytes_per_pixel)
        super().__init__(_tempdict)


class HirisBinFile:
    def __init__(self, file_path, seq_file):
        self.path = file_path
        self.seq_file = seq_file

    @property
    def byte_size(self):
        try:
            return self._bytesize
        except AttributeError:
            self._bytesize = os.path.getsize(self.path)
            return self._bytesize
        except Exception as e:
            raise ValueError(f"Could not read size of file {self.path}: invoked cause : {e}")

    @property
    def frame_bin_size(self):
        try:
            return self._framebinsize
        except AttributeError:
            self._framebinsize = self.seq_file.get("FrameBinSize")
            return self._framebinsize

    @property
    def remainder(self):
        return self.byte_size % self.frame_bin_size

    @property
    def frames_number(self):
        return int((self.byte_size - self.remainder) / self.frame_bin_size)

    def frames(self):
        with open(self.path, "rb") as f_bin:
            for frame_id in range(self.frames_number):
                yield self._get_frame(frame_id, f_bin)

    def get_frame(self, frame_slice):
        frames = []
        with open(self.path, "rb") as f_bin:
            for frame_id in range(frame_slice.start, frame_slice.stop, frame_slice.step):
                frames.append(self._get_frame(frame_id, f_bin))
        return frames

    def _get_frame(self, frame_id, file_object):
        offset = frame_id * self.frame_bin_size
        file_object.seek(offset, os.SEEK_SET)
        bytes_content = file_object.read(self.frame_bin_size)
        buffer = np.frombuffer(bytes_content, dtype=np.uint8)

        if len(bytes_content) < self.frame_bin_size:
            raise EOFError("Invalid slice dimension to recreate a valid image frame")

        return buffer.reshape(int(self.seq_file.get("Height")), int(self.seq_file.get("Width")))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    root = Path("F:/Timothe/DATA/BehavioralVideos/Whisker_Video/Whisker_Topview/Expect_3_mush/Mouse66/210503_1")
    test = HirisReader(root / "2021-05-03T19.19.19/Trial.seq")
    for index, frame in enumerate(test.frames()):
        print(index)
        plt.imshow(frame)
        plt.show()
