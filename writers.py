# -*- coding: utf-8 -*-

"""Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Tue Oct 12 18:54:37 2021
@author: Timothe
</div>
"""

import os, warnings
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

try :
    import hiris
except ImportError as e :
    hiris = e


def select_writer(file_path):
    if os.path.splitext(file_path)[1] == ".avi" :
        return AviWriter
    else :
        raise NotImplementedError("File extension/CODEC not supported yet")

class AviWriter:
    def __init__(self,path,**kwargs):
        """
        Creates an object that contains all parameters to create a video,
        as specified with kwargs listed below.
        The first time object.addFrame is called, the video is actually opened,
        and arrays are written to it as frames.
        When the video is written, one can call self.close() to release
        python handle over the file or it is implicity called if used in structure :
        ```with frames_ToAVI(params) as object``` wich by the way is advised for better
        stability.

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            - fps :
                playspeed of video
            - codec :
                4 char string representing an existing CODEC installed on  your machine
                "MJPG" is default and works great for .avi files
                List of FOURCC codecs :
                https://www.fourcc.org/codecs.php
            - dtype :
            - rgbconv :

            root
            filename


        Returns
        -------
        None.

        """

        filename = kwargs.get("filename",None)
        root = kwargs.get("root",None)
        if root is not None :
            path = os.path.join(root,path)
        if filename is not None :
            path = os.path.join(path,filename)
        if not os.path.isdir(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        self.path = path

        self.rgbconv = kwargs.get("rgbconv",None)
        # /!\ : if data from matplotlib exported arrays : color layers are not right
        #necessary to add parameter  rgbconv = "RGB2BGR"

        self.fps = kwargs.get("fps", 30)
        self.codec = kwargs.get("codec", "MJPG")
        self.dtype = kwargs.get("dtype", 'uint8')

        self.fourcc = VideoWriter_fourcc(*self.codec)

    def __enter__(self):
        self.vid = None
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self.vid is None :
            warnings.warn("No data has been given, video was not created")
            return
        self.close()

    def write_frame(self,array):

        if self.vid is None :
            self.size = np.shape(array)[1], np.shape(array)[0]
            self.vid = VideoWriter(self.path, self.fourcc, self.fps, self.size, True)#color is always True because...

        frame = array.astype(self.dtype)
        if len(frame.shape) < 3 :
            frame = np.repeat(frame[:,:,np.newaxis],3,axis = 2)#...I just duplicate 3 times gray data if it isn't
        elif self.rgbconv is not None :
            frame = eval( f"cv2.cvtColor(frame, cv2.COLOR_{self.rgbconv})" )
        self.vid.write(frame)

    def close(self):
        self.vid.release()






