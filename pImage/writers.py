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
from cv2 import cvtColor, COLOR_RGB2BGR, COLOR_HSV2BGR, COLOR_BGRA2BGR, COLOR_RGBA2BGR

def select_extension_writer(file_path):
    if os.path.splitext(file_path)[1] == ".avi" :
        return AviWriter
    if os.path.splitext(file_path)[1] == ".mp4" or os.path.splitext(file_path)[1] == ".m4v" :
        return MP4Writer
    if os.path.splitext(file_path)[1] == ".mkv" :
        return MKVWriter
    if os.path.splitext(file_path)[1] == ".tiff" :
        return TiffWriter
    if os.path.splitext(file_path)[1] == ".gif" :
        return GifWriter
    else :
        raise NotImplementedError("File extension/CODEC not supported yet")

class VideoWriter:
    """
    This class, based on the file path extension provided, automatically selects and instantiates the appropriate video writer class.
    """
    def __new__(cls,path,**kwargs):
        selected_writer_class = select_extension_writer(path)
        return selected_writer_class(path,**kwargs)

class DefaultWriter:
    """
    This is the base video writer class, providing a common interface and flow for video writing. Specific video writer classes can inherit from this and override methods as needed.
    """
    ############## Methods that needs to be overriden :      
    def __init__(self,**kwargs):
        pass
    
    def _write_frame(self,array):
        raise NotImplementedError
        #implement the actual writing of one frame to the file output
        
        
    ############## Methods to overrride if necessary :  
    def open(self):
        pass
    
    def close(self):
        pass
    
    ############## Methods to keep :      
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.close()
        
    def write_from(self,frame_yielder):
        """
        writes to disk all available frames from a yielder.
        The yielder can be anything providing valid image data, (in a consistant manner from the first frame to the last the writer recieved)
        It's intended to be used specifically with a reader (reader = pImage.VideoReader(videopath)) and method reader.frames() (in this case it gives all the frames available)
        or reader.sequence(start,stop) (in this case it gives frames between frame 'start' and frame 'stop' supplied as integers)
        """
        def activity_bar(l_index):#just a subclass to handle an activitybar made of small dots "moving", to see if the process crashes.
            nonlocal msg
            if l_index%10 == 0:
               if len(msg)> 6:
                   print('       ',end='\r')
                   msg = ''
               else :
                   msg += '.'
                   print(msg,end='\r')  
                   
        msg = ''  
        print("Writing")
        for index, frame in enumerate(frame_yielder) :
            activity_bar(index)
            self.write(frame)
        print("Writing finished")
        
    def write(self,array):
        self._write_frame(array)

class OpenCVWriter(DefaultWriter):
    """
    OpenCVWriter inherits from DefaultWriter and uses OpenCV VideoWriter to create videos. The writer can be customized using various parameters provided as kwargs while initializing the writer.
    """
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
        path = os.path.abspath(path)
        if not os.path.isdir(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
            
        
        self.path = path
        self.rgbconv = kwargs.get("rgbconv","RGB2BGR")
        # /!\ : if data from matplotlib exported arrays : color layers are not right
        #necessary to add parameter  rgbconv = "RGB2BGR"

        self.fps = kwargs.get("fps", 30)
        self.codec = kwargs.get("codec", "MJPG")
        self.dtype = kwargs.get("dtype", 'uint8')

        self.fourcc = VideoWriter_fourcc(*self.codec)
        
        self.file_handle = None
        
    def _write_frame(self,array):
        
        if self.file_handle is None :
            
            self.size = np.shape(array)[1], np.shape(array)[0]
            self.file_handle = VideoWriter(self.path, self.fourcc, self.fps, self.size, True)#color is always True because...

        frame = array.astype(self.dtype)
        if len(frame.shape) < 3 :
            frame = np.repeat(frame[:,:,np.newaxis],3,axis = 2)#...I just duplicate 3 times gray data if it isn't
        elif self.rgbconv is not None :
            frame = eval( f"cvtColor(frame, COLOR_{self.rgbconv})" )
        self.file_handle.write(frame)

    def close(self):
        if self.file_handle is None :
            warnings.warn("No data has been given, video was not created")
            return 
        self.file_handle.release()
        
class AviWriter(OpenCVWriter):
    """
    AviWriter inherits from OpenCVWriter, and is specifically for writing .avi video files. It provides less quality reduction and results in relatively larger files.
    Has the huge advantage of being able to terminate file almost propery if not closed.
    """
    def __init__(self,path,**kwargs):
        super().__init__(path,**kwargs)
        self.codec = kwargs.get("codec", "MJPG")
        self.fourcc = VideoWriter_fourcc(*self.codec)
        
class MP4Writer(OpenCVWriter):
    """
    MP4Writer inherits from OpenCVWriter, and is specifically for writing .mp4 or .m4v video files.
    """
    def __init__(self,path,**kwargs):
        super().__init__(path,**kwargs)
        self.codec = kwargs.get("codec", "mp4v")
        self.fourcc = VideoWriter_fourcc(*self.codec)

class MKVWriter(OpenCVWriter):
    """
    MKVWriter inherits from OpenCVWriter, and is specifically for writing .mkv video files. The DIVX codec used by this writer provides high disk-space efficiency but might result in quality loss.
    Use for presentations with light videos. Avoid using for data analysis.
    """
    def __init__(self,path,**kwargs):
        super().__init__(path,**kwargs)
        self.codec = kwargs.get("codec", "DIVX")
        self.fourcc = VideoWriter_fourcc(*self.codec)

class TiffWriter(DefaultWriter):
    """
    TiffWriter inherits from DefaultWriter, and is specifically for writing .tiff image files. Notably, it uses libtiff to create single frame TIFF images.
    """
    
    #libtiff is broken. Do not use for now, need to find a replacement at some point. 
    try :
        from tifffile import TiffWriter
    except ImportError as e:
        warnings.warn("Could not import tifffile. You will not be able to use TiffWriter")

    def __init__(self,path,**kwargs):
        self.path =  os.path.dirname(path)
        self.file_prefix = os.path.splitext(os.path.basename(path))[0]
        self.index = 0
        
    def _make_full_fullpath(self,index):
        if not os.path.isdir(self.path) :
            os.makedirs(self.path)
        return os.path.join(self.path,self.file_prefix + f"_{str(index).zfill(5)}.tiff")
        
    def _write_frame(self,array):
        _fullpath = self._make_full_fullpath(self.index)
            
        with self.TiffWriter(_fullpath) as tiff_writer: 
            tiff_writer.write(array)
            self.index += 1

class GifWriter(DefaultWriter):
    """
    GifWriter inherits from DefaultWriter, and is specifically for creating GIFs. This writer accumulates frames and stores them in a GIF format. It allows customization such as framerate, seamless looping, optimization, etc.
    """  
    try :
        from PIL import Image as pillow_image
    except ImportError as e:
        warnings.warn("Could not import pillow. You will not be able to use GifWriter")

    def __init__(self,path,**kwargs):
        self.path = path
        self.frame_bank = []
        self.seamless = kwargs.get("seamless",False)
        self.frameduration = 1000/kwargs.get("framerate",33.3)
        self.optimize =  kwargs.get("optimize",True)
        self.infinite = kwargs.get("infinite",True)
        
    def _write_frame(self,array):
        
        self.frame_bank.append(self.pillow_image.fromarray(array))

    def flush(self):
        start = 0
        stop = len(self.frame_bank)
        img0 = self.frame_bank[start]
        if self.seamless :
            rest_of_imgs = self.frame_bank[start+1:stop]+self.frame_bank[stop:start:-1]
        else :
            rest_of_imgs = self.frame_bank[start+1:stop]
        
        img0.save(self.path, 
            save_all = True, 
            append_images=rest_of_imgs,
            duration=self.frameduration, 
            loop=self.infinite, 
            optimize = self.optimize)

    def close(self):
        self.flush()
        



