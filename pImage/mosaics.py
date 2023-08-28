#%% Definitions
import os, math
import numpy as np
import cv2

import warnings

from . import readers
    
class vignette_object():
    """
    Core component for VignettBuilder's flexibility 
    It provides a common API for acessing frames or objects, 
    regardless of the actual content class and type of the object
    
    The API is :
        - shape : returns (x,y,time) color info is not specified, implicit, see get_frame for why
        - get_frame : returns a frame. frame are ALWAYS returned in color mode, with last dimension = 3.
        - set_static : works only for objects that contains arrays for now. 
            Sets the fact that get_frame returns the same image whatever the index, for use as constant info in layouts 
        - close : self explanatory. Usefull for memaps, 
            althoug the use of memaps is strongly discouraged for now, until they get fixed both in term of ram consistency and speed
        
    """
    
    def __init__(self,_object,parent = None):
        if isinstance(_object,np.ndarray):
            self.type= "array"
        elif isinstance(_object,readers.DefaultReader):
            self.type = "reader"
        elif isinstance(_object,VignetteBuilder):
            self.type = "builder"
        elif isinstance(_object,VariableFrameProvider):
            self.type = "provider"
        else :
            raise(f"vignette object type not understood :{_object.__class__.__name__}")
        self.static_mode = False
        self.object = _object
        self.parent = parent
            
    @property
    def shape(self):
        if self.type == "memap" :
             return self.object.shape
        elif self.type == "array" :
            shape =  self.object.shape
            return (shape[2],shape[0],shape[1])
        elif self.type == "reader"  :                
            shape =  self.object.shape
            return (shape[2],shape[0],shape[1])
        elif self.type == "builder" :
            if not self.object._layout_ready:
                self.object.get_background_factory()()
            return (self.object.get_total_duration(),self.object.background_height,self.object.background_width)
        elif self.type == "provider" :
            return self.object.get_shape()
        
    def set_static(self,static_mode):
        self.static_mode = static_mode
        
    def get_frame(self,frame_id):
        if self.type == "memap" :
            if 0 > frame_id or frame_id > self.object.shape[0]-1 :
                return np.ones((self.object.shape[1],self.object.shape[2],3),dtype = np.uint8) * self.parent.bg_color
            return self.object[frame_id]
        elif self.type == "array" :
            if self.static_mode :
                return self.object[:,:,0]
            if 0 > frame_id or frame_id >= self.object.shape[2] :
                return np.ones((self.object.shape[0],self.object.shape[1],3),dtype = np.uint8)  * self.parent.bg_color
            return self.object[:,:,frame_id]
        elif self.type == "reader" :
            if 0 > frame_id or frame_id > self.object.frames_number - 1 :
                return np.ones((self.object.width,self.object.height,3),dtype = np.uint8)  * self.parent.bg_color
            #only BW readers for now
            if self.object.color is True :
                return self.object[:,:,frame_id]
            return np.repeat(self.object[:,:,frame_id][:,:,np.newaxis], 3, axis = 2)
        elif self.type == "builder" :
            if 0 > frame_id or frame_id > self.object.get_total_duration() :
                return self.object.get_background() 
            return self.object.frame(frame_id)
        elif self.type == "provider" :
            return self.object.get_frame(frame_id)
        
    def close(self):
        if self.type == "memap" :
            self.object.close()
            
    def __str__(self):
        return type(self).__name__ + " Type :" + self.type
    
    def __repr__(self):
        return vignette_object.__str__(self)


def dummy_patch_processor(vignette_builder,patch):
    return patch

class VariableFrameProvider():
    def bound(self,function):
        import types
        setattr(self,"get_frame",types.MethodType(function,self))
    
    def __init__(self,**variables):
        for key, value in variables.items():
            setattr(self,key,value)

    def get_shape(self):
        try :
            return self.get_frame(0).shape[0:2]
        except AttributeError:
            raise("No method to get frames has been bound to this instance. Be sure to call 'instance.bound(function)' on every of your instances. Function should take as input an index and return the frame accordingly")
    
class VignetteBuilder():
    """
    A class for creating video mosaics or aggregated videos from multiple video sources.

    Attributes:
        add_video (method): Adds a video to the mosaic.
        frame (method): Returns composite frame at the given index. 
        frames (method): Generator that returns each frame of the composite video.
        close (method): Closes the video objects associated with the builder.
        get_total_duration (method): Returns the total duration of video.
    """
        
    def __init__(self,layout_style = "grid", **kwargs) :
        """
        All kwargs key values are the keys of the setters.
        More info in each of them.
        """
        self.v_objects = []
        self.time_offsets = []
        self.sampling_rate_multipliers = []
        self.post_transforms = []
        self.layout_args = [(),{}]
        
        if layout_style is not None :
            self.set_layout(layout_style,**kwargs)
        
        self.set_max_size(**kwargs)
        self.set_border(**kwargs)
        self.set_padding(**kwargs)
        self.set_bg_color(**kwargs)
        
        #Usefull only if gridlayout
        self.set_target_aspect_ratio(**kwargs)
        #Usefull only if snappylayout
        self.set_first_object(**kwargs)
        self.set_fit_to(**kwargs)
        
        self.set_resize_algorithm(**kwargs)
        self._layout_ready = False
        self._duration_ready = False
        
    ### PARAMETERS SETTERS
                
    def set_layout(self, layout_style, *args, **kwargs ):
        """_summary_

        Args:
            layout_style (string): _description_
                possible values :
                - "grid"
                    in that case, accepts arguments :
                    - columns 
                    - lines
                    To force the number of lines or columns. 
                    Note that the given target_aspect_ratio will not be taken into account in that case
                    - target_ar , the target aspect ratio, defaults to 16/9
                - "snappy"
                in that case, accepts arguments :
                    - alignment 
                        with values "hori", "horizontal", "vert" or "vertical"
                        self explanatory
                    - fit_to, index of object to fit to (dimension of the object fitted will depend of alignment)
                    - first_object, index of first_object 
                       (object labelled first = left or top, so the second = right or bottom in layout)
        """        
        self.layout = layout_style
        self.layout_args = [args,kwargs]
        self._layout_ready = False
        
    def set_max_size(self, max_size = 1000,**extras):
        """_summary_

        Args:
            max_size (int, optional): _description_. Defaults to 1000.
        """
        self.maxwidth = self.maxheight = max_size
        self._layout_ready = False
        
    def set_border(self,border = 0,**extras):
        """_summary_

        Args:
            border (int, optional): _description_. Defaults to 0.
        """
        self.border = border
        self._layout_ready = False
        
    def set_spacing(self,padding = 0,**extras):
        """_summary_

        Args:
            padding (int, optional): _description_. Defaults to 0.
        """
        self.padding = padding
        self._layout_ready = False
        
    set_padding = set_spacing #alternative name for backward compatibility
        
    def set_bg_color(self,bg_color=0,**extras):
        """_summary_

        Args:
            bg_color (int, optional): _description_. Defaults to 0.
        """
        self.bg_color = bg_color
        
    def set_resize_algorithm(self,resize_algorithm = cv2.INTER_AREA ,**extras):
        """_summary_

        Args:
            resize_algorithm (_type_, optional): _description_. Defaults to cv2.INTER_AREA.
        """
        self.resize_algorithm = resize_algorithm
                
    def set_first_object(self,first_object = 0,**extras):
        """_summary_

        Args:
            first_object (int, optional): _description_. Defaults to 0.
        """
        #first_object = fist object index
        self._f_o = first_object #first_object
        self.layout_args[1].update({"first_object":first_object})
        self._layout_ready = False 
    
    def set_fit_to(self,fit_to = 0,**extras):
        """_summary_

        Args:
            fit_to (int, optional): _description_. Defaults to 0.
        """
        #fit_to : index of object to fit the other one to. The dimension that will be fited depends on shappy layout's orientation
        self._fit_to = fit_to
        self.layout_args[1].update({"fit_to":fit_to})
        self._layout_ready = False
    
    def set_target_aspect_ratio(self,target_ar = 16/9,**extras):
        self.target_aspect_ratio = target_ar
        self.layout_args[1].update({"target_ar":target_ar})
        
    ### DEPRECATED SETTERS 
        
    def fit_to(self,fit_to):
        warnings.warn('fit_to is deprecated, use set_fit_to instead (setters naming convention)', DeprecationWarning, stacklevel=2)
        self.set_fit_to(fit_to)
        
    def add_border(self,width):
        warnings.warn('add_border is deprecated, use set_border instead (setters naming convention)', DeprecationWarning, stacklevel=2)
        self.set_border(width)
        
    def add_padding(self,thickness):
        warnings.warn('add_padding is deprecated, use set_padding instead (setters naming convention)', DeprecationWarning, stacklevel=2)
        self.set_padding(thickness)


    ### MAIN EXTERNAL METHODS

    def add_video(self,_object,
                  array_mode = None,
                  sampling_rate_multiplier = 1,
                  time_offset = 0,
                  transform_func = dummy_patch_processor,
                  memmapping = False, 
                  **kwargs):
        """
        Add a video to create a mosaic. Order of addition matters for grid layout. It can be tuned for snappy layout 
        (if snappy = only 2 videos, no more, no less. One can create snappy layouts with more than 2 videos by stacking builders, see tutorial)
        
        Args:
            _object (TYPE): input.
              Takes as input :
                - any DefaultReader child classes (works with color or black&white inputs)
                - a numpy array. By default, must be a color array with time dimension.
                   Hence, dimensions must be 4, and in this specific order :
                    0 : x
                    1 : y
                    2 : time
                    3 : color 
                    One can also specify other array types , and provide kwarg ``array_mode`` with a certain flag string.
                    flags can be :
                        - np_bwt : expects blackwhite image with time 
                            3d array, dimensions as follow : 
                            0 : x
                            1 : y
                            2 : time
                        - np_bw : expects blackwhite image with no time
                            2D array, dimensions as follow : 
                            0 : x
                            1 : y
                        - np_col : expects color image with no time
                            3d array, dimensions as follow : 
                            0 : x
                            1 : y
                            2 : color
                            
                        Last two np_bw and np_col will create static images, 
                        that won't change across frames indices are called in the builder.
                - another builder. It will feed it's frames to the child builder only when asked to build them. 
                  Can be stacked as many times as RAM allows.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        set_static = False
        array_mode =  kwargs.pop("array_mode",None)
        if array_mode == "np_bwt" :#create color dimension from blackwhite image with time 
            _object = np.repeat( _object[:,:,:,np.newaxis] , 3, axis = 3 )
        if array_mode =="np_bw" : #create time then color dimension from blackwhite image with no time (only 2 frames)
            _object = np.repeat( _object[:,:,np.newaxis] , 2 , axis = 2  )
            _object = np.repeat( _object[:,:,:,np.newaxis] , 3 , axis = 3  )
            set_static = True
        if array_mode =="np_col" : #create time dimension from color image with no time (only 2 frames)
            _object = np.repeat( _object[:,:,np.newaxis,:] , 2, axis = 2 )
            set_static = True
            
        self.sampling_rate_multipliers.append( sampling_rate_multiplier )
        self.time_offsets.append( time_offset )
        self.post_transforms.append( transform_func )
        if memmapping and isinstance(_object,np.ndarray):
            raise DeprecationWarning
            #_object = array_video_color(_object,**kwargs)
        vobj = vignette_object(_object, self)
        if set_static :
            vobj.set_static(True)
        self.v_objects.append(vobj )
        self._layout_ready = False
        self._duration_ready = False
        #self.v_objects[-1].object.flush()
        
    def frame(self,index):
        background = self.get_background()
        if self.layout == "grid":
            return self._grid_frame_getter(background,index)
                
        if self.layout == "snappy" :
            return self._snappy_frame_getter(background,index)

    def frames(self):
        total_time = self.get_total_duration()
        for time_index in range(total_time):
            yield self.frame(time_index)

    def close(self):
        for array in self.v_objects:
            try :
                array.close()
            except ValueError :
                pass
        
    def get_total_duration(self):
        if self._duration_ready == False :
            self._calculate_time_offsets()
        return self._total_duration
    
    ### WORKING INTERNAL METHODS
        
    def get_layout_factory(self):
        if self.layout == "grid":
            return self._apply_grid_layout
        elif self.layout == "snappy":
            return self._apply_snappy_layout
        else :
            raise ValueError("Unknown layout style")
    


    def _apply_grid_layout(self,columns=None,lines=None,target_ar = None,**extras):
        video_count = len(self.v_objects)
        ratios = []
        
        if target_ar is not None :
            self.set_target_aspect_ratio(target_ar)
        
        if columns is not None or lines is not None :
            if columns is not None and lines is None :
                self.columns = columns
                self.lines = math.ceil(video_count/self.columns)
            elif lines is not None and columns is None :
                self.lines = lines
                self.columns = math.ceil(video_count/self.lines)
            else :
                self.columns = columns
                self.lines = lines
                if self.columns*self.lines < video_count:
                    raise ValueError("Cannot set {self.columns} columns and {self.lines} lines for a {video_count} videos mosaic")
        else :
            for columns in range(1,video_count):
                lines = math.ceil(video_count/columns)
                aspectratio = (self.v_objects[0].shape[2] * columns ) / (self.v_objects[0].shape[1] * lines )
                ratios.append( abs( aspectratio / self.target_aspect_ratio - 1 ) )
            
            try :
                self.columns = next(index for index , value in enumerate(ratios) if value == min(ratios)) + 1 
            except StopIteration :
                raise ValueError("Must have more than one video to make a grid layout. Add more videos")
            self.lines = math.ceil(video_count/self.columns)
       
        
    def _apply_snappy_layout(self,alignment = "hori",first_object = None,fit_to = None,**extras):
        if len(self.v_objects) > 2:
            raise ValueError("Cannot snap more than two objects with one vignette builder in current developpement state. Simply nest builders to achieve the desired number of snapped images.")
            
        if first_object is not None :
            self.set_first_object(first_object)
        if fit_to is not None :
            self.set_fit_to(fit_to)
            
        if alignment == "hori" or alignment == "horizontal" :
            self.lines = 1
            self.columns = 2
            
        if alignment == "vert" or alignment == "vertical" :
            self.lines = 2
            self.columns = 1
        
        
    def get_frame_time_index(self,object_index,frame_index):
        #frame index expressed in lowest sampling rate (1)
        sampling_rate_multiplier = self.sampling_rate_multipliers[object_index]
        time_offset = self.get_time_offset(object_index)#return already with a - so set + in equation below
        return int(np.round((frame_index + time_offset) / sampling_rate_multiplier))
        
    def get_frame_location(self,index):
        col = 0
        lin = 0
        irange = range(len(self.v_objects)-1,-1,-1) if self.layout == "snappy" and self._f_o else range(len(self.v_objects))
        for i in irange:
            if index == i :
                break
            col = col + 1
            if col > self.columns-1 :
                lin = lin + 1 
                col = 0
        return col, lin
    
    def get_frame_ccordinates(self,index):
        col,lin = self.get_frame_location(index)
        x = self.frames_xorigin + (lin * self.padding) + (lin * self.frameheight )
        y = self.frames_yorigin + (col * self.padding) + (col * self.framewidth )
        return x, y, x + self.frameheight, y + self.framewidth
    
    def _create_snappy_bg(self):
        # object 1 shape does not change
        # self.frameheight and self.framewidth are destined to object 2
        adjust_index = self._fit_to 
        if self.lines == 1 :
            real_height = self.v_objects[adjust_index].shape[1]
            self.frameheight =  real_height #if real_height <= self.maxheight else self.maxheight
            # TODO : change ability to fix a dimension for snappy later.
            scale_multipler = self.frameheight / self.v_objects[not adjust_index].shape[1] 
            self.framewidth = math.ceil( scale_multipler * self.v_objects[not adjust_index].shape[2])
            
            self.background_height = self.frameheight
            self.background_width = self.framewidth + self.v_objects[adjust_index].shape[2]
            
        elif self.columns == 1 :
            real_width = self.v_objects[adjust_index].shape[2]
            self.framewidth =  real_width# if real_width <= self.maxwidth else self.maxwidth
            # TODO : change ability to fix a dimension for snappy later.
            scale_multipler = self.framewidth / self.v_objects[not adjust_index].shape[2] 
            self.frameheight = math.ceil( scale_multipler * self.v_objects[not adjust_index].shape[1])
        
            self.background_width = self.framewidth
            self.background_height = self.frameheight + self.v_objects[adjust_index].shape[1]
        
        else :
            raise ValueError("At least one dimension must be equal to 1")
        
        self.frames_xorigin = self.frames_yorigin = 0
        self._make_spacings()
        self._layout_ready = True
        
    def _create_grid_bg(self):
        real_width = self.columns * self.v_objects[0].shape[2]
        real_height = self.lines * self.v_objects[0].shape[1]
        print("Real dimensions would be : ",real_width, real_height) 
        self.frames_xorigin = self.frames_yorigin = 0
        if  real_width > self.maxwidth or real_height > self.maxheight :
            if real_width / self.maxwidth > real_height / self.maxheight :
                width = self.maxwidth
                height = math.ceil(real_height / (real_width / self.maxwidth))
            else :
                height = self.maxheight
                width = math.ceil(real_width / (real_height / self.maxheight))
        else :
            width = real_width
            height = real_height
            
        self.framewidth = math.ceil( width / self.columns)
        self.frameheight = math.ceil( height / self.lines)
        
        self.background_width = self.framewidth * self.columns
        self.background_height = self.frameheight * self.lines
        
        self._make_spacings()
              
        self._layout_ready = True
        
    def _make_spacings(self):
        
        #PADDING
        self.background_height = self.background_height + ((self.lines - 1) * self.padding)
        self.background_width = self.background_width + ((self.columns - 1) * self.padding)
        
        #BORDER
        self.background_height = self.background_height + (2*self.border)
        self.background_width = self.background_width + (2*self.border)
        self.frames_xorigin = self.frames_xorigin + self.border
        self.frames_yorigin = self.frames_yorigin + self.border
        
    def get_background_factory(self):
        self.get_layout_factory()(*self.layout_args[0],**self.layout_args[1])
        
        if self.layout == "snappy" :
            return self._create_snappy_bg
        elif self.layout == "grid" :
            return self._create_grid_bg
        else :
            raise ValueError("layout type not understood")
            
    def get_background(self):
        if not self._layout_ready:
            self.get_background_factory()()
        return np.ones((self.background_height,self.background_width,3),dtype = np.uint8) * self.bg_color
        
    def _calculate_time_offsets(self):
        min_value = min(self.time_offsets)
        self._positive_offsets = [offset + min_value for offset in self.time_offsets]
        videos_ends = [self.v_objects[i].shape[0] + self._positive_offsets[i] for i in range(len(self.v_objects))]
        self._total_duration = max(videos_ends)
    
    def get_time_offset(self,object_index):
        if self._duration_ready == False :
            self._calculate_time_offsets()
        return -self._positive_offsets[object_index] 
        
    def _get_object_frame(self,object_index,frame_index):
        return self.v_objects[object_index].get_frame(self.get_frame_time_index(object_index,frame_index))
        
    def _snappy_frame_getter(self,frame,index): 
        f_o = self._f_o #first_object
        x , y = self.frames_xorigin , self.frames_yorigin
        if not self._fit_to == f_o :
            patch = cv2.resize(self._get_object_frame(f_o,index), ( self.framewidth, self.frameheight), interpolation = self.resize_algorithm)    
        else :
            patch =  self._get_object_frame(f_o,index)
        ex,ey = x + patch.shape[0] , y + patch.shape[1]
        frame[x:ex,y:ey,:] = self._process_patch(patch,f_o)
        
        col,lin = self.get_frame_location(not f_o)
        x2 = x + (patch.shape[0]*(lin)) + (lin * self.padding) 
        y2 = y + (patch.shape[1]*(col)) + (col * self.padding) 
    
        if self._fit_to == f_o :
            patch = cv2.resize(self._get_object_frame(not f_o,index), ( self.framewidth, self.frameheight), interpolation = self.resize_algorithm)
        else :
            patch = self._get_object_frame(not f_o,index)
        ex2 , ey2 = x2 + patch.shape[0], y2 + patch.shape[1]
            
        frame[x2:ex2,y2:ey2,:] = self._process_patch(patch,not f_o)
        return frame
    
    def _process_patch(self,patch,index):
         return self.post_transforms[index](self,patch) 
    
    def _grid_frame_getter(self,frame,index):
        resize_arrays = []
        for i in range(len( self.v_objects )):
            _fullsizevig = self._get_object_frame(i,index)
            resize_arrays.append(cv2.resize(_fullsizevig, (self.framewidth, self.frameheight), interpolation = self.resize_algorithm))
        
        for i in range(len( resize_arrays )):
            x,y,ex,ey = self.get_frame_ccordinates(i)
            patch = self._process_patch(resize_arrays[i],i)
            frame[x:ex,y:ey,:] = patch 
        return frame
        
def get_title_band(array,title, height = 50, bg_color = 0, **kwargs):
    """
    Create a band with the provided `title` and `bg_color` to be used as a title for an `array`. 

    Parameters:
    -----------
    array : numpy array
        The array to which the title band will be added as a title.
    title : str
        The title to be displayed in the title band.
    height : int, optional
        The height of the title band, by default 50.
    bg_color : int or tuple of ints, optional
        The background color of the title band, by default 0. 
    **kwargs : dict
        Additional keyword arguments to be passed to the `annotate_image` function.

    Returns:
    --------
    numpy array
        A title band with the provided `title`, `height` and `bg_color`.

    """
    from transformations import annotate_image
    title_band = np.ones(shape = (height,array.shape[1],3),dtype = array.dtype)# a band of width 50 of the same width as the Y axis of the array
    title_band = annotate_image(title_band * bg_color, title , **kwargs)#fontsize=55, font = "DejaVuSans-Bold.ttf",shadow_size = 1)
    return title_band



# possibnle optimisations : order of indexes in memaps , the select index (time) should be first maybe for faster access : answer, yes it does
# pillow SIMD resize ? https://github.com/uploadcare/pillow-simd
# and possibly, instead of resizing each image then writing in inside the background, maybe write each inside a full size background 
# and resize once to the desires background final shape... 
        
#%% Main test
if __name__ == "__main__" :
    
    #assign_job(create_job())
    #limit_memory(10000 * 1024 * 1024) #10GB memory Max
    
    import os , sys
    import matplotlib.pyplot as plt
    sys.path.append(r"D:\Tim\Documents\Scripts\__packages__")
    import pGenUtils as guti
    import pLabAna as lana
    import pImage
    
    path = r"\\Xps139370-1-ds\data_tim_2\Timothe\DATA\BehavioralVideos\Whisker_Video\Whisker_Topview\Expect_3_mush\Mouse60\210428_1"
    #videos = guti.re_folder_search(path,r".*.avi")
    videos = range(30)
    vb = VignetteBuilder()
    
    for video in videos :
        vid = np.random.rand(1000,1000)*255
        vb.add_video( vid, max_time = 500, array_type = "2D_bw")
        #vb.add_video( pImage.VideoReader(video).frames() )
    vb.set_layout("grid")
