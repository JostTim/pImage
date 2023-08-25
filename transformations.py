# -*- coding: utf-8 -*-
"""

Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Thu Mar 10 21:58:07 2022
@author: Timothe
</div>
"""

try :
    from PIL import Image, ImageDraw, ImageFont
except ImportError :
    pass
import cv2
import numpy as np
import math

from .readers import _readers_factory
from .blend_modes import *

available_transforms = {"rotate","crop","annotate","resize","brightness","contrast","gamma","clahe","clipLimit","tileGridSize","sharpen"}

def TransformingReader(path,**kwargs):

    selected_reader_class = _readers_factory(path,**kwargs)

    class TransformingPolymorphicReader(selected_reader_class):
           
        callbacks = []
        
        rotation_amount = kwargs.pop("rotate",False)
        annotate_params = kwargs.pop("annotate",False)
        crop_params = kwargs.pop("crop",False)
        resize = kwargs.pop("resize",False)
        brightness = kwargs.pop("brightness",0)
        contrast = kwargs.pop("contrast",1)
        brightness_contrast = False if brightness == 0 and contrast == 1 else True
        gamma = kwargs.pop("gamma",None)
        inv_gamma = kwargs.pop("inv_gamma",True)
        clahe = kwargs.pop("clahe",False)
        sharpen_value = kwargs.pop("sharpen",None)
        #parameters for clahe auto set below if not supplied
        if (clahe and isinstance(clahe,bool)) or "clipLimit" in kwargs.keys() or "tileGridSize" in kwargs.keys():
            clahe = cv2.createCLAHE(kwargs.pop("clipLimit",8),kwargs.pop("tileGridSize",(5,5)))
        if crop_params :
            try :
                crop_params = make_crop_params(**crop_params) 
            except TypeError:
                crop_params = make_crop_params(*crop_params)
                
        def _transform_frame(self,frame):
            if self.crop_params :
                frame = crop(frame,*self.crop_params)
            if self.rotation_amount :
                frame = np.rot90(frame,self.rotation_amount,axes=(0, 1))
            if self.resize :
                frame = cv2.resize(frame, (int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)), interpolation = cv2.INTER_AREA)
            if self.clahe :
                try :
                    frame = self.clahe.apply(frame)
                except :
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:,:,0] = self.clahe.apply(lab[:,:,0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if self.brightness_contrast :
                frame = contrast_brightness(frame,self.contrast,self.brightness)
            if self.gamma is not None :
                frame = gamma(frame,self.gamma,self.inv_gamma)
            if self.annotate_params :
                frame = annotate_image(frame, self.annotate_params["text"], **self.annotate_params["params"])
            if self.sharpen_value is not None :
                frame = sharpen_img(frame, self.sharpen_value)
            
            for callback in self.callbacks :
                frame = callback(frame,self)
            
            return frame
        
        def add_callback(self,function):#you can add your own callbacks to a transformingreader 
        #they shoudl be functions that take a frame as input and give back a frame as output
        #the second argument they take is the reader itself, 
        #so that you can implement your own arguments and values withing the function (attach them to the obj before)
            self.callbacks.append(function)
            
        def frames(self):
            for item in self._get_all():
                yield self._transform_frame(item)
                
        def frame(self,frame_id):
            return self._transform_frame(super().frame(frame_id))
                    
            # @property
            # def shape(self):
            #     _shape = super().shape
            #     if self.rotation_amount % 2 :
            #         return (_shape[1], _shape[0] , _shape[2])
            #     return _shape
                
    return TransformingPolymorphicReader(path,**kwargs)

def rescale_to_8bit( input_array, vmin = None, vmax = None,fullrange = False):
    #try to find vmin vmax from input array dtype

    if vmin is None or vmax is None :
        _vmin, _vmax = get_array_minmax(input_array,fullrange)
        if vmin is None :
            vmin= _vmin
        if vmax is None:
            vmax = _vmax
        
    try :        
        return np.interp(input_array.data, (vmin, vmax), (0, 255)).astype(np.uint8)
    except AttributeError: #'memoryview' object has no attribute 'data'
        return np.interp(input_array, (vmin, vmax), (0, 255)).astype(np.uint8)
    
def get_array_minmax(input_array,fullrange = False):
    if fullrange:
        if np.issubdtype(input_array.dtype, np.integer) :
            vmin = np.iinfo( input_array.dtype ).min 
            vmax = np.iinfo( input_array.dtype ).max
        else :
            vmin = np.finfo( input_array.dtype ).min
            vmax = np.finfo( input_array.dtype ).max
    else :
        vmin = input_array.min()
        vmax = input_array.max()
    return vmin,vmax

def array_gray_to_color( input_array, vmin = None, vmax = None, fullrange = False, cmap = cv2.COLORMAP_JET , reverse = False, mask_where = None, mask_color = 0 ):
    """
    
    Args:
        input_array (TYPE): DESCRIPTION.
        **kwargs (TYPE): DESCRIPTION.

    Returns:
        TYPE: DESCRIPTION.
        
    Example :
        plt.imshow(pImage.array_gray_to_color(deltaframes[:,:,0],vmin = -0.005, vmax = 0.01,reverse = True))

    """
        
    _temp_array = rescale_to_8bit(input_array.__array__(),vmin,vmax,fullrange)
    if not reverse :
        _temp_array = np.invert(_temp_array)
    
    _temp_array = cv2.applyColorMap(_temp_array, cmap)
    
    if mask_where is not None :
        #border_mask_3D = imarrays.mask_sequence(kwargs.get("mask"),deltaframes.shape[2])
        #deltaframes[border_mask_3D[:,:,0] == 0] = kwargs.get("bg_color",0)
        _temp_array[mask_where] = mask_color
        
    return _temp_array

def sequence_gray_to_color(sequence, vmin = None, vmax = None, fullrange = False, cmap = cv2.COLORMAP_JET , reverse = False , mask_where = None, mask_color = 0, time_dimension = 2 ):
    
    def dimension_iterator(i):
        slicer = [slice(None)] * len(sequence.shape)
        slicer[time_dimension] = i
        return tuple(slicer)
    
    color_sequence = []
    
    if vmin is None or vmax is None :
        _vmin, _vmax = get_array_minmax(sequence,fullrange)
        if vmin is None :
            vmin= _vmin
        if vmax is None:
            vmax = _vmax
    
    for i in range(sequence.shape[time_dimension]):
        slicer = dimension_iterator(i)
        if mask_where is not None :
            mask = mask_where[slicer]
        else :
            mask = None
        color_sequence.append(array_gray_to_color(sequence[slicer], vmin,vmax,fullrange,cmap,reverse, mask, mask_color))
    
    return np.array(color_sequence) if time_dimension == 0 else np.moveaxis(np.array(color_sequence),0,time_dimension)

def annotate_image( input_array, text, 
                    x = 5,
                    y = 5,
                    fontsize = 100,
                    font = 'arial.ttf',
                    color = 'black',
                    shadow_color = False,
                    shadow_size = None):
    """
    Parameters
    ----------
    input_array : TYPE
        DESCRIPTION.
    text : TYPE
        DESCRIPTION.
    x : TYPE, optional
        DESCRIPTION. The default is 5.
    y : TYPE, optional
        DESCRIPTION. The default is 5.
    fontsize : TYPE, optional
        DESCRIPTION. The default is 100.
    font : TYPE, optional
        DESCRIPTION. The default is 'arial.ttf'.
        You can check the fonts avilable in your system by calling :
            import matplotlib.font_manager
            matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    color : TYPE, optional
        DESCRIPTION. The default is 'black'.
    shadow_color : TYPE, optional
        DESCRIPTION. The default is False.
    shadow_size : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    import itertools
    _temp_image = Image.fromarray(input_array)
    
    if shadow_size is not None or shadow_color :
        if shadow_size is None :
            shadow_size = 5
        shadow_font = ImageFont.truetype(font, fontsize + ( shadow_size*1))
        for i, j in itertools.product((-shadow_size, 0, shadow_size), (-shadow_size, 0, shadow_size)):
            ImageDraw.Draw(_temp_image).text( (x+i, y+j) , text , fill=shadow_color ,font = shadow_font)
        
    default_font = ImageFont.truetype(font, fontsize)
    ImageDraw.Draw(_temp_image).text( (x,y) , text , fill=color ,font = default_font)

    return np.array(_temp_image)

def annotate_sequence( sequence, text, time_dimension = 2, **kwargs ) :
    
    def dimension_iterator(i):
        slicer = [slice(None)] * len(sequence.shape)
        slicer[time_dimension] = i
        return tuple(slicer)
    
    anno_sequence = []
    for i in range(sequence.shape[time_dimension]):
        anno_sequence.append(annotate_image(sequence[dimension_iterator(i)] , text, **kwargs ))
    
    return np.array(anno_sequence) if time_dimension == 0 else np.moveaxis(np.array(anno_sequence),0,time_dimension)        

def make_crop_params(*args,**kwargs):
    if len(args) == 4:
        return args
    if len(args) == 1 and not "value" in kwargs.keys():
        kwargs.update({"value":args[0]})
    def set_value_if_not_none(sval):
        return kwargs.get(sval) if kwargs.get(sval,None) is not None else kwargs.get("value",None)
    sides = ["top","bottom","left","right"]
    values = []
    for side in sides:
        _val = set_value_if_not_none(side)
        if _val is None :
            raise ValueError(f"Must specify at least a general value if specific {side} argument is missing")
        values.append(_val)
    return values


def binarize(image,threshold,boolean = True):
    """
    Binarize an image at a given threshold.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    threshold : int
        Pixel value at which all pixels above are defined as white (255) and all pixels below are defined as black(0).
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    binimg : TYPE
        DESCRIPTION.

    """

    _, binimg = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)

    if boolean:
        return binimg.astype(np.bool)
    return binimg

def crop(array,*args,**kwargs):
    values = make_crop_params(*args,**kwargs)
    return array[values[0]:array.shape[0]-values[1],values[2]:array.shape[1]-values[3]]

def contrast_brightness(array,alpha,beta):
    #alpha = contrast, beta = brightness
    return (np.clip(( array.astype(np.int16) * alpha ) + beta, a_min = 0, a_max = 255)).astype(np.uint8)

def gamma(array,gamma,inv= True):
    return apply_lut(array, make_lut_gamma(gamma,inv))

def curve(array,slope,shift):
    return apply_lut(array, make_lut_curve(slope,shift))

def clahe(array,clahe= None,clipLimit = 8, tileGridSize = (5,5) ):
    if clahe is None :
        clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
    return clahe.apply(array)
    
def apply_lut(array, lut):
    return np.take(lut,array)
    
def make_lut_gamma(gamma,inv_gamma = True):
    if inv_gamma : 
        gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 255)]).astype("uint8")
    return table

def make_lut_curve(slope=4,inflexion_point=0.5):
    slope = constrain(slope,1,None)
    inflexion_point = constrain(inflexion_point,0,1)
    cut = slope*inflexion_point
    l_cut, r_cut = cut,  slope - cut
    return [constrain(to_uint(math.erf(i))) for i in np.linspace(-l_cut,r_cut,256)]

def sharpen_img(img,amount = 0.7):
    from scipy.ndimage.filters import median_filter
    lap = cv2.Laplacian(median_filter(img, 1),cv2.CV_64F)
    return img - amount*lap

def to_uint(value):#-1 - 1 to 0 - 255
    return int((value + 1) * (255/2))

def to_norm(value):#0-255 to -1 - 1
    return int((value - (255/2))/ (255/2))

def constrain(value, mini = 0 , maxi = 255):
    if mini is not None and value < mini :
        return mini
    if maxi is not None and value > maxi :
        return maxi
    return value
    
def grayscale(image,biases = [1/3,1/3,1/3]):
    """
    Calculate gray image value from RGB value. May include bias values to correct for luminance differences in layers.

    Parameters
    ----------
    rgb : TYPE
        DESCRIPTION.
    biases : TYPE, optional
        DESCRIPTION. The default is [1/3,1/3,1/3].

    Returns
    -------
    gray : numpy.ndarray
        Gray image (2D).

    """
    try :
        gray = np.zeros_like(image[:,:,0])
    except IndexError :
        return image
    for color_dim in range(3):
        gray = gray + (image[:,:,color_dim] * biases[color_dim])
    return gray

def pad(image,value,mode = "constant",**kwargs):
    """
    Pad an image with black (0) borders of a given width (in pixels).
    The padding is homogeneous on the 4 sides of the image.

    Parameters
    ----------
    binimg : numpy.ndarra y(2D)
        Input image.
    value : int
        Pad width (in pixels).
    **kwargs : TYPE
       - mode : "constant" default
       -constant_value : value of pixel if mode is constant

    Returns
    -------
    binimg : numpy.ndarray
        Output image.

    """
    import numpy as np
    return np.pad(image, ((value,value),(value,value)), mode, **kwargs )

def null_image(shape):
    """
    Generate a black image of a given dimension with a white cross on the middle, to use as "no loaded image" user readout.

    Parameters
    ----------
    shape : (tuple) with :
        X : (int) Shape for first dimension on the generated array (X)
        Y : (int) Shape for second dimension on the generated array (Y).

    Returns
    -------
    img : numpy.ndarray
        Blank image with white cross, of [X,Y] shape.

    """
    from skimage.draw import line_aa
    X,Y = shape
    img = np.zeros((X, Y), dtype=np.uint8)
    rr, cc, val = line_aa(0, 0, X-1, Y-1)
    img[rr, cc] = val * 255
    rr, cc, val = line_aa(0, Y-1, X-1, 0)
    img[rr, cc] = val * 255

    return img

def gaussian(frame,value):
    """
    Blur a 2D image (apply a gaussian 2D filter on it).

    Parameters
    ----------
    frame : numpy.ndarray (2D)
        Input image.
    value : int
        Width of the 2D gaussian curve that is used to look for adjacent pixels values during blurring.

    Returns
    -------
    frame : numpy.ndarray (2D)
        Output image (blurred).

    """
    from skimage import filters
    frame = filters.gaussian(frame, sigma=(value, value), truncate = 6, preserve_range = True).astype('uint8')
    return frame

def convert_scale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25, return_metrics = False):
    try :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except :
        gray = image

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convert_scale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta) if return_metrics else auto_result

def mpl_colorify_array(array, cmap):
    """Converts an image as numpy array from 2D (grayscale) to 3D with 3rd dimension having 4 values, RGB and A. 
    (red green blue and alpha respectively)
    
    Args:
        array (_type_): _description_
        cmap (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(cmap,list):
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list("blackgreen", cmap)
    import matplotlib.pyplot as plt
    norm = plt.Normalize(0,255)
    def get_value(value):
        nonlocal norm
        return cmap(norm(value))
    new_array = np.apply_along_axis(get_value, arr = array.ravel(), axis = 0) *255
    return new_array.reshape(array.shape+(4,)).astype(np.uint8)

def compute_transform_matrix(image,dx,dy,angle,tfirst = False):
    """
    This function calculates the transformation matrix for a given image.
    
    Parameters:
        image (numpy.ndarray): 2d array representing image. 
        dx (int): Horizontal translation in pixels.
        dy (int): Vertical translation in pixels.
        angle (float): Rotation angle in degrees.
        tfirst (bool, optional): If True, the translation is applied before the rotation. Default is False.

    Returns:
        numpy.ndarray: The resulting transformation matrix.
    """
    Matrix_r = np.vstack((cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2), angle, 1.0),np.array([0,0,1]))) 
    Matrix_t = np.vstack((np.array([[1,0,dx],[0,1,dy]]),np.array([0,0,1]))) 
    if tfirst :
        return np.matmul(Matrix_r,Matrix_t)[:2,:] #translation then rotation, change order to do otherwise
    return np.matmul(Matrix_t,Matrix_r)[:2,:] #rotation then translation, change order to do otherwise

def affine_transform(array,dx,dy,angle,tfirst = False,bordervalue = 0):
    """
    This function performs an affine transform on a given image.
    
    Parameters:
        array (numpy.ndarray): 2d array representing image. 
        dx (int): Horizontal translation in pixels.
        dy (int): Vertical translation in pixels.
        angle (float): Rotation angle in degrees.
        tfirst (bool, optional): If True, the translation is applied before the rotation. Default is False.
        bordervalue (int, optional): Pixel value for the borders after the affine transformation. Default is 0.

    Returns:
        numpy.ndarray: Transformed image.
    """
    id_matrix = compute_transform_matrix(array,dx,dy,angle,tfirst) #these have been tested and are the good direction 
    #(+x is pushing data toward right, +y toward bottom on a default matplotlib imshow)
    if np.isnan(bordervalue):
        array = array.astype(np.float32)
    return cv2.warpAffine(array,id_matrix,tuple(reversed(array.shape[0:2])),flags=cv2.INTER_LANCZOS4, borderMode  = cv2.BORDER_CONSTANT, borderValue = bordervalue)

def find_best_exposure(image, method = "percentile", pmin = 2, pmax = 98):
    """This function calculates the best exposure values for a given image. 

    The function utilizes OpenCV-python library to calculate the histogram of the image.
    The cumulative sum of histogram bins is calculated using numpy, and is used to identify 
    the values below and above which 1% of the pixel intensities lie. This range of intensity (vmin, vmax)

    Args:
        image (np.ndarray): An image in numpy array format for which best exposure needs to be calculated.

    Returns:
        tuple: A tuple containing two elements, vmin and vmax representing the range of best exposure values.
    """
    # Calculate the histogram of the image
    if method == "iqr" :
        #TODO : #this method doesn't work, need to fix it
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        hist = hist.ravel()/hist.sum()
        Q1, Q3 = np.percentile(hist, [25 ,75])
        iqr = Q3 - Q1

        vmin = max(0, Q1 - 1.5*iqr)
        vmax = min(255, Q3 + 1.5*iqr)

    if method == "percentile" :

        vmin = np.percentile(image, pmin)
        vmax = np.percentile(image, pmax)

    return (vmin, vmax)

if __name__ == "__main__" :
    
    test  = TransformingReader("tes.avi",rotate = 1)
    