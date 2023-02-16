# pImage

The goal of this package is to provide a common and unified API for reading and writing sequences of 2D arrays from farious formats to python numpy arrays, and the other way around. It uses openCV lib for most of the job, and Pillow for some cases (as gif for example)
As a supplementary feature, this package also allows for manipulating numpy arrays for esthetic purposes :
- providing classes to snap together multiple arrays synchronized in a single output (mutiple videos layed out next to each other in a singe avi or gif for example), 
- helper functions for making colored arrays (3D RGB arrays) from arrays of 1D values, using colormaps
- save matplotlib plots outputs to rasterized RGB arrays, in order to be used in the API like any other array and be saveable in videos.
- provide image transformations functions (contrast, brightness, clahe, padding, cropping, gaussian filtering, sharp filters, text annotation) to 2D or 3D arrays, and be able to perform these at read time from a single frame of a reader function (in order to minimize RAM load when working with very large videos).
- For the same purpose of minimizing RAM load, a reader and writer can be piped together so that the later writes frames to disk as soon as the reader provides a frame, keeping RAM usage low during vide conversion (as well as video tratment, as a reader can be transformed into a "transforming_reader" using any transformation function cited above.)

 
Usage : 

**Simple guide for video compression :**

``` python
import pImage

input_video_path = r"foo/myfoovideo.seq"
output_video_path = r"bar/myconvertedfoovideo.avi"

converter = pImage.Standard_Converter(input_video_path, output_video_path)
converter.start()
```
Implement progressbar for files of over 100 frames.
Once finished, the script will display `Done` in console

**Under the hood :**

It automatically selects a reader and a writer depending on the extension of your input and output video pathes, and performs reading and writing on different processes to maximize speed in case reading and writing are done on different hard drives. (for multi-core processors)
With optionnal arguments, one can also make the converter execute any function provided to it inbetween the reader and writer to modify the images as wished. (rotation, scale, LUT ,image enhancement, CLAHE,  etc..)

The writer is a class that will generate the file and work variables only at first writer.write(frame) call. That way, it avoids anticipatory declarations of width, height, data type etc at instanciation, an will still work as long as you keep feeding consistant data to the object.

## Notes for dev for me later :
Reduce RAM usage intensity for mosaics of snapped and resized videos, by using https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/ ZARR instead of numpy memmaps (numpy memmaps rely on OS APIs and are not well tuneable (at least on windows) + not efficient timewise + proved to not even be usefull regarding ram as memaps starts to discharge from ram only when ram is completely saturated, making the computer practically unuseable in the background, a shame for a main station running analysis while working on something else....)
