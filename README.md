# pImage
 
Usage : 

**For video compression / modification :**

``` python
import pImage

input_video_path = r"foo/myfoovideo.seq"
output_video_path = r"bar/myconvertedfoovideo.avi"

converter = pImage.converters.Standard_Converter(input_video_path, output_video_path, optionnal_key_value_arguments)
converter.start()
```

Once finished, the script will display `Done` in console

**Under the hood :**

It automatically selects a reader and a writer depending on the extension of your input and output video pathes, and performs reading and writing on different processes to maximize speed in case reading and writing are done on different hard drives. (for multi-core processors)
With optionnal arguments, one can also make third process intervene inbetween the reader and writer to modify the image as wished. (rotation, scale, image enhancement, custom operations supported by providing your own transform function as argument)
