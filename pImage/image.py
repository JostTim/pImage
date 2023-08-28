import os

def save_as_image(image,filepath):
    import PIL as pillow
    filepath = os.path.abspath(filepath)
    original = pillow.Image.fromarray(image)
    format = os.path.splitext(filepath)[1][1:]
    original.save(filepath, format=format)