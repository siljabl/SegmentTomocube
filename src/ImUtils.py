import imageio
import skimage.io as io
    

def commonStackReader(filename):
    ## imageio.v3
    img = imageio.v3.imread(filename)
    if (img.ndim == 3):
        return img
    ## skimage
    del img
    img = io.imread(filename,plugin="pil")    
    if (img.ndim == 3):
        return img

    print("/!\ No valid reader found!!!")



def commonMultiChannelStackReader(filename):
    ## imageio.v3
    img = imageio.v3.imread(filename)
    if (img.ndim == 4):
        return img
    ## skimage
    del img
    img = io.imread(filename,plugin="pil")    
    if (img.ndim == 4):
        return img

    print("/!\ No valid reader found!!!")
