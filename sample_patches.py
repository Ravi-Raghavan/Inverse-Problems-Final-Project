import cv2
import numpy as np

## File to Sample Patches from Image
def sample_patches(im, patch_size, patch_num, upscale):
    H = None
    L = None
    
    #Initialize the High Resolution Image
    hIm = im
    if im.shape[2] == 3:
        hIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
    #Generate Low Resolution Image
    
    lIm = cv2.resize(hIm, tuple(int(x * (1/upscale)) for x in hIm.shape), interpolation = cv2.INTER_CUBIC)
    lIm = cv2.resize(lIm, hIm.shape, interpolation = cv2.INTER_CUBIC)
    
    #Get dimensions of hIm
    nrow, ncol = hIm.shape
    
    #Get posible values of (x, y) that is bottom right corner of patch
    x = np.random.permutation(np.arange(1, nrow-2*patch_size)) + patch_size
    y = np.random.permutation(np.arange(1, ncol-2*patch_size)) + patch_size
    
    #Generated Meshgrid
    X, Y = np.meshgrid(x, y)
    
    #Flatten X and Y column wise
    xrow = X.flatten(order = 'F')
    ycol = Y.flatten(order = 'F')
    
    if patch_num < len(xrow):
        xrow = xrow[: patch_num]
        ycol = ycol[: patch_num]
    
    patch_num = len(xrow)
    
    H = np.zeros(shape = (patch_size ** 2, patch_num))
    L = np.zeros(shape = (4 * (patch_size ** 2), patch_num))
    
    print(im.shape, "<->", hIm.shape)
    return H, L