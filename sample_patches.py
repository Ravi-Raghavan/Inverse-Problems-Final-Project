import cv2
import numpy as np
from scipy import signal

## File to Sample Patches from Image
def sample_patches(im, patch_size, patch_num, upscale):
    #Initialize the High Resolution Image
    hIm = im
    if im.shape[2] == 3:
        hIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
    #Generate Low Resolution Image
    lIm = cv2.resize(hIm, tuple(int(x * (1/upscale)) for x in hIm.shape)[::-1], interpolation = cv2.INTER_CUBIC)
    lIm = cv2.resize(lIm, hIm.shape[::-1], interpolation = cv2.INTER_CUBIC)
    nrow, ncol = hIm.shape #Get dimensions of hIm
    
    #Get posible values of (x, y) that is top left corner of patch
    x = np.random.permutation(np.arange(0, nrow - 2 * patch_size - 1)) + patch_size
    y = np.random.permutation(np.arange(0, ncol - 2 * patch_size - 1)) + patch_size
    
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
    
    #Compute first order derivatives
    hf1 = np.array([-1,0,1]).reshape((1, -1))
    vf1 = hf1.T
    
    lImG11 = signal.convolve2d(lIm, hf1,'same') #row wise 1st order derivative
    lImG12 = signal.convolve2d(lIm, vf1,'same') #column wise 1st order derivative
    
    #Compute second order derivatives
    hf2 = np.array([1,0,-2,0,1]).reshape((1, -1))
    vf2 = hf2.T
    
    lImG21 = signal.convolve2d(lIm, hf2, 'same') #row wise 2nd order derivative
    lImG22 = signal.convolve2d(lIm, vf2, 'same') #column wise 2nd order derivative
    
    for idx in range(patch_num):
        row, col = xrow[idx], ycol[idx]
        
        #Get the patch from High Resolution Image
        Hpatch = hIm[row: row + patch_size, col: col + patch_size].flatten(order = 'F')
        H[:, idx] = Hpatch - np.mean(Hpatch)
        
        #Get the patch from Low Resolution Image
        Lpatch1 = lImG11[row:row+patch_size,col:col+patch_size].flatten(order = 'F')
        Lpatch2 = lImG12[row:row+patch_size,col:col+patch_size].flatten(order = 'F')
        Lpatch3 = lImG21[row:row+patch_size,col:col+patch_size].flatten(order = 'F')
        Lpatch4 = lImG22[row:row+patch_size,col:col+patch_size].flatten(order = 'F')

        Lpatch = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4))
        L[:, idx] = Lpatch
    
    return H, L