import numpy as np 
from skimage.transform import resize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def gauss2D(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def backprojection(U, img_lr, maxIter):
    p = gauss2D((5, 5), 1)
    p = np.multiply(p, p)
    p = np.divide(p, np.sum(p))
    
    _, r = U.shape
    c = np.random.normal(size = (r, 1))

    for i in range(maxIter):
        img_sr = U @ c
        img_lr_ds = resize(img_sr, img_lr.shape, anti_aliasing=1)
        img_diff = img_lr - img_lr_ds

        img_diff = resize(img_diff, img_sr.shape)
        c += U.T @ convolve2d(img_diff, p, 'same')
    
    img_sr = U @ c
    return img_sr