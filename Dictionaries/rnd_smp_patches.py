import os
import numpy as np
import cv2
from sample_patches import sample_patches
import random

#Randomly Sample Patches
#img_path: path of image
#img_type: type of image
#patch_size: size of patch
#num_patch: number of patches we want to sample
#upscale: upscale factor
def rnd_smp_patch(img_path, img_type, patch_size, num_patch, upscale):
    img_dir = [file for file in os.listdir(img_path) if file.endswith(img_type)]
        
    Xh = [] #Store High Resolution Patches
    Xl = [] #Store Low Resolution Patches
    
    img_num = len(img_dir) #Total number of images
    nper_img = np.zeros(shape = (img_num, ))
    print("Number of Images From Training Dataset we have Sampled: ", img_num)
    
    for idx in range(img_num):
        im = cv2.imread(os.path.join(img_path, img_dir[idx]))
        nper_img[idx] = im.size
    
    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img)).astype(int) #number of patches per image
    
    for idx in range(img_num):
        patch_num = nper_img[idx] #number of patches from this image to select
        im = cv2.imread(os.path.join(img_path, img_dir[idx]))
        
        #Sample the Patches
        H, L = sample_patches(im, patch_size, patch_num, upscale)
                
        #Append to Xh and Xl
        Xh.append(H)
        Xl.append(L)
    
    Xh, Xl = np.concatenate(Xh, axis = 1), np.concatenate(Xl, axis = 1)
    return Xh, Xl