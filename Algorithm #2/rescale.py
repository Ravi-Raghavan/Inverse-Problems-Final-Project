import numpy as np 
from skimage.transform import rescale
from skimage.io import imread, imsave
from os import listdir
from tqdm import tqdm
import cv2

# Set train and val HR and LR paths
test_path = '../Data/Testing/'

numTestImages = len(listdir(test_path))
upscale = 4.0

for i in tqdm(range(numTestImages)):
    img_name = listdir(test_path)[i]
    
    if 'Fox_gnd' in img_name:
        lr_name = img_name.replace("_gnd", "")
        hIm = cv2.imread('{}{}'.format(test_path, img_name))
        new_img = cv2.resize(hIm, tuple(int(x * (1/upscale)) for x in hIm.shape)[-2::-1], interpolation = cv2.INTER_NEAREST)
        cv2.imwrite('{}{}'.format(test_path, lr_name), new_img)
        print(lr_name)