import numpy as np 
from skimage.transform import resize
from featureSign import featureSign
from scipy.signal import convolve2d
from tqdm import tqdm

#Compute Features using First and Second Order Gradient Filters
def F(img_lr: np.ndarray):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat

def lin_scale(xh, us_norm):
    hr_norm = np.linalg.norm(xh)

    if hr_norm > 0:
        s = us_norm * 1.2 / hr_norm
        xh *= s
    return xh

def SR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap):
    patch_size = 5

    #Upsample the low resolution image
    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    
    #Initialize the high resolution image
    img_hr = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    #Extract first order and second order gradients from the upscaled, low resolution image
    img_lr_y_feat = F(img_us)

    #Create a grid over which we can obtain all the patches
    gridx = np.append(np.arange(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(np.arange(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0

    #Iterate over each point in the grid
    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            #Get Upsampled patch from the Low Res Image
            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(us_patch)
            us_patch = us_patch.flatten(order='F') - us_mean
            us_norm = np.linalg.norm(us_patch)

            #Get Feature Patch from Gradient Patch
            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = feat_patch.flatten(order='F')
            feat_norm = np.linalg.norm(feat_patch)
            
            #Normalize Feature Patch if needed
            if feat_norm > 1:
                y = feat_patch / feat_norm
            else:
                y = feat_patch
                
            w = featureSign(y, Dl, lmbd)

            hr_patch = Dh @ w
            hr_patch = lin_scale(hr_patch, us_norm)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] += 1

    index = np.where(cnt_matrix < 1)[0]
    img_hr[index] = img_us[index]
    cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)
    return img_hr