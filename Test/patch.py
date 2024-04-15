import numpy as np

#I: Image/Array
#patch_shape: shape of patch
#stride: stride
#patch_num: number of patch
def extract_patch(I: np.ndarray, patch_shape: tuple, stride, patch_num):
    R, C = I.shape #image shape
    Kr, Kc = patch_shape #patch shape
    
    pH = 1 + int((C - Kc) / stride) #number of patches per horizontal strip
    pC = 1 + int((R - Kr) / stride) #number of patches per vertical strip
    
    #extract patch
    hStrip = patch_num // pH 
    vStrip = patch_num % pH
    patch = I[stride * hStrip: stride * hStrip + Kr, stride * vStrip: stride * vStrip + Kc]
    return patch

def generate_P(X0: np.ndarray, patch_shape: tuple, stride, patch_num):
    R, C = X0.shape #image shape
    Kr, Kc = patch_shape #patch shape
    
    pH = 1 + int((C - Kc) / stride) #number of patches per horizontal strip
    pC = 1 + int((R - Kr) / stride) #number of patches per vertical strip
    
    hStrip = patch_num // pH 
    vStrip = patch_num % pH
        
    P = np.zeros(shape = (Kr * Kc, Kr * Kc))
    idx = []
    if (hStrip > 0 and vStrip > 0):
        idx = np.unique(np.concatenate(( Kr * np.arange(Kc), np.arange(Kr))))
    elif (hStrip > 0 and vStrip == 0):
        idx = Kr * np.arange(Kc)
    elif (hStrip == 0 and vStrip > 0):
        idx = np.arange(Kr)
    
    idx = np.sort(idx)
    P[idx, idx] = 1
    return P

A = np.arange(10 * 10).reshape(10, 10)
patch_shape = (5, 5)
stride = 4

total_patches = (1 + int((A.shape[0] - patch_shape[0]) / stride)) ** 2 #number of total patches in the low resolution image
print(f"Total Patches: {total_patches}")
for patch_num in range(total_patches):
    if patch_num > 0:
        print(generate_P(A, patch_shape, stride, patch_num))
    
