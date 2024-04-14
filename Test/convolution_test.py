import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix

#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
def pad_image(image: np.ndarray, kernel: np.ndarray):
    Mk, Nk = kernel.shape
    
    #Add Mk - 1 Padding on each side row-wise. Add Nk - 1 padding on each side column wise
    image = np.hstack([np.zeros((image.shape[0], (Nk - 1) // 2)), image, np.zeros((image.shape[0], (Nk - 1) // 2))])
    image = np.vstack([np.zeros(((Mk - 1) // 2, image.shape[1])), image, np.zeros(((Mk - 1) // 2, image.shape[1]))])    
    
    return image

def convolution_matrix_fast(Mi, Ni, kernel: np.ndarray):
    Mk, Nk = kernel.shape
    
    kernel_flat = kernel.flatten(order = 'C')
    data = np.tile(kernel_flat, (Mi - Mk + 1) * (Ni - Nk + 1))
    rows = np.repeat(np.arange((Mi - Mk + 1) * (Ni - Nk + 1)), Mk * Nk)
    
    f = np.arange(Nk)[np.newaxis, :] + Ni * (np.arange(Mk)[:, np.newaxis])
    f = f.flatten(order = 'C')
    
    f = f[np.newaxis, :] + np.arange(Ni - Nk + 1)[:, np.newaxis]
    f = f.flatten(order = 'C')
    
    f = f[np.newaxis, :] + Ni * (np.arange(Mi - Mk + 1)[:, np.newaxis])
    f = f.flatten(order = 'C')
    cols = f
    
    F = csr_matrix((data, (rows, cols)), shape = ((Mi - Mk + 1) * (Ni - Nk + 1), Ni * Mi)) #generate the convolution matrix via sparse matrix representation
    return F

y = np.arange(25).reshape(5, 5)
blur_kernel = np.ones(shape = (3, 3)) / 9

y_padded = pad_image(y, blur_kernel)
F = convolution_matrix_fast(y_padded.shape[0], y_padded.shape[1], blur_kernel)

A = convolve2d(y, blur_kernel, 'same')

B = F @ y_padded.reshape((-1, 1))

print(A)

print(B.reshape(y.shape))

# hf1 = np.array([-1,0,1]).reshape((1, -1))
# vf1 = hf1.T
    
# yG11 = convolve2d(y, hf1[::-1, ::-1],'same') #row wise 1st order derivative
# yG12 = convolve2d(y, vf1[::-1, ::-1],'same') #column wise 1st order derivative

# #Compute second order derivatives
# hf2 = np.array([1,0,-2,0,1]).reshape((1, -1))
# vf2 = hf2.T

# yG21 = convolve2d(y, hf2[::-1, ::-1], 'same') #row wise 2nd order derivative
# yG22 = convolve2d(y, vf2[::-1, ::-1], 'same') #column wise 2nd order derivative


# print(y)

# print("------")
# print(yG11)
