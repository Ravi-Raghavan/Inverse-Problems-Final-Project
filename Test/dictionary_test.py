import numpy as np
from scipy.signal import convolve2d
import cv2

def F(y: np.ndarray):
    #Compute first order derivatives
    hf1 = np.array([-1,0,1]).reshape((1, -1))
    vf1 = hf1.T
    
    yG11 = convolve2d(y, hf1[::-1, ::-1],'same').flatten(order = 'F') #row wise 1st order derivative
    yG12 = convolve2d(y, vf1[::-1, ::-1],'same').flatten(order = 'F') #column wise 1st order derivative
    
    #Compute second order derivatives
    hf2 = np.array([1,0,-2,0,1]).reshape((1, -1))
    vf2 = hf2.T
    
    yG21 = convolve2d(y, hf2[::-1, ::-1], 'same').flatten(order = 'F') #row wise 2nd order derivative
    yG22 = convolve2d(y, vf2[::-1, ::-1], 'same').flatten(order = 'F') #column wise 2nd order derivative
    
    y_features = np.concatenate((yG11, yG12, yG21, yG22)).reshape((-1, 1))
    return y_features


Dl = np.load("../Dictionaries/Dl_512_0.15_5.npy")
print(np.min(Dl), np.max(Dl))

Dh = np.load("../Dictionaries/Dh_512_0.15_5.npy")
print(np.min(Dh), np.max(Dh))

#Norm Checks
Dl_norms = np.linalg.norm(Dl, axis = 0)
print(np.min(Dl_norms), np.max(Dl_norms))

# patch = np.array([[253, 254, 253, 253, 254],
#     [253, 254, 252, 252, 254.],
#     [252, 252, 252, 252, 252.],
#     [252, 252, 252, 252, 252.],
#     [252, 252, 252, 252, 251.]])


# print(Dl[:, 0])
# print(F(patch))

image_path = "../Data/Testing/Child.png"
im = cv2.imread(image_path)
lIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

#Compute first order derivatives
hf1 = np.array([-1,0,1]).reshape((1, -1))
vf1 = hf1.T

lImG11 = convolve2d(lIm, hf1[::-1, ::-1],'same') #row wise 1st order derivative
lImG12 = convolve2d(lIm, vf1[::-1, ::-1],'same') #column wise 1st order derivative

#Compute second order derivatives
hf2 = np.array([1,0,-2,0,1]).reshape((1, -1))
vf2 = hf2.T

lImG21 = convolve2d(lIm, hf2[::-1, ::-1], 'same') #row wise 2nd order derivative
lImG22 = convolve2d(lIm, vf2[::-1, ::-1], 'same') #column wise 2nd order derivative
    
print(np.max(lImG11))

Dl = np.load("../Dictionaries/Dl.npy")
print(np.min(Dl), np.max(Dl))

Dh = np.load("../Dictionaries/Dh.npy")
print(np.min(Dh), np.max(Dh))