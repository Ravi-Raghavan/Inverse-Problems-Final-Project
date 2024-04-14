import numpy as np
from scipy.signal import convolve2d


y = np.arange(25).reshape(5, 5)

hf1 = np.array([-1,0,1]).reshape((1, -1))
vf1 = hf1.T
    
yG11 = convolve2d(y, hf1[::-1, ::-1],'same') #row wise 1st order derivative
yG12 = convolve2d(y, vf1[::-1, ::-1],'same') #column wise 1st order derivative

#Compute second order derivatives
hf2 = np.array([1,0,-2,0,1]).reshape((1, -1))
vf2 = hf2.T

yG21 = convolve2d(y, hf2[::-1, ::-1], 'same') #row wise 2nd order derivative
yG22 = convolve2d(y, vf2[::-1, ::-1], 'same') #column wise 2nd order derivative


print(y)

print("------")
print(yG11)
