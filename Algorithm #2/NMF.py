import numpy as np
import cv2
from skimage.color import rgb2ycbcr

#Non-Negative Matrix Factorization
def NMF(X: np.ndarray, iterations = 10):
    n, m = X.shape
    
    r = int((n * m) / (n + m))
    
    U = np.abs(np.random.normal(size = (n, r)))
    V = np.abs(np.random.normal(size = (r, m)))
    
    for iter in range(iterations):
        DeltaV = ((U.T @ X) / (U.T @ U @ V))
        DeltaU = ((X @ V.T) / (U @ V @ V.T))
        
        V *= DeltaV
        U *= DeltaU
        
        print(f"Finished Iteration {iter + 1}")
    
    return U

#Conduct Non-Negative Matrix Factorization
img_path = '../Data/Training/face.jpg'
X = cv2.imread(img_path)
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

X = rgb2ycbcr(X) #Get Convert RGB to Y, CB, Cr for Low Resolution Image
X = X[:, :, 0] #Y Component

A = NMF(X)
U, R = np.linalg.qr(A)
np.save("U.npy", U)