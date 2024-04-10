#Test Class to downsample images
import numpy as np
from scipy.sparse import csr_matrix
import cv2
import matplotlib.pyplot as plt


#Downsample an Image
def downsample(image: np.ndarray):
    M, N = image.shape
    downsample_matrix = np.eye(M * N)
    
    data = np.ones(M * N)
    rows = np.arange(M * N)
    
    cols = np.repeat(np.arange(start = 0, stop = N, step = 2), 2)[np.newaxis, :] + (N * (np.arange(start = 0, stop = M, step = 2)[:, np.newaxis]))
    cols = np.repeat(cols, 2, axis = 0) 
    cols = cols.flatten()
    
    print(len(data), len(rows), len(cols))
    downsample_matrix = csr_matrix((data, (rows, cols)), shape = (M * N, M * N))
    return downsample_matrix


# Load an image using OpenCV
image = cv2.imread('../Image Experimentation/test.jpeg')

# Downsample the image using OpenCV's resize function
hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
upscale = 2
print(hIm)

M = downsample(hIm)

lIm = M @ hIm.reshape((-1, 1))
lIm = lIm.reshape(hIm.shape)
print(lIm)

print(np.min(lIm), np.max(lIm))
print(np.min(hIm), np.max(hIm))


# Display the downsampled image
# cv2.imshow('High Resolution Image', hIm)
# cv2.imshow('Low Resolution Image', lIm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(lIm, cmap='gray')
plt.axis('off')  # Turn off axis
plt.show()
