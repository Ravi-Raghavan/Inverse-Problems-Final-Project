#Test Class to downsample images
import numpy as np
from scipy.sparse import csr_matrix

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


image = np.random.randn(512, 512)

downsample_matrix = downsample(image)

downsampled_image = downsample_matrix @ image.reshape((-1, 1))
downsampled_image = downsampled_image.reshape(image.shape)