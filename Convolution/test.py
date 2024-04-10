#Test Class to represent Convolution as a Matrix Operation
import numpy as np
from scipy.sparse import csr_matrix
import time

#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
def convolution_matrix(image: np.ndarray, kernel: np.ndarray):
    Mi, Ni = image.shape
    Mk, Nk = kernel.shape
    
    D = np.zeros(shape = (Ni - Nk + 1, Ni * Mk))
    for idx in range(Mk):
        C = np.zeros(shape = (Ni - Nk + 1, Ni))
        row = np.concatenate((kernel[idx, :], np.zeros(Ni - Nk)), axis = None)
        for shift in range(Ni - Nk + 1):
            C[shift] = np.roll(row, shift)
        
        D[:, (idx * Ni) : ((idx + 1) * Ni)] = C
    
    
    F = np.zeros(shape = ((Mi - Mk + 1) * (Ni - Nk + 1), Ni * Mi))
    for idx in range(Mi - Mk + 1):
        E = np.zeros(shape = (Ni - Nk + 1, Ni * Mi))
        E[:, (idx * Ni) : ((idx + Mk) * Ni)] = D
        F[(idx * (Ni - Nk + 1)) : ((idx + 1) * (Ni - Nk + 1)), :] = E
    
    return F
    
#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
def convolution_matrix_fast(image: np.ndarray, kernel: np.ndarray):
    original_Mi, original_Ni = image.shape
    Mi, Ni = image.shape
    Mk, Nk = kernel.shape
    
    #Add Mk - 1 Padding row-wise. Add Nk - 1 padding column wise
    # image = np.vstack([np.zeros((Mk - 1, image.shape[1])), image, np.zeros((Mk - 1, image.shape[1]))])    
    # image = np.hstack([np.zeros((image.shape[0], Nk - 1)), image, np.zeros((image.shape[0], Nk - 1))])
    
    #Recalculate Mi and Ni
    Mi, Ni = image.shape
    
    #Reshape image into column vector
    image = image.reshape((-1, 1))
    
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
    
    output = F @ image #compute output
    
    #Reshape output back into a matrix
    output = output.reshape(Mi - Mk + 1, Ni - Nk + 1)
    
    # # Calculate the starting row and column indices for the middle chunk
    # output_M, output_N = output.shape
    # start_row = (output_M - original_Mi) // 2
    # start_col = (output_N - original_Ni) // 2
    
    # # Extract the middle chunk corresponding to image size
    # output = output[start_row : start_row + original_Mi, start_col : start_col + original_Ni]
    return F, output

def downsample():
    
    pass

start = time.time()

image = np.ones(shape = (512, 512))

kernel = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

F, output = convolution_matrix_fast(image, kernel)
print(F.shape)

F = F.T
print(F.shape)

print(output.shape)

end = time.time()

# Calculate the elapsed time
elapsed_time = end - start
print("Elapsed Time:", elapsed_time, "seconds")
