### Experimenting Around with Algorithm 2
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

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

#I: Image/Array
#patch_shape: shape of patch
#stride: stride
#patch_num: number of patch
def insert_patch(I: np.ndarray, patch_shape: tuple, stride, patch_num, patch: np.ndarray):
    R, C = I.shape #image shape
    Kr, Kc = patch_shape #patch shape
    
    pH = 1 + int((C - Kc) / stride) #number of patches per horizontal strip
    pC = 1 + int((R - Kr) / stride) #number of patches per vertical strip
    
    #insert patch
    hStrip = patch_num // pH 
    vStrip = patch_num % pH
    I[stride * hStrip: stride * hStrip + Kr, stride * vStrip: stride * vStrip + Kc] = patch
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
        idx = [0, 1, 2, 3, 6]
    elif (hStrip > 0 and vStrip == 0):
        idx = [0, 3, 6]
    elif (hStrip == 0 and vStrip > 0):
        idx = [0, 1, 2]
    
    P[idx, idx] = 1
    return P

def generate_w(X0: np.ndarray, patch_shape: tuple, stride, patch_num):
    R, C = X0.shape #High Resolution Image shape
    Kr, Kc = patch_shape #patch shape
    
    pH = 1 + int((C - Kc) / stride) #number of patches per horizontal strip
    pC = 1 + int((R - Kr) / stride) #number of patches per vertical strip
    
    hStrip = patch_num // pH 
    vStrip = patch_num % pH
    
    patch = extract_patch(X0, patch_shape, stride, patch_num)
    
    if (hStrip > 0 and vStrip > 0):
        patch[1: , 1: ] = 0
    elif (hStrip > 0 and vStrip == 0):
        patch[1:, :] = 0
    elif (hStrip == 0 and vStrip > 0):
        patch[:, 1:] = 0
    
    return patch.flatten(order = 'F')

#Super Resolution via Sparse Representation
#Dh: Dictionary for High Resolution Patches
#Dl: Dictionary with Feature Vectors for each Vectorized Upsampled Low Resolution Patch
#Y: Low Resolution Image
def SR(Dh: np.ndarray, Dl: np.ndarray, Y: np.ndarray, blur_kernel: np.ndarray):
    #Dh is a matrix of size N x Kh where N is the size of each vectorized high resolution patch
    #Dl is a matrix of size M x Kl where M is the size of the corresponding feature vector for each vectorized, upsampled low resolution patch
    N, _ = Dh.shape
    M, _ = Dl.shape
    
    #This parameter was set to 1 in all the authors' experiments
    beta = 1
    
    #Patch size that will be used to extract patches from low resolution image
    patch_shape = (3, 3)
    patch_size = patch_shape[0] * patch_shape[1]
    stride = patch_shape[0] - 1
    
    #We will be multiplying this with the approximate patches of the low resolution image to extract features
    #This will be a row-wise gradient extractor
    F = -1 * np.eye(patch_size)
    indices = np.arange(patch_size)[:: patch_shape[0]]
    insert_indices = indices + 2
    F[indices, insert_indices] = 1
    
    X0 = np.zeros(shape = Y.shape) #approximation of high resolution image
    total_patches = (1 + int((Y.shape[0] - patch_shape[0]) / stride)) ** 2 #number of total patches in the low resolution image
    
    print(f"Total Patches: {total_patches}")
    for patch_num in range(total_patches):
        y = extract_patch(Y, (3, 3), 2, patch_num).flatten(order = 'F').reshape((-1, 1))
        #Normalize to have 0 mean
        m = np.mean(y)
        y -= m
        
        #Solve Optimization Problem Outlined in Equation (8)
        D_tilde = Dl
        y_tilde = F @ y
        
        if patch_num > 0:
            P = generate_P(X0, (3, 3), 2, patch_num)
            w = generate_w(X0, (3, 3), 2, patch_num).reshape((-1, 1)) #make w a column vector
            
            D_tilde = np.concatenate((D_tilde, beta * (P @ Dh)), axis = 0)
            y_tilde = np.concatenate((y_tilde, beta * w), axis = 0)
        
        a = proximal_GD(D_tilde, y_tilde, 0.001, 0.1, 10) 
        x = Dh @ a + m
        x = x.reshape(patch_shape, order = 'F')
        insert_patch(X0, patch_shape, stride, patch_num, x)
        
        print(f"Finished Processing Patch # {patch_num + 1}")
    
    X = GD(Y, X0, 0.001, 0.01, 100, blur_kernel)
    return X   
        
        
#prox operator
def prox(x, alpha):
    return np.piecewise(x, [x < -alpha, (x >= -alpha) & (x <= alpha), x >= alpha], [lambda x: x + alpha, 0, lambda x: x - alpha])
    

#Use Proximal GD to Solve Optimization Problem Outlined in Equation (8)
def proximal_GD(D_tilde: np.ndarray, y_tilde: np.ndarray, step_size, lamb, iterations):
    a = np.random.normal(size = (D_tilde.shape[1], 1))
    loss = (np.linalg.norm(D_tilde @ a - y_tilde) ** 2) + (lamb * (np.sum(np.abs(a))))
    # print(f"Loss: {loss}, Iteration: 0")
    
    for i in range(iterations):
        grad = 2 * (D_tilde.T @ D_tilde @ a) - 2 * (D_tilde.T @ y_tilde)
        a = a + step_size * (-1 * grad)
        a = prox(a, step_size * lamb)
        
        loss = (np.linalg.norm(D_tilde @ a - y_tilde) ** 2) + (lamb * (np.sum(np.abs(a))))
        # print(f"Loss: {loss}, Iteration: {i + 1}")
    
    return a

#Downsample an Image of size M X N
def downsample(M, N):
    downsample_matrix = np.eye(M * N)
    
    data = np.ones(M * N)
    rows = np.arange(M * N)
    
    cols = np.repeat(np.arange(start = 0, stop = N, step = 2), 2)[np.newaxis, :] + (N * (np.arange(start = 0, stop = M, step = 2)[:, np.newaxis]))
    cols = np.repeat(cols, 2, axis = 0) 
    cols = cols.flatten(order = 'C')
    
    print(len(data), len(rows), len(cols))
    downsample_matrix = csr_matrix((data, (rows, cols)), shape = (M * N, M * N))
    return downsample_matrix

#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
def pad_image(image: np.ndarray, kernel: np.ndarray):
    Mk, Nk = kernel.shape
    
    #Add Mk - 1 Padding row-wise. Add Nk - 1 padding column wise
    image = np.vstack([image, np.zeros((Mk - 1, image.shape[1]))])    
    image = np.hstack([image, np.zeros((image.shape[0], Nk - 1))])
    
    return image

#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
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
        
def GD(Y: np.ndarray, X0: np.ndarray, step_size, c, iterations, blur_kernel):
    Mi, Ni = X0.shape
    S = downsample(Mi, Ni)
    
    Mk, Nk = blur_kernel.shape
    H = convolution_matrix_fast(Mi + Mk - 1, Ni + Nk - 1, blur_kernel)
    
    print(S.shape, H.shape, X0.shape, Y.shape)
    
    X = X0
    
    X0_padded = pad_image(X0, blur_kernel)
    X_padded = pad_image(X, blur_kernel)
    
    padded_matrix_shape = X0_padded.shape
    
    Y_reshaped = Y.reshape((-1, 1))
    X0_padded = X0_padded.reshape((-1, 1))
    X_padded = X_padded.reshape((-1, 1))
    print(X_padded.shape)
    
    for iter in range(iterations):
        B = c * (X_padded - X0_padded)
        Q = (H.T @ S.T @ (Y_reshaped - (S @ H @ X_padded))) + (c * (X_padded - X0_padded))
        X_padded += step_size * Q
        
        print(f"Iteration {iter + 1} of GD Finished")
    
    X_padded = X_padded.reshape(padded_matrix_shape)
    X = X_padded[0: Mi, 0: Ni]
    return X

#Test Patch Extraction
# Define the values for the array
values = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

# Create a NumPy array from the values
A = np.array(values)
for patch_num in range(4):
    print(extract_patch(A, (3, 3), 2, patch_num))
    

#Run a brief test of SR Algorithm with Dummy Matrices
#Let's say we were working with 3 x 3 patches from the High Resolution Image and the Upsampled, Low Resolution Image
Dh = np.load("../Dictionaries/Dh.npy")
Dl = np.load("../Dictionaries/Dl.npy")\

# Load an image using OpenCV
image = cv2.imread('../Data/Testing/Child.png')
lIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

image = cv2.imread("../Data/Testing/Child_gnd.bmp")
hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

lIm = cv2.resize(lIm, hIm.shape[::-1], interpolation = cv2.INTER_NEAREST)

print(lIm.shape)

X = SR(Dh, Dl, lIm, np.ones(shape = (3, 3)) / 9)

print("Finished Super Resolution")

# Plot the matrix as an image
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

axes[0].imshow(X, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[0].set_title('SuperResolution Image')  # Set title
axes[0].set_xlabel('Columns')  # Set label for x-axis
axes[0].set_ylabel('Rows')  # Set label for y-axis

axes[1].imshow(lIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[1].set_title('Original Low Resolution Image')  # Set title
axes[1].set_xlabel('Columns')  # Set label for x-axis
axes[1].set_ylabel('Rows')  # Set label for y-axis

axes[2].imshow(hIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[2].set_title('Correct High Resolution Image')  # Set title
axes[2].set_xlabel('Columns')  # Set label for x-axis
axes[2].set_ylabel('Rows')  # Set label for y-axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the figure with subplots