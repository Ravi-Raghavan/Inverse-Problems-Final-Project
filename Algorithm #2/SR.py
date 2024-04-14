### Experimenting Around with Algorithm 2
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn import linear_model
from tqdm import tqdm

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
        idx = np.unique(np.concatenate(( Kr * np.arange(Kc), np.arange(Kr))))
    elif (hStrip > 0 and vStrip == 0):
        idx = Kr * np.arange(Kc)
    elif (hStrip == 0 and vStrip > 0):
        idx = np.arange(Kr)
    
    idx = np.sort(idx)
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

#Super Resolution via Sparse Representation
#Dh: Dictionary for High Resolution Patches
#Dl: Dictionary with Feature Vectors for each Vectorized Upsampled Low Resolution Patch
#Y: Low Resolution Image
def SR(Dh: np.ndarray, Dl: np.ndarray, Y: np.ndarray, blur_kernel: np.ndarray, upscale):
    #This parameter was set to 1 in all the authors' experiments
    beta = 1
    
    #Patch size that will be used to extract patches from low resolution image
    patch_shape = (5, 5)
    stride = patch_shape[0] - 1
    
    #Set up the Approximation of the High Resolution Image
    X0 = np.zeros(shape = Y.shape) #approximation of high resolution image
    total_patches = (1 + int((Y.shape[0] - patch_shape[0]) / stride)) ** 2 #number of total patches in the low resolution image
    
    for patch_num in tqdm(range(total_patches)):
        #Extract patch from low resolution image
        y = extract_patch(Y, patch_shape, stride, patch_num)
        
        #Calculate Mean
        m = np.mean(y)
        
        #Solve Optimization Problem Outlined in Equation (8)
        D_tilde = Dl
        y_tilde = F(y)
        
        if patch_num > 0:
            P = generate_P(X0, patch_shape, stride, patch_num)
            w = generate_w(X0, patch_shape, stride, patch_num).reshape((-1, 1)) #make w a column vector
            
            D_tilde = np.concatenate((D_tilde, beta * (P @ Dh)), axis = 0)
            y_tilde = np.concatenate((y_tilde, beta * w), axis = 0)
        
        a = lasso_optimization(D_tilde, y_tilde, 0.1)
        x = Dh @ a + m
        x = x.reshape(patch_shape, order = 'F')
        insert_patch(X0, patch_shape, stride, patch_num, x)
        
    X = GD(Y, X0, 0.001, 0.00001, 2000, blur_kernel, upscale)
    return X  

#Lasso Optimization
#Solves ||D_tilde a - y_tilde ||^2_2 + lambda ||a||_1
def lasso_optimization(D_tilde, y_tilde, lamb):
    clf = linear_model.Lasso(alpha = lamb, max_iter = 100000, fit_intercept = True)
    clf.fit(D_tilde, y_tilde)
    # print(f"R^2 Score: {clf.score(D_tilde, y_tilde)}")
    return clf.coef_.reshape((-1, 1))
        
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
def downsample(M, N, upscale):    
    data = np.ones(M * N)
    rows = np.arange(M * N)
    
    #compute cols for the first row
    f_cols = np.repeat(np.array([0, upscale]), upscale // 2)[np.newaxis, :] + (upscale * (np.arange(start = 0, stop = (N // upscale) - 1, step = 1)[:, np.newaxis]))
    f_cols = np.concatenate((f_cols.flatten(order = 'C'), np.repeat(np.array([N - upscale]), upscale)))
    f_cols = f_cols[np.newaxis, :]
    
    #block set of cols
    b_cols = np.repeat(f_cols, upscale // 2, axis = 0)
    b_cols = b_cols.flatten(order = 'C')[np.newaxis, :]
    print(b_cols.shape)
    
    #create a temp array
    b_rows = np.array([0, upscale])[np.newaxis, :] + (upscale * (np.arange(start = 0, stop = (M // upscale) - 1, step = 1)[:, np.newaxis]))
    b_rows = b_rows.flatten(order = 'C')
    b_rows = np.concatenate((b_rows, np.array([b_rows[-1], b_rows[-1]])))
    b_rows = b_rows[:, np.newaxis]
    print(b_rows.shape)
    
    cols = b_cols + (N * b_rows)
    cols = cols.flatten(order = 'C')
    
    print(len(data), len(rows), len(cols))
    downsample_matrix = csr_matrix((data, (rows, cols)), shape = (M * N, M * N))
    return downsample_matrix

#I: Image is assumed to be of size (Mi, Ni)
#K: Kernel is assumed to be of size (Mk, Nk)
def pad_image(image: np.ndarray, kernel: np.ndarray):
    Mk, Nk = kernel.shape
    
    #Add Mk - 1 Padding on each side row-wise. Add Nk - 1 padding on each side column wise
    image = np.hstack([np.zeros((image.shape[0], (Nk - 1) // 2)), image, np.zeros((image.shape[0], (Nk - 1) // 2))])
    image = np.vstack([np.zeros(((Mk - 1) // 2, image.shape[1])), image, np.zeros(((Mk - 1) // 2, image.shape[1]))])    
    
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
        
def GD(Y: np.ndarray, X0: np.ndarray, step_size, c, iterations, blur_kernel, upscale):
    Mi, Ni = X0.shape
    S = downsample(Mi, Ni, upscale)
    
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
    
    loss = np.linalg.norm(Y_reshaped - (S @ H @ X_padded)) ** 2
    print(f"Loss at Iteration 0: {loss}")
    
    for iter in range(iterations):
        B = c * (X_padded - X0_padded)
        Q = (H.T @ S.T @ (Y_reshaped - (S @ H @ X_padded))) + (c * (X_padded - X0_padded))
        X_padded += step_size * Q
        
        loss = np.linalg.norm(Y_reshaped - (S @ H @ X_padded)) ** 2
        print(f"Loss at Iteration {iter + 1}: {loss}")
    
    X_padded = X_padded.reshape(padded_matrix_shape)
    
    start = (Mk - 1) // 2
    end = (Nk - 1) // 2
    X = X_padded[start: Mi + start, end: Ni + end]
    return X

#Run SR Algorithm on a Test Low Resolution Image
#Let's say we were working with 3 x 3 patches from the High Resolution Image and the Upsampled, Low Resolution Image
Dh = np.load("../Dictionaries/Dh_512_0.15_5.npy")
Dl = np.load("../Dictionaries/Dl_512_0.15_5.npy")
print(Dh.shape, Dl.shape)

# Load an image using OpenCV
image = cv2.imread("../Data/Testing/Child.png")
lIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

blur_kernel = np.ones(shape = (3, 3)) / 9
upscale = 2.0
X = SR(Dh, Dl, lIm, blur_kernel, upscale)
print("Finished Super Resolution")

#Save X, lIm, and hIm to npy files
np.save('X.npy', X)
np.save('lIm.npy', lIm)

# Plot the matrix as an image
# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

axes[0].imshow(X, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[0].set_title('SuperResolution Image')  # Set title
axes[0].set_xlabel('Columns')  # Set label for x-axis
axes[0].set_ylabel('Rows')  # Set label for y-axis

axes[1].imshow(lIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[1].set_title('Original Low Resolution Image')  # Set title
axes[1].set_xlabel('Columns')  # Set label for x-axis
axes[1].set_ylabel('Rows')  # Set label for y-axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the figure with subplots