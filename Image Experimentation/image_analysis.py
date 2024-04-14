### Analyze images developed by SR Algorithm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


image = cv2.imread("../Data/Testing/Child_gnd.bmp")
hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blur_kernel = np.ones(shape = (3, 3)) / 9
blurred_hIm = convolve2d(hIm, blur_kernel, mode = 'same')

#Downsample
upscale = 2.0
lIm = cv2.resize(blurred_hIm, tuple(int(x * (1/upscale)) for x in blurred_hIm.shape)[::-1], interpolation = cv2.INTER_NEAREST)
lIm = cv2.resize(lIm, blurred_hIm.shape[::-1], interpolation = cv2.INTER_NEAREST)

print(hIm.shape, blurred_hIm.shape, lIm.shape)
# Plot the matrix as an image
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

axes[0].imshow(hIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[0].set_title('High Resolution Image')  # Set title
axes[0].set_xlabel('Columns')  # Set label for x-axis
axes[0].set_ylabel('Rows')  # Set label for y-axis

axes[1].imshow(blurred_hIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[1].set_title('Blurred High Resolution Image')  # Set title
axes[1].set_xlabel('Columns')  # Set label for x-axis
axes[1].set_ylabel('Rows')  # Set label for y-axis

axes[2].imshow(lIm, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[2].set_title('Low Resolution Image')  # Set title
axes[2].set_xlabel('Columns')  # Set label for x-axis
axes[2].set_ylabel('Rows')  # Set label for y-axis


plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the figure with subplots

# lIm_npy = "../Algorithm #2/lIm.npy"
# hIm_npy = "../Algorithm #2/hIm.npy"
# X_npy = "../Algorithm #2/X.npy"

# lIm = np.load(lIm_npy)
# hIm = np.load(hIm_npy)
# X = np.load(X_npy)

# # Load an image using OpenCV
# image = cv2.imread('../Data/Testing/Child.png')
# lIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

# image = cv2.imread("../Data/Testing/Child_gnd.bmp")
# hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

# lIm = cv2.resize(lIm, hIm.shape[::-1], interpolation = cv2.INTER_NEAREST)


# lIm_subset = lIm[150: 250, 100: 200]
# hIm_subset = hIm[150: 250, 100: 200]
# X_subset = X[150: 250, 100: 200]

# # Find the minimum and maximum values in the array
# min_val = np.min(X)
# max_val = np.max(X)

# # Scale the values to be within the range of 0 to 255
# scaled_X = (X - min_val) * (255 / (max_val - min_val))
# scaled_X_subset = scaled_X[150:250, 100:200]

# print(lIm_subset)
# print("----------------")
# print(hIm_subset)
# print("----------------")
# print(scaled_X_subset)


# # Plot the matrix as an image
# # Create a figure with subplots
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

# axes[0].imshow(scaled_X_subset, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
# axes[0].set_title('SuperResolution Image')  # Set title
# axes[0].set_xlabel('Columns')  # Set label for x-axis
# axes[0].set_ylabel('Rows')  # Set label for y-axis

# axes[1].imshow(lIm_subset, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
# axes[1].set_title('Original Low Resolution Image')  # Set title
# axes[1].set_xlabel('Columns')  # Set label for x-axis
# axes[1].set_ylabel('Rows')  # Set label for y-axis

# axes[2].imshow(hIm_subset, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
# axes[2].set_title('Correct High Resolution Image')  # Set title
# axes[2].set_xlabel('Columns')  # Set label for x-axis
# axes[2].set_ylabel('Rows')  # Set label for y-axis

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()  # Display the figure with subplots