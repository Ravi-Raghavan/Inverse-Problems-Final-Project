import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


lIm_npy = "../Algorithm #2/lIm.npy"
hIm_npy = "../Algorithm #2/hIm.npy"
X_npy = "../Algorithm #2/X.npy"

lIm = np.load(lIm_npy)
hIm = np.load(hIm_npy)
X = np.load(X_npy)

X = np.clip(X, 0, 255)
print(X)  # Output: [ 5  5 10 15 15]


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