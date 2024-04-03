import numpy as np
from scipy.signal import convolve2d
import cv2

# Load an image using OpenCV
image = cv2.imread('test.jpeg')

# Define a blur kernel
kernel_size = 80
blur_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the blur kernel using convolution
blurred_image = convolve2d(gray_image, blur_kernel, mode='same', boundary='symm')

# Convert the blurred image back to uint8 format
blurred_image = blurred_image.astype(np.uint8)

# Display the original and blurred images
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()