import cv2
import numpy as np

# Load an image using OpenCV
image = cv2.imread('test.jpeg')
hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

print(hIm.shape)

# Blur Image
blurred_image = cv2.GaussianBlur(hIm, (15, 15), 0)
print(blurred_image.shape)

upscale = 2

lIm = cv2.resize(blurred_image, tuple(int(x * (1/upscale)) for x in blurred_image.shape)[::-1], interpolation = cv2.INTER_CUBIC)
print(lIm.shape)

lIm = cv2.resize(lIm, blurred_image.shape[::-1], interpolation = cv2.INTER_CUBIC)
print(lIm.shape)

# Display the downsampled image
cv2.imshow('High Resolution Image', hIm)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Low Resolution Image', lIm)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Y = SH X
#Y X^{-1} = SH

SH = lIm @ np.linalg.inv(hIm)

print(SH)