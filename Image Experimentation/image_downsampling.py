import cv2

# Load an image using OpenCV
image = cv2.imread('test.jpeg')

# Downsample the image using OpenCV's resize function
hIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
upscale = 2

print(hIm.shape)

lIm = cv2.resize(hIm, tuple(int(x * (1/upscale)) for x in hIm.shape)[::-1], interpolation = cv2.INTER_CUBIC)
print(lIm.shape)

lIm = cv2.resize(lIm, hIm.shape[::-1], interpolation = cv2.INTER_CUBIC)
print(lIm)

# Display the downsampled image
cv2.imshow('High Resolution Image', hIm)
cv2.imshow('Low Resolution Image', lIm)
cv2.waitKey(0)
cv2.destroyAllWindows()