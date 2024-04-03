import cv2

# Load an image using OpenCV
image = cv2.imread('test.jpeg')

# Downsample the image using OpenCV's resize function
downsampled_image = cv2.pyrDown(image)

# Display the downsampled image
cv2.imshow('Original Image', image)
cv2.imshow('Downsampled Image', downsampled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()