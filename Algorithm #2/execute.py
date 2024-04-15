import numpy as np 
import cv2
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import resize
from SR import SR
from backprojection import backprojection
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Set which dictionary you want to use    
Dh = np.load("../Dictionaries/Dh_512_0.15_5.npy")
Dl = np.load("../Dictionaries/Dl_512_0.15_5.npy")

Dh = normalize(Dh)
Dl = normalize(Dl)

### SET PARAMETERS
lmbd = 0.1
patch_size= 5
D_size = 512
US_mag = 3

overlap = 1
lmbd = 0.1
upscale = 3
maxIter = 100

lr_image_path = "../Data/Testing/Lion.jpg"
hr_image_path = "../Data/Testing/Lion_gnd.jpg"

#Read Low Resolution Image. Cv2 reads in BGR order so must be flipped! 
img_lr = cv2.imread(lr_image_path)
img_lr_ori = img_lr #store original low resolution image
img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

# Read and save ground truth image[High Resolution Image]
img_hr = cv2.imread(hr_image_path)
img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

# Change color space for Low Res and High Res Image
img_hr_y = rgb2ycbcr(img_hr)[:, :, 0] #Get Convert RGB to Y, CB, Cr and get Y component for High Resolution Image

img_lr = rgb2ycbcr(img_lr) #Get Convert RGB to Y, CB, Cr for Low Resolution Image
img_lr_y = img_lr[:, :, 0] #Y Component
img_lr_cb = img_lr[:, :, 1] #CB Component
img_lr_cr = img_lr[:, :, 2] #Cr Component

# Upscale chrominance to color SR images
img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)

# Super Resolution via Sparse Representation
img_sr_y = SR(img_lr_y, img_hr_y.shape, upscale, Dh, Dl, lmbd, overlap)
img_sr_y = backprojection(img_sr_y, img_lr_y, maxIter)
    
# Create colored SR images
img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
img_sr = ycbcr2rgb(img_sr)

fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

axes[0].imshow(img_sr)  # You can choose different colormaps ('gray' for grayscale)
axes[0].set_title('SuperResolution Image')  # Set title
axes[0].set_xlabel('Columns')  # Set label for x-axis
axes[0].set_ylabel('Rows')  # Set label for y-axis

axes[1].imshow(cv2.cvtColor(img_lr_ori, cv2.COLOR_BGR2RGB), cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[1].set_title('Original Low Resolution Image')  # Set title
axes[1].set_xlabel('Columns')  # Set label for x-axis
axes[1].set_ylabel('Rows')  # Set label for y-axis


axes[2].imshow(img_hr, cmap='gray')  # You can choose different colormaps ('gray' for grayscale)
axes[2].set_title('Ground Truth High Res Resolution Image')  # Set title
axes[2].set_xlabel('Columns')  # Set label for x-axis
axes[2].set_ylabel('Rows')  # Set label for y-axis

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the figure with subplots