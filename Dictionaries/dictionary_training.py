from rnd_smp_patches import rnd_smp_patch
from patch_pruning import patch_pruning
from train_coupled_dict import train_coupled_dict
import numpy as np

### Code used to Train Dictionaries
training_image_path = "../Data/Training"

dict_size = 128 #Dictionary Size will be 128
lamb = 0.15 #sparsity regularization
patch_size = 3 #size of patches will be 3 x 3
nSmp = 1000 #number of patches to sample
upscale = 2 #upscale factor

#randomly generate patches
Xh, Xl = rnd_smp_patch(training_image_path, '.bmp', patch_size, nSmp, upscale)

#Prune patches with small variance
Xh, Xl = patch_pruning(Xh, Xl, threshold = 10)

print(Xh.shape, Xl.shape)

#Joint Dictionary Training
step_size = 0.001
threshold = 0.0001
Dc = train_coupled_dict(Xh, Xl, dict_size, step_size, lamb, threshold)

N, M = Xh.shape[0], Xl.shape[0]
Dh, Dl = Dc[:N], Dc[N:]

#Save Dh and Dl to npy files
np.save('Dh.npy', Dh)
np.save('Dl.npy', Dl)