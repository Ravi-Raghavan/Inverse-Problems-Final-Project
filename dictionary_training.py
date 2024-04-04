from rnd_smp_patches import rnd_smp_patch
from patch_pruning import patch_pruning

### Code used to Train Dictionaries
training_image_path = "Data/Training"

dict_size = 512 #Dictionary Size will be 512
lamb = 0.15 #sparsity regularization
patch_size = 5 #size of patches will be 5 x 5
nSmp = 100000 #number of patches to sample
upscale = 2 #upscale factor

#randomly generate patches
Xh, Xl = rnd_smp_patch(training_image_path, '.bmp', patch_size, nSmp, upscale)

#Prune patches with small variance
Xh, Xl = patch_pruning(Xh, Xl, threshold = 10)

print(Xh.shape, Xl.shape)