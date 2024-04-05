import scipy.io as sio
import numpy as np

# Load .mat file
mat_data = sio.loadmat('D_512_0.15_5.mat')

Dh, Dl = mat_data['Dh'], mat_data['Dl']
print(Dh.shape, Dl.shape)

# Save Dh and Dl as .npy files
np.save('Dh_512_0.15_5.npy', Dh)
np.save('Dl_512_0.15_5.npy', Dl)

# Load .mat file
mat_data = sio.loadmat('D_1024_0.15_5.mat')

Dh, Dl = mat_data['Dh'], mat_data['Dl']
print(Dh.shape, Dl.shape)

# Save Dh and Dl as .npy files
np.save('Dh_1024_0.15_5.npy', Dh)
np.save('Dl_1024_0.15_5.npy', Dl)

