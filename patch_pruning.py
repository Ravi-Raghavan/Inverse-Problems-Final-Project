import numpy as np

def patch_pruning(Xh, Xl, threshold):
    variances = np.var(Xh, 0)
    
    idx = variances > threshold
    
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    
    return Xh, Xl