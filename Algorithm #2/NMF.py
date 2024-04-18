import numpy as np

#Non-Negative Matrix Factorization
def NMF(X: np.ndarray, iterations = 100):
    n, m = X.shape
    
    r = int((n * m) / (n + m))
    
    U = np.random.normal(size = (n, r))
    V = np.random.normal(size = (r, m))
    
    for _ in range(iterations):
        DeltaV = ((U.T @ X) / (U.T @ U @ V))
        DeltaU = ((X @ V.T) / (U @ V @ V.T))
        
        V *= DeltaV
        U *= DeltaU
    
    return U