### File to Jointly Train Dictionaries
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

def train_coupled_dict(Xh, Xl, dict_size, step_size, lamb, threshold):
    print("STARTING TRAINING")
    N, M = Xh.shape[0], Xl.shape[0]
    
    a1 = 1 / np.sqrt(N)
    a2 = 1 / np.sqrt(M)
    
    #Initialize Xc
    Xc = np.concatenate((a1 * Xh, a2 * Xl), axis = 0)
    print(f"Xc shape: {Xc.shape}")
    
    #Initialize D as a random Gaussian Matrix
    Dc = np.random.normal(size = (N + M, dict_size))
    Dc = normalize(Dc)
    print(f"Dc shape: {Dc.shape}")
    
    #cap maximum iterations at 100
    for iter in range(100):
        Z = linear_programming(Xc, Dc, Dc.shape[1], Xc.shape[1], step_size, lamb, threshold, 100)
        Dc = quadratic_programming(Xc, Z, Dc.shape[0], Dc.shape[1], step_size, threshold, 100)
        print(f"Completed Iteration {iter + 1}/100")
    
    return Dc
        
#prox operator
def prox(x, alpha):
    return np.piecewise(x, [x < -alpha, (x >= -alpha) & (x <= alpha), x >= alpha], [lambda x: x + alpha, 0, lambda x: x - alpha])

## Solve the Linear Programming Portion of Joint Dictionary Training
## Goal: Find Z that minimizes || X - DZ||_2^2 + lambda * ||Z||_1
def linear_programming(X: np.ndarray, D: np.ndarray, Zr, Zc, step_size, lamb, threshold, max_iter):
    Z = np.random.normal(size = (Zr, Zc))
    
    #Run Proximal Gradient Descent
    loss = (np.linalg.norm(X - (D @ Z)) ** 2) + (lamb * np.sum(np.abs(Z)))
    
    for iter in range(max_iter):
        grad = (-2 * (D.T @ X)) + (2 * (D.T @ D @ Z))
        
        #Update Z
        Z = Z - (step_size * grad)
        Z = prox(Z, step_size * lamb)
        
        loss = (np.linalg.norm(X - (D @ Z)) ** 2) + (lamb * np.sum(np.abs(Z)))
        if np.linalg.norm(grad) <= threshold:
            break
    
    print(f"Loss at Iteration {iter} = {loss}, Magnitude of Gradient = {np.linalg.norm(grad)}")
    return Z

def normalize(D: np.ndarray):
    column_norms = np.linalg.norm(D, axis=0, keepdims = True)
    normalized_matrix = D / np.maximum(column_norms, 1)
    return normalized_matrix

def quadratic_programming(X: np.ndarray, Z: np.ndarray, Dr, Dc, step_size, threshold, max_iter):    
    #Run Projected Gradient Descent
    D = np.random.normal(size = (Dr, Dc))
    D = normalize(D)
    loss = (np.linalg.norm(X - (D @ Z)) ** 2)
    
    for iter in range(max_iter):
        grad = (-2 * (X @ Z.T)) + (2 * (D @ Z @ Z.T))
        
        D = D - (step_size * grad)
        D = normalize(D)
        loss = (np.linalg.norm(X - (D @ Z)) ** 2)
        
        if np.linalg.norm(grad) <= threshold:
            break
    
    print(f"Loss at Iteration {iter} = {loss}, Magnitude of Gradient = {np.linalg.norm(grad)}")
    return D
    

# D = np.random.normal(size = (25, 512))
# Z = np.random.normal(size = (512, 30))
# X = D @ Z

# print("Testing Linear Programming")
# linear_programming(X, D, Z.shape[0], Z.shape[1], 0.001, 0, 0.0001, 100)

# print("Testing Quadratic Programming")
# D = np.random.normal(size = (25, 512))
# D = normalize(D)

# Z = np.random.normal(size = (512, 30))
# X = D @ Z
# quadratic_programming(X, Z, D.shape[0], D.shape[1], 0.001, 0.0001, 100)