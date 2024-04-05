import numpy as np
from scipy.optimize import minimize

def objective_function_Z(Z_flat, X, D, lamb):
    Z = Z_flat.reshape(D.shape[1], X.shape[1])
    residual = X - D @ Z
    l2_norm = np.linalg.norm(residual)**2
    l1_norm = lamb * np.sum(np.abs(Z))
    return l2_norm + l1_norm

def objective_function_D(D_flat, X, Z):
    D = D_flat.reshape(X.shape[0], Z.shape[0])
    residual = X - D @ Z
    return np.linalg.norm(residual)**2

def constraint_function_D(D_flat, X, Z):
    D = D_flat.reshape(X.shape[0], Z.shape[0])
    return 1 - np.sum(np.square(np.linalg.norm(D, axis=0)))

def linear_programming_subroutine(Xc, Dc, lamb):
    print("Beginning Linear Programming Subroutine")
    Z0 = np.zeros(Dc.shape[1] * Xc.shape[1])
    
    # Minimize the objective function
    result = minimize(objective_function_Z, Z0, args=(Xc, Dc, lamb), method='BFGS')

    # Extract the optimal Z
    optimal_Z = result.x.reshape(Dc.shape[1], Xc.shape[1])
    return optimal_Z

def quadratic_programming_subroutine(Xc, Z):
    print("Beginning Quadratic Programming Subroutine")
    
    # Initial guess for D
    D0 = np.random.rand(Xc.shape[0] * Z.shape[0])

    # Define the constraint dictionary
    cons = []
    for idx in range(Z.shape[0]):
        constraint = {'type': 'ineq', 'args': (Xc, Z),  'fun': lambda D_flat, X, Z: 1 - (np.square(np.linalg.norm(D_flat.reshape(X.shape[0], Z.shape[0]), axis = 0)))[idx]}
        cons.append(constraint)
    
    # Minimize the objective function subject to the constraint
    result = minimize(objective_function_D, D0, args=(Xc, Z), constraints = cons)

    # Extract the optimal D
    optimal_D = result.x.reshape(Xc.shape[0], Z.shape[0])
    return optimal_D

def train_coupled_dict(Xh, Xl, dict_size, lamb):
    print("STARTING TRAINING")
    N, M = Xh.shape[0], Xl.shape[0]
    
    a1 = 1 / np.sqrt(N)
    a2 = 1 / np.sqrt(M)
    
    #Initialize Xc
    Xc = np.concatenate((a1 * Xh, a2 * Xl), axis = 0)
    print(f"Xc shape: {Xc.shape}")
    
    #Initialize D as a random Gaussian Matrix
    Dc = np.random.normal(size = (N + M, dict_size))
    column_norms = np.linalg.norm(Dc, axis=0, keepdims = True)
    Dc = Dc / column_norms
    print(f"Dc shape: {Dc.shape}")
    
    #cap maximum iterations at 5000
    for iter in range(5000):
        print("-------------------------------------------")
        Z_star = linear_programming_subroutine(Xc, Dc, lamb)
        D_star = quadratic_programming_subroutine(Xc, Z_star)
        Dc = D_star
        print(f"Completed Iteration: {iter + 1}")