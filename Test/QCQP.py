### Experimentation with Quadratically Constrained Quadratic Program
import cvxpy as cp
import numpy as np
import time

def optimize_D(X, Z):
    # Define dimensions
    n, m = X.shape[0], Z.shape[0]

    # Define the optimization variable
    D = cp.Variable((n, m))

    # Define the objective function: ||X - DZ||^2
    objective = cp.Minimize(cp.norm(X - D @ Z, 'fro')**2)

    # Define the constraints: ||Di||^2 <= 1 for each column Di of D
    constraints = [cp.norm(D[:, j], 'fro') <= 1 for j in range(m)]

    # Define the optimization problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Get the optimal value of D
    D_opt = D.value

    return D_opt

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

# Example usage
start_time = time.time()
Z = np.random.randn(512, 30)  # Example matrix Z
D = np.random.normal(size = (25, 512))

X = D @ Z # Example matrix X
D_opt = quadratic_programming(X, Z, X.shape[0], Z.shape[0], 0.001, 0.0001, 100)
print(D_opt)
end_time = time.time()

print(f"Elapsed Time: {end_time - start_time}")
