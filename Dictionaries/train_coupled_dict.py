### File to Jointly Train Dictionaries
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn import linear_model
import cvxpy as cp

def train_coupled_dict(Xh, Xl, dict_size, step_size, lamb, threshold, max_iter):
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
    
    #cap maximum iterations at max_iter
    for iter in range(max_iter):
        # Z = linear_programming(Xc, Dc, Dc.shape[1], Xc.shape[1], step_size, lamb, threshold, 100)
        Z = lasso_optimization(Xc, Dc, lamb)
        
        Xc_pred = Dc @ Z
        print(f"Iteration {iter + 1}/{max_iter} Linear Programming Stat: {np.linalg.norm(Xc - Xc_pred) / np.linalg.norm(Xc)}")
        
        Dc = quadratic_programming(Xc, Z, Dc.shape[0], Dc.shape[1], step_size, threshold, 30)
        # Dc = quadratic_programming_CVXPY(Xc, Z)
        
        Xc_pred = Dc @ Z
        print(f"Iteration {iter + 1}/{max_iter} Quadratic Programming Stat: {np.linalg.norm(Xc - Xc_pred) / np.linalg.norm(Xc)}")
        print(f"Completed Iteration {iter + 1}/{max_iter}")
    
    return Dc

#Lasso Optimization
def lasso_optimization(Xc, Dc, lamb):
    # print("Beginning Lasso Optimization")
    clf = linear_model.Lasso(alpha = lamb, max_iter = 100000, fit_intercept = False)
    # print("Initialized Lasso Model")
    clf.fit(Dc, Xc)
    # print("Fitted lasso model")
    return clf.coef_.T
        
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
    
    # print(f"Loss at Iteration {iter} = {loss}, Magnitude of Gradient = {np.linalg.norm(grad)}")
    return Z

def normalize(D: np.ndarray):
    norms = np.linalg.norm(D, axis=0)  # Calculate column norms
    mask = norms > 1  # Find columns with norms > 1
    D[:, mask] /= norms[mask]  # Normalize only columns with norms > 1
    return D

def quadratic_programming(X: np.ndarray, Z: np.ndarray, Dr, Dc, step_size, threshold, max_iter):    
    #Run Projected Gradient Descent
    D = np.random.normal(size = (Dr, Dc))
    D = normalize(D)
    loss = (np.linalg.norm(X - (D @ Z)) ** 2) / (np.linalg.norm(X))
    
    for iter in range(max_iter):
        grad = (-2 * (X @ Z.T)) + (2 * (D @ Z @ Z.T))
        
        D = D - (step_size * grad)
        D = normalize(D)
        loss = (np.linalg.norm(X - (D @ Z)) ** 2) / (np.linalg.norm(X))
        
        if np.linalg.norm(grad) <= threshold:
            break
    
        # print(f"Loss at Iteration {iter} = {loss}, Magnitude of Gradient = {np.linalg.norm(grad)}")
    return D
    
#Quadratic Programing
def quadratic_objective_function(D_flat, X, Z):
    # Reshape the flattened D to its original shape
    D = D_flat.reshape(X.shape[0], -1)
    # Compute the objective function
    return np.linalg.norm(X - np.dot(D, Z)) ** 2

def quadratic_constraints(D_flat):
    # Reshape the flattened D to its original shape
    D = D_flat.reshape(D_flat.shape[0], -1)
    # Compute the norm squared for each column of D
    norm_squared = np.sum(D**2, axis=0)
    # Return the constraint function as a vector
    return norm_squared - 1

def quadratic_programming_scipy(X, Z):
    # Initial guess for D
    initial_guess_D = np.random.rand(X.shape[0] * Z.shape[0])

    # Define additional arguments for the objective and constraint functions
    args = (X, Z)

    # Minimize the objective function with the constraint
    result = minimize(quadratic_objective_function, initial_guess_D, args=args, method='BFGS', constraints={'type': 'ineq', 'fun': quadratic_constraints})

    # Extract the optimized D
    optimized_D = result.x.reshape(X.shape[0], Z.shape[0])
    
    return optimized_D

def quadratic_programming_CVXPY(X, Z):
    X = cp.Constant(X)  # Assuming X is an n x p numpy array
    Z = cp.Constant(Z)  # Assuming Z is an m x p numpy array
    
    # Define the optimization variables
    D = cp.Variable((X.shape[0], Z.shape[0]))  # Assuming n x m matrix D

    # Define the objective function
    objective = cp.Minimize(cp.sum_squares(X - D @ Z))

    # Define the constraints using numpy operations
    norm_squared = cp.sum(D**2, axis=0)  # Calculate the norm squared of each column of D
    constraints = [cp.norm(D, 'fro', axis=(0, 1)) <= np.sqrt(D.shape[1])]

    # Formulate the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Get the optimized D
    optimized_D = D.value
    
    return optimized_D


# D = np.random.normal(size = (125, 512))
# Z = np.random.normal(size = (512, 96))
# X = D @ Z

# test_Z = lasso_optimization(X, D, 0.15)
# print(test_Z.shape)

# X_pred = D @ test_Z

# print(np.linalg.norm(X - X_pred) / np.linalg.norm(X))

# print("Testing Linear Programming")
# linear_programming(X, D, Z.shape[0], Z.shape[1], 0.001, 0, 0.0001, 100)

# print("Testing Quadratic Programming")
# D = np.random.normal(size = (25, 512))
# D = normalize(D)

# Z = np.random.normal(size = (512, 30))
# X = D @ Z
# quadratic_programming(X, Z, D.shape[0], D.shape[1], 0.001, 0.0001, 100)