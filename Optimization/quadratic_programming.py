import numpy as np
from scipy.optimize import minimize

def objective_function(D_flat, X, Z):
    D = D_flat.reshape(X.shape[0], Z.shape[0])
    residual = X - D @ Z
    return np.linalg.norm(residual)**2

def constraint_function(D_flat):
    D = D_flat.reshape(X.shape[0], Z.shape[0])
    return 1 - np.sum(np.square(np.linalg.norm(D, axis=0)))

# Example data
X = np.array([[1, 2], [3, 4], [5, 6]])
Z = np.array([[1, 2], [3, 4]])

# Initial guess for D
D0 = np.random.rand(X.shape[0] * Z.shape[0])

# Define the constraint dictionary
constraint = {'type': 'ineq', 'fun': constraint_function}

# Minimize the objective function subject to the constraint
result = minimize(objective_function, D0, args=(X, Z), constraints=constraint)

# Extract the optimal D
optimal_D = result.x.reshape(X.shape[0], Z.shape[0])

print("Optimal D:")
print(optimal_D)
