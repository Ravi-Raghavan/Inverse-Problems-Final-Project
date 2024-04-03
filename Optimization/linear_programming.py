import numpy as np
from scipy.optimize import minimize

#Objective Function to Optimize

def objective_function(Z, X, D, lamb):
    residual = X - np.dot(D, Z.reshape(-1, D.shape[1]).T)
    l2_norm = np.linalg.norm(residual)**2
    l1_norm = lamb * np.sum(np.abs(Z))
    return l2_norm + l1_norm

# Example data
X = np.array([[1, 2], [3, 4], [5, 6]])
D = np.array([[0.5, 0.3], [0.2, 0.4], [0.1, 0.2]])
lambda_value = 0.1

# Initial guess for Z
Z0 = np.zeros(D.shape[1] * X.shape[1])

# Minimize the objective function
result = minimize(objective_function, Z0, args=(X, D, lambda_value), method='BFGS')

# Extract the optimal Z
optimal_Z = result.x.reshape(-1, D.shape[1])

print("Optimal Z:")
print(optimal_Z)
