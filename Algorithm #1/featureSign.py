import numpy as np 
from scipy import sparse

def featureSign(y, A, lamb):
    EPS = 1e-9 #epsilon for comparisons
    
    x = np.zeros((A.shape[1], 1)) #initialize x
    theta = np.sign(x) #initialize theta
    y = y.reshape((-1, 1)) #reshape y to a column vector for ease of calculations
    gradient = (2 * (A.T @ A @ x)) - (2 * (A.T @ y))  #initial value of gradient 
    active_set = np.zeros(shape = x.shape) #initialize active set
    
    #Iterate
    for _ in range(10):
        gradient_max_idx = np.argmax(np.abs(gradient) * (x == 0))
        Ai = A[:, gradient_max_idx].flatten().reshape((-1, 1))
        
        if gradient[gradient_max_idx] > lamb + EPS:
            x[gradient_max_idx] = (lamb - gradient[gradient_max_idx]) / (2 * (np.linalg.norm(Ai) ** 2))
            theta[gradient_max_idx] = -1
            active_set[gradient_max_idx] = 1
        elif gradient[gradient_max_idx] < -lamb - EPS:
            x[gradient_max_idx] = (-lamb - gradient[gradient_max_idx]) / (2 * (np.linalg.norm(Ai) ** 2))
            theta[gradient_max_idx] = 1
            active_set[gradient_max_idx] = 1
        else:
            if np.all(x == 0):
                break
        
        #Feature-Sign Step
        for _ in range(10):
            x_non_zero_idxes = (x != 0).flatten()
            A_hat = A[:, x_non_zero_idxes]
            x_hat = x[x_non_zero_idxes, :]
            theta_hat = theta[x_non_zero_idxes, :]
            
            x_hat_new = np.linalg.lstsq(A_hat.T @ A_hat, A_hat.T @ y - 0.5 * lamb * theta_hat, rcond=None)[0]            
            loss_new = np.linalg.norm(y - A_hat @ x_hat_new) ** 2 + lamb * np.sum(abs(x_hat_new))
            
            idx_hats = np.where(x_hat * x_hat_new < 0) [0]
            if np.all(idx_hats == 0):
                x_hat = x_hat_new
                
                x[x_non_zero_idxes, :] = x_hat
                theta = np.sign(x)
                active_set = (x != 0)
                
            else:
                x_min = x_hat_new
                loss_min = loss_new
                diff = x_min - x_hat
                delta = diff / x_hat
                
                for zd in idx_hats.T:
                    x_s = x_hat - diff / delta[zd]
                    x_s[zd] = 0
                    x_s_idx = (x_s == 0).flatten()
                    
                    A_hat_s_idx = A_hat[:, x_s_idx]
                    x_s_modified = x_s[x_s_idx, :]
                    
                    loss = np.linalg.norm(y - A_hat_s_idx @ x_s_modified) ** 2 + lamb * np.sum(abs(x_s_modified))
                    if loss < loss_min:
                        x_min = x_s
                        loss_min = loss
                
                x[x_non_zero_idxes, :] = x_min
                theta = np.sign(x)
                active_set = (x != 0)     
            
            gradient = (2 * (A.T @ A @ x)) - (2 * (A.T @ y))
            non_zero_idxs = (x != 0).flatten()
            if np.all(gradient[non_zero_idxs, :] + lamb * theta[non_zero_idxs, :] == 0):
                break
            
        zero_idxs = (x == 0).flatten()
        if np.all(gradient[zero_idxs, :] <= lamb):
            break
                    
    return x