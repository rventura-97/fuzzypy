import numpy as np
from numpy import transpose, dot
from numpy.linalg import norm

omega_0 = 10

class almmo():
    def __init__(self):
        self.K = 0 # iteration
        self.mu_K = [] # global mean
        self.X_K = [] # global average scalar product
        self.N_K = 0 # number of clouds
        self.mu_K_N = [] # cloud focal points
        self.S_K_N = [] # cloud supports
        self.X_K_N = [] # cloud average scalar products
        self.eta_K_N = [] # cloud utilities
        self.A_K_N = [] # cloud consequent parameters
        self.C_K_N = [] # cloud covariance matrices
        
        
    def fit(self, X, y):
        # training loop
        for k in range(0, X.shape[1]):
            self.update(transpose(X[k,:]).reshape((-1,1)), y[k])
        

    def update(self, x_k, y_k):
        if self.K == 0: # initialize model with first sample
            self.K = 1
            self.mu_K = x_k
            self.X_K = dot(transpose(x_k),x_k).item()
            self.N_K = 1
            self.mu_K_N = x_k[:,:,np.newaxis]
            self.S_K_N = np.ones((1,1,1))
            self.X_K_N = dot(transpose(x_k),x_k)[:,:,np.newaxis]
            self.eta_K_N = np.ones((1,1,1))
            self.A_K_N = np.zeros((x_k.size + 1, y_k.size, 1))
            self.C_K_N = omega_0 * np.eye(x_k.size + 1)[:,:,np.newaxis]
        else:
            self.K += 1 # increment current iteration
            self.__update_global_params(x_k) # update global parameters
            D_global = self.__compute_global_densities() # compute global densities
            D_local = self.__compute_local_densities(x_k[:,:,np.newaxis]) # compute local densities
            lambdas = self.__compute_lambdas(D_local) # compute cloud activations
            y_hat = self.__compute_output(x_k, lambdas) # compute system output
            
        
    def __update_global_params(self, x_k):
        self.mu_K = (1/self.K)*((self.K-1)*self.mu_K+x_k) # update global mean
        self.X_K = (1/self.K)*((self.K-1)*self.X_K+dot(transpose(x_k),x_k).item()) # update global scalar product
        
    def __compute_global_densities(self):
        D = np.zeros((self.N_K))
        delta_X = self.X_K - np.power(norm(self.mu_K),2)
        delta_mu = np.power(norm(self.mu_K_N-self.mu_K),2)
        D = 1/(1+delta_mu/delta_X)
        return D
    
    def __compute_local_densities(self, x_k):
        D = np.zeros((self.N_K))
        X_K_N_temp = (self.S_K_N*self.X_K_N+dot(transpose(np.squeeze(x_k,2)),np.squeeze(x_k,2)))/(self.S_K_N+1)
        mu_K_N_temp = (self.S_K_N*self.mu_K_N+x_k)/(self.S_K_N+1)
        delta_X = X_K_N_temp - np.power(norm(mu_K_N_temp,keepdims=True),2)
        delta_mu = np.power(norm(x_k-mu_K_N_temp,keepdims=True),2)
        D = 1/(1+delta_mu/delta_X)
        return D
    
    def __compute_lambdas(self, D):
        if np.sum(D) == 0:
            lambdas = np.ones(self.N_K)/self.N_K
        else:
            lambdas = D/np.sum(D)
        return lambdas
    
    def __compute_output(self, x_k, lambdas):
        u = np.vstack((np.ones((1,1)),x_k))[:,:,np.newaxis]
        y_hat = 0
        return y_hat
    
        