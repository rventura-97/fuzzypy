import numpy as np
import scipy as scp
from aux_functions import xie_beni_index


class fcm:
    def __init__(self, Nc=2, m=2.0, max_iters=200, min_improv=0.001, rnd_seed=471997,save_history=False):
        self.Nc = Nc
        self.m = m
        self.max_iters = max_iters
        self.min_improv = min_improv
        self.rnd_seed = rnd_seed

        self.V = []
        self.U = []
        self.J = -1*np.ones(self.max_iters)
        self.val_meas = [] 
        self.X = []
        self.save_history = save_history
        self.V_iters = []
        
    def fit(self, X):
        if self.save_history == True:
            self.X = X
        # Set random seed
        np.random.seed(self.rnd_seed)
        # Randomly initialize fuzzy partition matrix U
        self.U = np.random.rand(self.Nc,X.shape[0])
        self.U = np.divide(self.U,np.sum(self.U,0))
        # Optimization loop
        stop_cond = False
        i = 1
        while stop_cond == False:
            U_m = np.power(self.U,self.m)
            # Update cluster centers
            self.V = np.divide(np.matmul(U_m,X), np.repeat(np.sum(U_m,1).reshape(-1,1),X.shape[1],axis=1))
            # Compute distance between samples and each cluster center
            d = np.transpose(scp.spatial.distance.cdist(X,self.V))
            # Update fuzzy partition matrix
            d_m = np.power(np.maximum(d,np.spacing(1)),-2/(self.m-1))
            self.U = np.divide(d_m,np.repeat(np.sum(d_m,axis=0).reshape(1,-1),self.Nc,axis=0))
            # Update cost function
            self.J[i-1] = np.sum(np.multiply(np.power(d,2),U_m))
            # Check stop condition
            if (i>1) and ((i==self.max_iters) or (abs(self.J[i-1]-self.J[i-2])/self.J[i-1]<=self.min_improv)):
                stop_cond = True
                self.J = self.J[0:i]
            else:
                i += 1


class fcm_optimizer:
    
    def __init__(self, Nc_vals, m_vals, num_runs, valid_func, rnd_seed):
        self.Nc_vals = Nc_vals
        self.m_vals = m_vals
        self.num_runs = num_runs
        
        if valid_func == 'xie-beni':
            self.valid_func = xie_beni_index
        
        self.rnd_seed = rnd_seed
        
        self.valid_vals = np.zeros((len(self.Nc_vals),len(self.m_vals),self.num_runs))
        
        self.best_valid_val = []
        self.best_model = []
        
        
    def fit(self, X):
        run_seeds = np.random.randint(low=1,high=1000000,size=(self.num_runs,))
        c = 1
        for i in range(0,len(self.Nc_vals)):
            for j in range(0, len(self.m_vals)):
                for k in range(0, self.num_runs):
                    Nc = self.Nc_vals[i]
                    m = self.m_vals[j]
                    rnd_seed = run_seeds[k]
                    fcm_mod = fcm(Nc=Nc,m=m,rnd_seed=rnd_seed)
                    fcm_mod.fit(X)
                    valid_val = self.valid_func(X, fcm_mod.U, fcm_mod.V, fcm_mod.m)
                    self.valid_vals[i,j,k] = valid_val
                    if c == 1:
                        self.best_valid_val = valid_val
                        self.best_model = fcm_mod
                    else:
                        if valid_val < self.best_valid_val:
                            self.best_valid_val = valid_val
                            self.best_model = fcm_mod
                    c += 1
                    
        
        return self.best_model


        



