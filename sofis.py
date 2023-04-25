import numpy as np

class sofis:
    
    def __init__(self):
        self.class_models = []
        self.feat_names = []
        
    def fit(self,X,y):
        # Pre-process data
        X = np.transpose(X)
        
        # Initialize class sub-models
        self.__init_class_models(y)
        
        for c in np.unique(y):
            self.class_models[c].fit(X[:,y==c]) 
        
    
    def __init_class_models(self,y):
        self.class_models = [class_model()]*np.unique(y).size
        
        
class class_model:
    
    def __init__(self):
        self.label = ''
        
        self.P = []
        
    def fit(self,X):
        # Find unique samples and their frequencies
        U,Uf = np.unique(np.transpose(X),axis=0,return_counts=True)
        U = np.transpose(U)
        
        # Compute multimodel densities for each unique sample
        D_mm = multimod_density(X, U, Uf)
        
        # Rank unique data samples
        r, D_mm_r = rank_samples(D_mm, U)
        
        # Identify prototypes
            
        
        
        return 0


def identify_prototypes(r, D_mm_r):
    
    
    return 0
    

def rank_samples(D_mm,U):
    D_mm_r = np.zeros_like(D_mm)
    r = np.zeros_like(U)
    
    D_mm_max_idx = np.argmax(D_mm) 
    r[:,0] = U[:,D_mm_max_idx]
    U = np.delete(U,(D_mm_max_idx),axis=1)
    D_mm_r[0] = D_mm[D_mm_max_idx]
    
    for k in range(0,D_mm.size-1):
        dist_k = np.zeros(U.shape[1])
        for i in range(0,dist_k.size):
            dist_k[i] = np.linalg.norm(r[:,k]-U[:,i])
        min_dist_idx = np.argmin(dist_k)
        r[:,k+1] = U[:,min_dist_idx]
        D_mm_r[k+1] = D_mm[min_dist_idx]
        U = np.delete(U,(min_dist_idx),axis=1)
        
    return r, D_mm_r

def multimod_density(X,U,Uf):
    D = np.zeros(U.shape[1])
    K = np.sum(Uf)
    
    # Compute numerator value
    num_val = 0
    for i in range(0,X.shape[1]):
        for j in range(0,X.shape[1]):
            if i!=j:
                num_val += np.power(np.linalg.norm(X[:,i]-X[:,j]),2)  
                
    # Compute multimodal densities
    for i in range(0,U.shape[1]):
        den_val = 0
        for j in range(0,X.shape[1]):
            den_val += np.power(np.linalg.norm(U[:,i]-X[:,j]),2)
        D[i] = (Uf[i]*num_val)/(2*K*den_val)
    
    
    return D
    
    