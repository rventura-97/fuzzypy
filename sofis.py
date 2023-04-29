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
        p = identify_prototypes(r, D_mm_r)    
        
        # Initialize clouds
        Phi, S = init_clouds(p, X)
        
        # Compute multimodal densities at cloud centers
        Phi_D_mmd = [len(S[i])*unimodal_density(Phi[i], X) for i in range(0,len(Phi))]
        
        # Compute radius of local influence around cloud centers
        G = radius_of_influence(X,2)
        
        # Select most representive clouds
        P = select_clouds(Phi, Phi_D_mmd, G)
        
        return 0


def select_clouds(Phi,Phi_D_mmd,G):
    P = []
    Phi_neigh = [[] for _ in range(len(Phi))]
    
    # Find neighbouring clouds of each cloud
    for i in range(0,len(Phi)):
        for j in range(0,len(Phi)):
            if i != j:
                if np.linalg.norm(Phi[i]-Phi[j])<=G:
                    Phi_neigh[i].append(j)
                    
    # Find most representative clouds
    for i in range(0,len(Phi)):
        if Phi_neigh[i] != []:
            if Phi_D_mmd[i] > np.max([Phi_D_mmd[j] for j in Phi_neigh[i]]):
                P.append(Phi[i])
        else:
            P.append(Phi[i])

    return P


def radius_of_influence(X,L):
    N = X.shape[1]
    pair_dists = np.zeros(int(N*(N-1)/2))
    c = 0
    for i in range(0,N):
        for j in range(0,N):
            if j > i:
                pair_dists[c] = np.linalg.norm(X[:,i]-X[:,j])
                c += 1
    
    pair_dists_2 = np.power(pair_dists,2)
    avg_dist = np.mean(pair_dists)   

    G = np.sum(pair_dists_2[pair_dists_2<=avg_dist]) / np.sum(pair_dists<avg_dist)

    if L > 1:
        for i in range(1,L):
            G = np.sum(pair_dists_2[pair_dists_2<=G]) / np.sum(pair_dists<G)

    return G

def unimodal_density(x,X):
    num = 0
    K = X.shape[1]
    for l in range(0,K):
        for j in range(0,K):
            num += np.power(np.linalg.norm(X[:,l]-X[:,j]),2)
            
    den = 0
    for j in range(0,K):
        den += np.power(np.linalg.norm(x-X[:,j]),2)
    
    D = num/(2*K*den)
    
    return D


def init_clouds(p,X):
    
    S = [[] for _ in range(p.shape[1])]
    
    for i in range(0,X.shape[1]):
        S[np.argmin(np.linalg.norm(X[:,i].reshape(-1,1)-p,axis=0))].append(X[:,i])
    
    
    phi = [np.mean(np.column_stack(S[i]),axis=1) for i in range(0,len(S))]
    
    return phi, S
    


def identify_prototypes(r, D_mm_r):
    p = []
    for i in range(1,D_mm_r.size-1):
        if D_mm_r[i]>D_mm_r[i-1] and D_mm_r[i]>D_mm_r[i+1]:
            p.append(r[:,i])
    
    p = np.column_stack(p)
    
    return p
    

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
    
    