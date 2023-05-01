import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

class sofis:
    
    def __init__(self,L=1,dist='euclidean'):
        self.class_models = []
        self.feat_names = []
        self.L = L
        # if dist=='euclidean':
        #     self.dist = dist_euclidean
        # elif dist=='mahalanobis':
        #     self.dist = dist_mahalanobis
        self.dist = dist
        
    def fit_offline(self,X,y):
        # Pre-process data
        X = np.transpose(X)
        # Initialize class sub-models
        self.__init_class_models(y)
        # Train each class model
        for c in np.unique(y):
            self.class_models[c].fit_offline(X[:,y==c]) 
        
    def fit_online(self,X,y):
        return 0
    
    def predict(self,X):
        X = np.transpose(X)
        y_pred = np.zeros(X.shape[1])
        
        for i in range(0,y_pred.size):
            max_lambdas = np.zeros(len(self.class_models))
            for c in range(0,max_lambdas.size):
                max_lambdas[c] = self.class_models[c].max_lambda(X[:,i])
            y_pred[i] = np.argmax(max_lambdas)
        
        return y_pred
        
    
    def plot_clouds(self,x1,x2,X,y):
        return 0
    
    def __init_class_models(self,y):
        self.class_models = [class_model(L=self.L,dist=self.dist) for _ in range(0,np.unique(y).size)]
        
        
class class_model:
    
    def __init__(self,L,dist):
        self.L = L
        self.dist = dist
        self.P = []
        self.Sigma = []
        self.Sigma_inv = []
        self.CumProx = []
        self.CumProxSum = 0
        
    def fit_offline(self,X):
        # Initialize global model parameters
        self.Mu = np.mean(X,axis=1)
        self.X = np.mean([np.dot(X[:,i],np.transpose(X[:,i])) for i in range(0,X.shape[1])])
        
        if self.dist == dist_mahalanobis:
            self.Sigma = np.cov(X)
            self.Sigma_inv = np.linalg.inv(self.Sigma)
            self.dist = lambda x1,x2: dist_mahalanobis(x1,x2,self.Sigma_inv)
            
        self.CumProx = cum_prox(X, self.dist)
        self.CumProxSum = np.sum(self.CumProx)
        
        
        # Find unique samples and their frequencies
        U,Uf = np.unique(np.transpose(X),axis=0,return_counts=True)
        U = np.transpose(U)
        
        # Compute multimodel densities for each unique sample
        D_mm = np.multiply(Uf,unimodal_density(U, X, self.dist,cum_prox_sum=self.CumProxSum))
        
        # Rank unique data samples
        r, D_mm_r = self.__rank_samples(D_mm, U,self.dist)
        

        # Initialize clouds
        Phi, S = self.__init_clouds(r, D_mm_r, X)
        
        # Compute multimodal densities at cloud centers
        Phi_D_mmd = [len(S[i])*unimodal_density(Phi[i], X,self.dist,cum_prox_sum=self.CumProxSum) for i in range(0,len(Phi))]
        
        # Compute radius of local influence around cloud centers
        G = self.__radius_of_influence(X,self.L)
        
        # Select most representive clouds
        self.P = np.column_stack(self.__select_clouds(Phi, Phi_D_mmd, G))
        
    def max_lambda(self,x):
        lambda_x_p = np.zeros(self.P.shape[1])
        for i in range(0,lambda_x_p.size):
            lambda_x_p[i] = np.exp(-np.power(self.dist(x,self.P[:,i]),2))
        return np.max(lambda_x_p)
        
    def __rank_samples(self,D_mm,U,dist_func):
        D_mm_r = np.zeros_like(D_mm)
        r = np.zeros_like(U)
        
        k_1_idx = np.argmax(D_mm) 
        r[:,0] = U[:,k_1_idx]
        D_mm_r[0] = D_mm[k_1_idx]
        
        pair_dists = squareform(pdist(np.transpose(U)))
        pair_dists[np.diag_indices(pair_dists.shape[0])] = np.nan
        
        for k in range(1,D_mm.size):
            k_idx = np.nanargmin(pair_dists[k_1_idx,:])
            r[:,k] = U[:,k_idx]
            D_mm_r[k] = D_mm[k_idx]
            pair_dists[:,k_1_idx] = np.nan
            pair_dists[k_1_idx,:] = np.nan
            k_1_idx = k_idx
            

            
        return r, D_mm_r

    def __init_clouds(self, r, D_mm_r,X):
        p = []
        
        if D_mm_r[0] > D_mm_r[1]:
            p.append(r[:,0])
        
        for i in range(1,D_mm_r.size-1):
            if D_mm_r[i]>D_mm_r[i-1] and D_mm_r[i]>D_mm_r[i+1]:
                p.append(r[:,i])
        
        if D_mm_r[-1] > D_mm_r[-2]:
            p.append(r[:,-1])
            
        p = np.column_stack(p)
        
        
        S = [[] for _ in range(p.shape[1])]
        
        for i in range(0,X.shape[1]):
            dist_x_p = np.zeros(p.shape[1])
            for j in range(0,p.shape[1]):
                dist_x_p[j] = self.dist(X[:,i],p[:,j])
            S[np.argmin(dist_x_p)].append(X[:,i])
        
        
        phi = [np.mean(np.column_stack(S[i]),axis=1) for i in range(0,len(S))]
        
        return phi, S
        
    def __radius_of_influence(self,X,L):
        N = X.shape[1]
        pair_dists = np.zeros(int(N*(N-1)/2))
        c = 0
        for i in range(0,N):
            for j in range(0,N):
                if j > i:
                    pair_dists[c] = np.power(self.dist(X[:,i],X[:,j]),2)
                    c += 1
        
        avg_dist = np.mean(pair_dists)   

        G = np.sum(pair_dists[pair_dists<=avg_dist]) / np.sum(pair_dists<=avg_dist)

        if L > 1:
            for i in range(1,L):
                G = np.sum(pair_dists[pair_dists<=G]) / np.sum(pair_dists<=G)

        return G        


    def __select_clouds(self,Phi,Phi_D_mmd,G):
        P = []
        Phi_neigh = [[] for _ in range(len(Phi))]
        
        # Find neighbouring clouds of each cloud
        for i in range(0,len(Phi)):
            for j in range(0,len(Phi)):
                if i != j:
                    if np.power(self.dist(Phi[i],Phi[j]),2)<=G:
                        Phi_neigh[i].append(j)
                        
        # Find most representative clouds
        for i in range(0,len(Phi)):
            if Phi_neigh[i] != []:
                if Phi_D_mmd[i] > np.max([Phi_D_mmd[j] for j in Phi_neigh[i]]):
                    P.append(Phi[i])
            else:
                P.append(Phi[i])
    
        return P


def unimodal_density(x,Xj,dist_func,cum_prox_sum=None):
    if cum_prox_sum == None:        
        num = np.sum(np.power(np.triu(squareform(pdist(np.transpose(Xj),'euclidean'))),2)) 
    else:
        num = cum_prox_sum
            
    if x.size == x.shape[0]:        
        den = 2*Xj.shape[1]*np.sum(np.power(cdist(x.reshape(1,-1),np.transpose(Xj),metric='euclidean'),2))
        
    else:
        den = 2*Xj.shape[1]*np.sum(np.power(cdist(np.transpose(x),np.transpose(Xj),metric='euclidean'),2),axis=1)
        
    D = num/den 
    return D

def cum_prox(X,dist_func,cov_inv=None):    
    if dist_func == 'euclidean':
        vals = np.sum(np.power(np.triu(squareform(pdist(np.transpose(X),'euclidean'))),2),axis=1)          
    return vals

def dist_euclidean(x1,x2):
    return np.linalg.norm(x2-x1)

def dist_mahalanobis(x1,x2,C_inv):
    return np.sqrt(np.matmul(np.matmul(x1-x2,C_inv),x1-x2))
    
    