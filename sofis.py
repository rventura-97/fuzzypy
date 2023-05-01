import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

class sofis:
    
    def __init__(self,L=1,dist='euclidean',predict_metric='distance'):
        self.class_models = []
        self.feat_names = []
        self.L = L
        self.N = 0
        self.dist = dist
        self.predict_metric = predict_metric
        
    def fit_offline(self,X,y):
        # Pre-process data
        X = np.transpose(X)
        # Initialize class sub-models
        self.__init_class_models(y)
        # Train each class model
        for c in np.unique(y):
            self.class_models[c].fit_offline(X[:,y==c]) 
            self.N += self.class_models[c].P.shape[1]
        
    def fit_online(self,X,y):
        X = np.transpose(X)
        
        for k in range(0,X.shape[1]):
            self.class_models[y[k]].update(X[:,k])
        
        return 0
    
    def predict(self,X):
        X = np.transpose(X)
        y_pred = np.zeros(X.shape[1])
        
        for i in range(0,y_pred.size):
            vals = np.zeros(len(self.class_models))
            for c in range(0,vals.size):
                vals[c] = self.class_models[c].predict(X[:,i],self.predict_metric)
                    
            if self.predict_metric=='lambda':     
                y_pred[i] = np.argmax(vals)
            else:
                y_pred[i] = np.argmin(vals)
        
        return y_pred
        
    
    def plot_clouds(self,x1,x2,X,y):
        return 0
    
    def __init_class_models(self,y):
        self.class_models = [class_model(L=self.L,dist=self.dist) for _ in range(0,np.unique(y).size)]
        
        
class class_model:
    
    def __init__(self,L,dist):
        self.L = L
        self.dist = dist
        self.K = []
        self.P = []
        self.S = []
        self.G = []
        self.x = []
        self.Xk = []
        self.Sigma = []
        self.Sigma_inv = []
        self.CumProx = []
        self.CumProxSum = 0
        self.MeanDist = 0
        
        
    def fit_offline(self,X):
        # Initialize global model parameters
        self.K = X.shape[1]
        self.Mu = np.mean(X,axis=1)
        self.X = np.mean([np.dot(X[:,i],np.transpose(X[:,i])) for i in range(0,X.shape[1])])
        self.x = X
        
        if self.dist == 'mahalanobis':
            self.Sigma = np.cov(X)
            self.Sigma_inv = np.linalg.inv(self.Sigma)
            
        
        p_dists, self.MeanDist, self.CumProx, self.CumProxSum = cum_prox(X, self.dist)
            
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
        Phi_D_mmd = np.multiply(S,unimodal_density(Phi, X, self.dist,cum_prox_sum=self.CumProxSum))
        
        # Compute radius of local influence around cloud centers
        self.G = self.__radius_of_influence(p_dists, self.MeanDist, self.L)
        
        # Select most representive clouds
        self.P, self.S = self.__select_clouds(Phi, Phi_D_mmd, self.G, S)
        
         
    def predict(self,x,predict_metric):
        vals = np.zeros(self.P.shape[1])
        if self.dist == 'euclidean':
            if predict_metric == 'lambda':
                vals = np.exp(-np.power(cdist(x.reshape(1,-1),np.transpose(self.P),metric='euclidean'),2))
            else:
                vals = np.power(cdist(x.reshape(1,-1),np.transpose(self.P),metric='euclidean'),2)
            
        if predict_metric == 'lambda':
            val = np.max(vals)
        else:
            val = np.min(vals)
                
        return val
    
    def update(self,x):
        # Update global model parameters
        Mu_k = ((self.K-1)/self.K)*self.Mu + (1/self.K)*x
        X_k = ((self.K-1)/self.K)*self.X + (1/self.K)*np.dot(x,np.transpose(x))
        dists_xk_x = np.squeeze(np.power(cdist(x.reshape(1,-1),np.transpose(self.x),metric='euclidean'),2))
        ###
        CumProx_k = np.append(self.CumProx + dists_xk_x,np.sum(dists_xk_x))
        CumProxSum_k = np.sum(CumProx_k)
        ###
        MeanDist_k = (1/np.power(self.K+1,2))*CumProxSum_k
        G_k = (MeanDist_k/self.MeanDist)*self.G
        
        # Compute sample unimodal density
        
        # Compute cloud unimodal densities
        
        # Evaluate density condition
        
        
        
        self.x = np.column_stack((self.x,x))
        self.K += 1
        
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
        p = np.zeros_like(r)
        p_idx = 0
        for i in range(0,D_mm_r.size):
            if i == 0 and D_mm_r[0] > D_mm_r[1]:
                p[:,p_idx] = r[:,0]
                p_idx += 1
            elif i == D_mm_r.size-1 and D_mm_r[-1] > D_mm_r[-2]:
                p[:,p_idx] = r[:,-1]
                p_idx += 1
            elif D_mm_r[i]>D_mm_r[i-1] and D_mm_r[i]>D_mm_r[i+1]:
                p[:,p_idx] = r[:,i]
                p_idx += 1
        p = p[:,0:p_idx]
        
        dist_X_p_mins = np.argmin(cdist(np.transpose(X),np.transpose(p),metric='euclidean'),axis=1)
        S = [[] for _ in range(p.shape[1])]
        phi = [[] for _ in range(p.shape[1])]
        
        for i in range(0,p.shape[1]):
            S[i] = X[:,dist_X_p_mins==i]
            phi[i] = np.mean(S[i],axis=1)
            S[i] = S[i].shape[1]
        
        S = np.array(S)
        phi = np.transpose(np.array(phi))
        
        return phi, S
        
    def __radius_of_influence(self,pair_dists,avg_dist,L):

        G = np.sum(pair_dists[pair_dists<=avg_dist]) / np.sum(pair_dists<=avg_dist)

        if L > 1:
            for i in range(1,L):
                G = np.sum(pair_dists[pair_dists<=G]) / np.sum(pair_dists<=G)

        return G        


    def __select_clouds(self,Phi,Phi_D_mmd,G,S):
        
        Phi_neigh = [[] for _ in range(Phi.shape[1])]
        dist_Phi = np.power(squareform(pdist(np.transpose(Phi),'euclidean')),2)
        dist_Phi[np.diag_indices(dist_Phi.shape[0])] = np.nan
        
        # Find neighbouring clouds of each cloud
        for i in range(0,len(Phi_neigh)):
            Phi_neigh[i] = dist_Phi[i,:]<=G
                        
        # Find most representative clouds
        P = np.zeros_like(Phi)
        P_S = np.zeros(len(Phi_neigh))
        P_idx = 0
        for i in range(0,len(Phi_neigh)):
            if np.sum(Phi_neigh[i]) > 0:
                if Phi_D_mmd[i] > np.max(Phi_D_mmd[Phi_neigh[i]]):
                    P[:,P_idx] = Phi[:,i]
                    P_S[P_idx] = S[i]
                    P_idx += 1
            else:
                P[:,P_idx] = Phi[:,i]
                P_S[P_idx] = S[i]
                P_idx += 1
    
        P = P[:,0:P_idx]
        P_S = P_S[0:P_idx]
    
        return P, P_S


def unimodal_density(x,Xj,dist_func,cum_prox_sum=None):
    if cum_prox_sum == None:        
        num = np.sum(np.power(np.triu(squareform(pdist(np.transpose(Xj),'euclidean'))),2)) ### retirar triu
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
        p_dists = squareform(np.power(pdist(np.transpose(X),'euclidean'),2))
        p_dists[np.diag_indices(p_dists.shape[0])] = np.nan

        cum_proxs = np.nansum(p_dists,axis=1)
        cum_proxs_sum = np.nansum(cum_proxs)
        
        p_dists = p_dists[np.triu_indices(p_dists.shape[0],k=1)]
        p_dist_mean = np.mean(p_dists)
               
    return p_dists, p_dist_mean, cum_proxs, cum_proxs_sum

    