import numpy as np
import scipy as scp

def xie_beni_index(X,U,V,m):
    # Compute distance between samples and each cluster center
    d = np.transpose(scp.spatial.distance.cdist(X,V))
    # Compute cost function
    J = np.sum(np.multiply(np.power(d,2),np.power(U,m)))
    # Compute minimum inter-cluster distance
    d_2_min = np.min(np.power(scp.spatial.distance.pdist(V),2))
    # Compute Xie-Beni index
    xb_idx = J/(V.shape[0]*d_2_min)
    
    return xb_idx

def fit_gauss_mf(x,c,m):
    sigma = np.sqrt(np.divide(-np.power(x-c,2),2*np.log(m)))
    sigma = np.sum(sigma)/x.size
    return sigma

def gauss_mf(x,c,sigma):
    return np.exp(-0.5*np.power((x-c)/sigma,2))

def gen_gauss_mf(x,c,sigma):
    return 0

def trap_mf():
    return 0

def fit_mf():
    
    return 0
    