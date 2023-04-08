import numpy as np
import matplotlib.pyplot as plt
from fcm import fcm
from aux_functions import fit_gauss_mf, gauss_mf

class tsk:
    
    def __init__(self, learn_mode='reg', mf_func_type='gauss', conseq_order='linear', **learn_params):
        self.learn_mode = learn_mode
        self.mf_func_type = mf_func_type
        self.conseq_order = conseq_order
        self.num_of_rules = []
        self.input_params = []
        self.output_params = []
        self.fit_type = []
        self.partition_method = []
        self.partition_method_params = []
        
        
        if 'partition_method' in learn_params.keys() and\
            'partition_method_params' in learn_params.keys():
                self.fit_type = 'fully_specified'  
                self.partition_method = learn_params['partition_method']
                self.partition_method_params = learn_params['partition_method_params']
        
        
    def fit(self,X,y):  
        # Format the data
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        # Cluster the data
        if self.fit_type == 'fully_specified':
            if self.partition_method == 'fcm':
                clust = fcm(Nc=self.partition_method_params['Nc'],\
                            m=self.partition_method_params['m'])
                clust.fit(np.column_stack((X,y)))
                self.num_of_rules = clust.V.shape[0]
                
        
        # Initialize rule parameters for the inputs
        if self.mf_func_type=='gauss':
            self.input_params = np.zeros((X.shape[1],2,clust.V.shape[0])) # [dim, #mf_params, #rules]
        
        # Obtain antecedent membership functions
        for r in range(0,self.input_params.shape[2]): # For each rule
            for d in range(0,self.input_params.shape[0]): # For each dimension
                self.input_params[d,0,r] = clust.V[r,d]
                self.input_params[d,1,r] = fit_gauss_mf(np.squeeze(X[:,d]),\
                                           np.squeeze(clust.V[r,d]),\
                                           np.squeeze(clust.U[r,:]))
                    
        # Initialize rule parameters for the outputs
        if self.conseq_order=='linear':
            self.output_params = np.zeros((X.shape[1]+1,clust.V.shape[0]))
                    
        # Obtain consequent parameters
        Xe = np.column_stack((X,np.ones(X.shape[0])))
        W = self.__compute_activations(X)
        for k in range(0,self.output_params.shape[1]):
            # Solve local least squares problem for each rule
            self.output_params[:,k] = np.squeeze(np.matmul(np.matmul(np.matmul(\
                                      np.linalg.inv(np.matmul(np.matmul(\
                                      np.transpose(Xe),W[:,:,k]),Xe)),\
                                      np.transpose(Xe)),W[:,:,k]),\
                                      y.reshape(-1,1)))
        
    def __compute_activations(self, X):
        W = np.zeros((X.shape[0],X.shape[0],self.output_params.shape[1]))

        for r in range(0,self.input_params.shape[2]): # For each rule
            for n in range(0,X.shape[0]): # For each sample
                mf_activations = np.zeros(self.input_params.shape[0])
                for d in range(0,self.input_params.shape[0]):  # For each dimension
                    mf_activations[d] = gauss_mf(X[n,d],self.input_params[d,0,r], self.input_params[d,1,r])
                W[n,n,r] = np.prod(mf_activations)
        
        return W
    
    def predict(self, X):
        # Format the data
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        Xe = np.column_stack((X,np.ones(X.shape[0])))
        W = self.__compute_activations(X)
        
        y_pred = np.zeros(X.shape[0])
        
        for n in range(0,Xe.shape[0]):
            for r in range(0,self.output_params.shape[1]):
                y_pred[n] = y_pred[n] + W[n,n,r]*np.dot(self.output_params[:,r],Xe[n,:])
        
        return y_pred
    
    def plotInputMembFuncs(self, x_lims, x_dim):
        x_vals = np.linspace(x_lims[0],x_lims[1],1000)
        mf_vals = np.zeros((self.input_params.shape[2],x_vals.size))
        
        for r in range(0,self.input_params.shape[2]):
            mf_params = self.input_params[x_dim,:,r]
            mf_vals[r,:] = np.apply_along_axis(func1d=gauss_mf,axis=0,arr=x_vals,c=mf_params[0],sigma=mf_params[1])            
            plt.plot(x_vals, mf_vals[r,:])
            
        plt.show()
            
        return mf_vals
    

    # def plotOutputFuncs(self, x_lims, x_dim, y_dim):
    #     x_vals = np.linspace(x_lims[0],x_lims[1],1000)
    #     mf_vals = np.zeros((self.input_params.shape[2],x_vals.size))
        
    #     for r in range(0,self.input_params.shape[2]):
    #         mf_params = self.input_params[x_dim,:,r]
    #         mf_vals[r,:] = np.apply_along_axis(func1d=gauss_mf,axis=0,arr=x_vals,c=mf_params[0],sigma=mf_params[1])            
            

    #     return 0
        
        