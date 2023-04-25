import numpy as np

class sofis:
    
    def __init__(self):
        self.class_models = []
        self.feat_names = []
        
    def fit(self,X,y):
        # Initialize class sub-models
        self.__init_class_models(y)
        
        # Offline training of each class sub-model
        
        return 0
    
    def __init_class_models(self,y):
        self.class_models = [class_model()]*np.unique(y).size
        
        
class class_model:
    
    def __init__(self):
        self.label = ''
        
        self.P = []
        
    def fit(X):
        return 0