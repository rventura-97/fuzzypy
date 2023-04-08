# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tsk import tsk

# %% SISO Regression 1 

data = pd.read_csv('DATA/siso_reg_2.csv')
X = np.array(data['x'])
y = np.array(data['y'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

my_tsk = tsk(learn_mode='reg',\
             mf_func_type='gauss',\
             conseq_order='linear',\
             partition_method='fcm',\
             partition_method_params={'Nc':3,'m':2.0})
    
my_tsk.fit(X_train,y_train)

plot_vals = my_tsk.plotInputMembFuncs(x_lims=[0,1], x_dim=0)

#y_pred = my_tsk.predict(X_test)
y_pred = my_tsk.predict(X)
plt.plot(X,y)
plt.plot(X,y_pred)
plt.show()


# %%
