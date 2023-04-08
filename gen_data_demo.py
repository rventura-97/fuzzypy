# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gen_data import siso_reg_1, siso_reg_2, siso_reg_3
from sklearn.model_selection import train_test_split

# %% SISO Regression 1
x, y = siso_reg_1()
plt.plot(x,y)
data = pd.DataFrame(data=np.column_stack([x,y]),columns=['x','y'])
data.to_csv('DATA/siso_reg_1.csv')

# %% SISO Regression 2
x, y = siso_reg_2()
plt.plot(x,y)
data = pd.DataFrame(data=np.column_stack([x,y]),columns=['x','y'])
data.to_csv('DATA/siso_reg_2.csv')


# %% SISO Regression 3
x, y = siso_reg_3()
plt.plot(x,y)
data = pd.DataFrame(data=np.column_stack([x,y]),columns=['x','y'])
data.to_csv('DATA/siso_reg_3.csv')