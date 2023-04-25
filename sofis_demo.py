# %%
import numpy as np
from sofis import sofis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# %%
iris_data = load_iris()
X = np.array(iris_data.data[:, :2])
y = np.array(iris_data.target)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %%
mod = sofis()
mod.fit(X,y)