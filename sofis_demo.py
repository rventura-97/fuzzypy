# %%
import numpy as np
from sofis import sofis
from sklearn.datasets import load_iris
from read_data import load_occupancy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% Iris dataset
iris_data = load_iris()
X = np.array(iris_data.data)
y = np.array(iris_data.target)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_off, X_on, y_off, y_on = train_test_split(X_train,y_train,test_size=0.5,random_state=42)

# %% Occupancy dataset
occup_data = load_occupancy()
X_train = occup_data['X_train']
X_test = occup_data['X_test']
y_train = occup_data['y_train']
y_test = occup_data['y_test']

# %% Offline training
mod = sofis(L=1,dist='euclidean')
mod.fit_offline(X_train,y_train)

# %%


# %% Test model
y_pred = mod.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)

