# %%
from sofis import sofis
from read_data import load_occupancy, load_optical_digits
from sklearn.metrics import accuracy_score

# %% Occupancy Dataset (Offline Training)
occup_data = load_occupancy()
X_train = occup_data['X_train']
X_test = occup_data['X_test']
y_train = occup_data['y_train']
y_test = occup_data['y_test']

# Train model
mod = sofis(L=5,dist='euclidean')
mod.fit_offline(X_train,y_train)

# %% Test model
y_pred = mod.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# %% Optical Dataset (Offline Training)
optic_data = load_optical_digits()
X_train = optic_data['X_train']
X_test = optic_data['X_test']
y_train = optic_data['y_train']
y_test = optic_data['y_test']

# Train model
mod = sofis(L=12,dist='euclidean')
mod.fit_offline(X_train,y_train)

# Test model
y_pred = mod.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# %%
mod.fit_online(X_test, y_test)

