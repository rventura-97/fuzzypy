# %%
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, make_blobs
from sklearn.preprocessing import MinMaxScaler
from fcm import fcm, fcm_optimizer
scaler = MinMaxScaler()
#X = load_iris().data
#X = scaler.fit_transform(load_diabetes().data)
X = make_blobs(n_samples=500,n_features=2,centers=np.array([[1,1],[1,-1],[-1,1],[-1,-1]]))[0]

# %% FCM Clustering
my_fcm = fcm(Nc=4, m=2.0, max_iters=200, min_improv=0.001, rnd_seed=471997)
my_fcm.fit(X);

# %% FCM Optimizer
my_fcm_opt = fcm_optimizer(list(range(2,41)), [1.1], 5, 'xie-beni', rnd_seed=2023)
best_fcm = my_fcm_opt.fit(X)

