# %% Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from almmo import almmo

# %% Regression (single output)
data = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

model = almmo()
model.fit(X_train, y_train)
