import sys
sys.path.append(
    "/Users/phoenix_20/Desktop/My Folder /Coding Stuff/ML /PyMLalgo/PyMLAlgos"
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],y,color="b",marker="o",s=30)
# plt.show()

# print(X_train.shape)
# print(y_train.shape)

from Algorithm.linear_regression import LinearRegression

regressor = LinearRegression(lr=1)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


mse_value = mse(y_test, predicted)
print(mse_value)
