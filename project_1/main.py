import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd

train_df = pd.read_csv("task1a_lm1d3za/train.csv", header=1)
train_data = train_df.to_numpy()

y = train_data[:, 1]
X = train_data[:, 2:]

reg_lambda_list = [0.01, 0.1, 1, 10, 100]

kf = KFold(n_splits=10, shuffle=True, random_state=53)
w = []

rmse = []

for reg_lambda in reg_lambda_list:
    mse = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w = np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + (np.eye(X_train.shape[1]) * reg_lambda)), np.dot(X_train.T, y_train))

        y_pred = np.dot(X_test, w)
        mse += mean_squared_error(y_test, y_pred)
    mse /= kf.get_n_splits()
    rmse.append(mse**0.5)

# print(rmse)
rmse_df = pd.DataFrame(rmse)
rmse_df.to_csv("output.csv", header=False, index=False)
