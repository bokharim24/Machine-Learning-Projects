import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from eda import *
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler

class LinearRegressionGD(object):

    def __init__(self, eta=0.15, n_iter=20):
        self._eta = eta
        self._n_iter = n_iter

    def fit(self, X, y):
        self._w = np.zeros(1 + X.shape[1])
        self._cost = []

        for i in range(0,self._n_iter):
            output = self.net_input(X)
            error = output - y
            self._w[1:] -= (self._eta * X.T.dot(error))/len(y)
            self._w[0] -= (self._eta * error.sum())/len(y)
            cost = 1/2*(error**2).sum()/len(y)
        
            self._cost.append(cost)

            print("Epoch count", i , ":", "Loss Function", self._cost[i])

#Rooms per person, change total rooms

    def net_input(self, X):
        return np.dot(X,self._w[1:]) + self._w[0]
    
    def predict(self, X):
        return self.net_input(X)

#Standardizing the data
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X_train)
y_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
X_test_std_whole = sc_x.transform(X_test)

X_test_five = X_test.head()
X_test_std_five = sc_x.transform(X_test_five)

lr = LinearRegressionGD()
lr.fit(X_std, y_std)


#Predicting for whole dataset
y_test_pred = lr.predict(X_test_std_whole)
y_test_pred = sc_y.inverse_transform(y_test_pred.reshape(1,-1))

#Predicting for just 5 values 
y_test_pred_five = lr.predict(X_test_std_five)
y_test_pred_five = sc_y.inverse_transform(y_test_pred_five.reshape(1,-1))


print("y actual:\n",y_test.head().to_string(index=False))
y_x = ""
for i in range(5):
    y_x += str(y_test_pred_five[0][i])
    y_x += "\n"

print("y_predicted: \n",y_x)
# Run predict for all but scale down 5 values

"""
plt.scatter(y_train_pred, y_train_pred - y_train,
             c='steelblue', marker='o', edgecolor='white',
             label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
             c='limegreen', marker='s', edgecolor='white',
             label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

"""

plt.plot(range(1, lr._n_iter+1), lr._cost)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()



#print('MSE train: %.3f, test: %.3f' % (
#        mean_squared_error(y_train, y_train_pred),
#        mean_squared_error(y_test, y_test_pred)))

#print('Slope: %.3f' % lr._w[1])
#print('Intercept: %.3f' % lr._w[0])
