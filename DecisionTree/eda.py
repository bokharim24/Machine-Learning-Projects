import pandas as pd
import numpy as np
from statistics import mean
import math
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('winequality-red.csv')
#threshold of keeping at .1 negative or positive Pearson's R correlation
#noticed fixed and volatile acidity have posatively correlated scatterplot linearities -- removing the
#one with less strong Pearson's R value -- fixed acidity
#Removing free sulfur dioxide as well since total SF already consists of it.
# Not considering density since it has a very strong negative relationship with alcohol, so using alcohol only
columns = ['volatile acidity','citric acid','chlorides','total sulfur dioxide','sulphates','alcohol','quality']
#scatterplotmatrix(df[columns].values, figsize=(22, 16),
#                  names=columns, alpha=0.5)
#plt.tight_layout()
#plt.show()
cm = np.corrcoef(df[columns].values.T)
hm = heatmap(cm, row_names=columns, column_names=columns)
plt.show()

#Shows distribution of quality of wine, very high class imbalance
df['quality'].hist()
plt.ylabel("Count")
plt.xlabel("Value")
plt.show()

#Boxplot to show the statistics for each column in our dataframe
plt.figure(figsize = (10,15))
for i, col in enumerate(list(df.columns.values)):
    plt.subplot(4,4,i+1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()
plt.show()

print(df.describe())

# #Seperating bad_quality, avg_quality, above avg_quality and good_quality, 3-4 bad, 5 avg, 6 above avg, 7-8 good
df['quality'].replace([3,4], [0,0], inplace= True)
df['quality'].replace(5, 1, inplace= True)
df['quality'].replace(6, 2, inplace= True)
df['quality'].replace([7,8], [3,3], inplace= True)



print(df['quality'].value_counts())


print('Class labels', np.unique(df['quality']))
#Kept sulphate even though it slightly decreased the accuracy, just to keep the model more generalizable
cols = ['volatile acidity','citric acid','chlorides','total sulfur dioxide','sulphates','alcohol','free sulfur dioxide','density','fixed acidity','residual sugar','pH']
cols = ['volatile acidity','citric acid','total sulfur dioxide','sulphates','alcohol']
#df[cols] = df[cols].astype(np.float16)
# X, y = df[cols].values, df['quality'].values
# X_train, X_test, y_train, y_test =    train_test_split(X, y, 
#                      test_size=0.3, 
#                      random_state=0, 
#                      stratify=y)

# # # Bringing features onto the same scale
# mms = MinMaxScaler()
# X_train_norm = mms.fit_transform(X_train)
# X_test_norm = mms.transform(X_test)
# stdsc = StandardScaler()
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.transform(X_test)

# class SBS():
#     def __init__(self, estimator, k_features, scoring=accuracy_score,
#                  test_size=0.25, random_state=1):
#         self.scoring = scoring
#         self.estimator = clone(estimator)
#         self.k_features = k_features
#         self.test_size = test_size
#         self.random_state = random_state
#     def fit(self, X, y):
        
#         X_train, X_test, y_train, y_test =             train_test_split(X, y, 
# test_size=self.test_size,
#                              random_state=self.random_state)
#         dim = X_train.shape[1]
#         self.indices_ = tuple(range(dim))
#         self.subsets_ = [self.indices_]
#         score = self._calc_score(X_train, y_train, 
#                                  X_test, y_test, self.indices_)
#         self.scores_ = [score]
#         while dim > self.k_features:
#             scores = []
#             subsets = []
#             for p in combinations(self.indices_, r=dim - 1):
#                 score = self._calc_score(X_train, y_train, 
#                                          X_test, y_test, p)
#                 scores.append(score)
#                 subsets.append(p)
#             best = np.argmax(scores)
#             self.indices_ = subsets[best]
#             self.subsets_.append(self.indices_)
#             dim -= 1
#             self.scores_.append(scores[best])
#         self.k_score_ = self.scores_[-1]
#         return self
#     def transform(self, X):
#         return X[:, self.indices_]
#     def _calc_score(self, X_train, y_train, X_test, y_test, indices):
#         self.estimator.fit(X_train[:, indices], y_train)
#         y_pred = self.estimator.predict(X_test[:, indices])
#         score = self.scoring(y_test, y_pred)
#         return score
# logreg = LogisticRegression()
# # selecting features
# sbs = SBS(logreg, k_features=1)
# sbs.fit(X_train_std, y_train)
# # plotting performance of feature subsets
# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.5, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.tight_layout()
# # plt.savefig('images/04_08.png', dpi=300)
# plt.show()
# k3 = list(sbs.subsets_[0])
# print(df[cols].values[k3])
# logreg.fit(X_train_std, y_train)
# print('Training accuracy:', logreg.score(X_train_std, y_train))
# print('Test accuracy:', logreg.score(X_test_std, y_test))
# logreg.fit(X_train_std[:, k3], y_train)
# print('Training accuracy:', logreg.score(X_train_std[:, k3], y_train))
# print('Test accuracy:', logreg.score(X_test_std[:, k3], y_test))


