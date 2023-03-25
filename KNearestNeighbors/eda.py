import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample

df = pd.read_csv('PulsarStar.csv')
print(df.head())
print(df['target'].value_counts())
print(list(df.columns))


#Removed features which had strong correlation with some other features, set threshold to be 0.4.
columns = ['mean of integrated profile','excess kurtosis of integrated profile', 'skewness of integrated profile', 'mean of DM-SNR curve', 'standard deviation of DM-SNR curve', 'excess kurtosis of DM-SNR curve']
scatterplotmatrix(df[columns].values, figsize=(22, 16),
                  names=columns, alpha=0.5)
plt.tight_layout()
plt.show()
cm = np.corrcoef(df[columns].values.T)
hm = heatmap(cm, row_names=columns, column_names=columns)
plt.show()


# print(df['excess kurtosis of integrated profile'].describe())
# print(df['mean of integrated profile'].describe())
# print(df['skewness of integrated profile'].describe())


#Tried getting rid of outliers in the dataset but ended up losing a lot of the dataset, so decided not to go ahead with it

# q_low = df["excess kurtosis of integrated profile"].quantile(0.01)
# q_hi  = df["excess kurtosis of integrated profile"].quantile(0.99)


# q2_low = df['skewness of integrated profile'].quantile(0.01)
# q2_hi = df['skewness of integrated profile'].quantile(0.99)

# df = df[(df["excess kurtosis of integrated profile"] < q_hi) & (df["excess kurtosis of integrated profile"] > q_low)
# & (df["skewness of integrated profile"] < q_hi) & (df["skewness of integrated profile"] > q_low)
# ]
# print(df['excess kurtosis of integrated profile'].describe())
# print(df['mean of integrated profile'].describe())
# print(df['skewness of integrated profile'].describe())


#Balancing the classes, upsampled the minority class which is one by adding about 3000 values.
df_majority = df[df.target==0]
df_minority = df[df.target==1]
 
target_sample = resample(df_minority,
             replace=True,
             n_samples=len(df_majority)-12000,
             random_state=42) 
 
df = pd.concat([target_sample, df_majority])
print(df.describe())

print(df['target'].value_counts())



#Sequential back search for finding suitable number of features to use.
class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
    
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
        
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score











