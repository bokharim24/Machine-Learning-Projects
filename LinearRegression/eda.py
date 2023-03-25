import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('housing.csv')

dummies = pd.get_dummies(df.ocean_proximity)
df = pd.concat([df,dummies], axis = 'columns')


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

dfs = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]




df = df.drop(columns=['<1H OCEAN','households','population','latitude','longitude','ISLAND','ocean_proximity','total_bedrooms',])

df = df.fillna(df.mean())
#print(df)


#print(df.describe())



#print(df.describe())

cols = ['housing_median_age','total_rooms','median_income','NEAR OCEAN', 'NEAR BAY', 'INLAND']

X = df[cols].values
y = df['median_house_value'].values


#Splitting dataset into training and testing
shuffle_df = df.sample(frac=1)
train_size = int(0.8 * len(df))
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

X_train = train_set[cols]
y_train = train_set['median_house_value']

X_test = test_set[cols]
y_test = test_set['median_house_value']

#print(y_test._get_value(1))

#print(y_test.head())
"""
scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()

plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

plt.show()
"""



