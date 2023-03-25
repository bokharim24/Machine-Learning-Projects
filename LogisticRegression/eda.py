import imp
from tkinter.tix import InputOnly
import pandas as pd
import numpy as np
from statistics import mean
import math
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('covid.csv')

#Removing values with 97 and 99 for the ICU column
df.drop(df[df['icu'] == 97].index, inplace = True)
df.drop(df[df['icu'] == 99].index, inplace = True)

df['entry_date']= pd.to_datetime(df['entry_date'],dayfirst=True)
df['date_symptoms']= pd.to_datetime(df['date_symptoms'],dayfirst=True)

#column to find difference between entry date and the date symptoms were discovered
df['difference'] = (df['entry_date'] - df['date_symptoms']).dt.days

#Changed a no value to be 0 instead of 2,99,98,97
changeCol = ['sex','pregnancy','covid_res','intubed','pneumonia','diabetes','copd','asthma','inmsupr','hypertension','other_disease',
'cardiovascular','obesity','renal_chronic','tobacco','icu','contact_other_covid']
for i in changeCol:
    df[i].replace(to_replace=2, value = 0,inplace=True)

changeCol = ['pregnancy','intubed','pneumonia','diabetes','copd','asthma','inmsupr','hypertension','other_disease',
'cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid']
for i in changeCol:
    df[i].replace(to_replace=99, value = 0,inplace=True)
    df[i].replace(to_replace=98, value = 0,inplace=True)
    df[i].replace(to_replace=97, value = 0,inplace=True)

df['covid_res'].replace(to_replace=3,value = 0,inplace=True)


#Removing IDS, they have no impact on the model.
df.drop(columns=['id'], inplace=True)

#If not hospitalized, then can't be in icu so does not make sense
df.drop(columns=['patient_type'], inplace=True)

df.drop(columns=['date_died'], inplace=True)

#Not important in determining outcome, redundant, using another feature which is difference
df.drop(columns = ["entry_date"],inplace=True)
df.drop(columns = ["date_symptoms"],inplace = True)
#df.drop(df[df['covid_res'] == 3].index, inplace = True)


#print(df['intubed'].value_counts(normalize=True)[97]  + df['intubed'].value_counts(normalize=True)[99])

#print(df.dtypes)

#cols = ['sex', 'patient_type', 'covid_res','age','diabetes','copd','asthma','inmsupr','hypertension','other_disease',
#'cardiovascular','obesity','renal_chronic','tobacco','difference','icu']

#cols = ['contact_other_covid','intubed','covid_res','pneumonia','difference','age']

#print(df.describe())


#Scalers to test out scaling

def Min_Max(data,col):
    nparr = df[col].to_numpy()
    newVals = []
    max_val = df[col].max()
    min_val = df[col].min()
    for n in nparr:
        val = (n - min_val) / (max_val - min_val)
        newVals.append(val)
    df[col] = newVals

def Standard_Scaler(data,col):
    nparr = df[col].to_numpy()
    newVals = []
    avg = np.mean(df[col])
    stdev = np.std(df[col])
    for n in nparr:
        val = (n - avg) / stdev 
        newVals.append(val)
    df[col] = newVals

def Max_ABS(data,col):
    nparr = df[col].to_numpy()
    newVals = []
    max_val = df[col].max()
    for n in nparr:
        val = n / max_val 
        newVals.append(val)
    df[col] = newVals

def Robust_Scaler(data,col):
    nparr = df[col].to_numpy()
    newVals = []
    median = np.median(df[col])
    upperQuart = np.quantile(df[col],0.75)
    lowerQuart = np.quantile(df[col],0.25)
    for n in nparr:
        val = (n - median) / (upperQuart - lowerQuart) 
        newVals.append(val)
    df[col] = newVals


#Min_Max(df,'age')
#Min_Max(df,'difference')
#Standard_Scaler(df,'age')
#Standard_Scaler(df,'difference')
#Robust_Scaler(df,'age')
#Robust_Scaler(df,'difference')
#Max_ABS(df,'age')
#Max_ABS(df,'difference')

#print(df)

cols = ['intubed','pneumonia','covid_res','difference','age']
print(df.describe())



#scatterplotmatrix(df[cols].values, figsize=(10, 8), 
#                  names=cols, alpha=0.5)
#plt.tight_layout()
#plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

plt.show()






