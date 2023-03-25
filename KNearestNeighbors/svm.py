from sklearn import datasets
import numpy as np
from sklearn import metrics
from eda import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, maxabs_scale
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

X = df[columns].values
y = df['target'].values

print('Class labels:', np.unique(y))
# Splitting data into 70% training and 30% test data:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
# Standardizing the features:


#Tried different scalers but used standard scaler since it gave the best metric results
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1,
                         C =500, gamma = 10,kernel = 'rbf'))
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

#Roc curve
metrics.plot_roc_curve(pipe_svc, X_test, y_test)
plt.show()


print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# svm = SVC(random_state=1,C =1000, gamma = 0.01,kernel = 'rbf')

#Using sequential back search to find the number of features which give the best results

# sbs = SBS(svm, k_features=1)
# sbs.fit(X_train, y_train)

# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.tight_layout()
# plt.show()#


#Using stratified validation to find best parameters

# param_grid = [{'svc__C': [100,500,1000], 
#                'svc__kernel': ['rbf'],
#                 'svc__gamma': [10,0.1,0.001]}]

# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=2)
# scores = cross_val_score(gs, X_train, y_train, 
#                          scoring='accuracy', cv=5)

# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))

# print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
# print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
# print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# gs = gs.fit(X_train, y_train)
# print("best score: ", gs.best_score_)
# print("best parameter: ",gs.best_params_)
# print("Best estimator:", gs.best_estimator_)
# clf = gs.best_estimator_
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))
# gs = GridSearchCV(estimator=pipe_svc, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   refit=True,
#                   cv=10,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)
# clf = gs.best_estimator_
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# ## Algorithm selection with nested cross-validation
