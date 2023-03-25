from sklearn import datasets
import numpy as np
from eda import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

X = df[cols].values
y = df['quality'].values

print('Class labels:', np.unique(y))
# Splitting data into 70% training and 30% test data:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
# Standardizing the features:
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# # Decision tree learning
# ## Maximizing information gain - getting the most bang for the buck
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])

# ## Building a decision tree
#Best features after running gridsearch
tree_model = DecisionTreeClassifier(criterion='entropy', 
                                    max_depth=12,
                                    max_features = 4,
                                    min_samples_split = 2,
                                    min_weight_fraction_leaf = 0,
                                    splitter = "best", 
                                    #max_leaf_nodes = 20,
                                    random_state=1)
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
y_pred = tree_model.predict(X_test)


fig, ax = plt.subplots(figsize=(12, 12))  # whatever size you want
tree.plot_tree(tree_model.fit(X, y), ax=ax, fontsize =6)
plt.show()

#plt.figure(figsize=(15,15))
#tree.plot_tree(tree_model,fontsize=4)
#plt.show()

# print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
# print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average= "weighted"))
# print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,average= "weighted"))
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,average= "weighted"))

# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(confmat)


# tree.plot_tree(tree_model,fontsize=6)
# plt.show()

#print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[
#     #0.6103579436258808
#     #best parameter:  {'criterion': 'entropy', 'max_depth': 12, 'max_features': 4, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'splitter': 'best'}
#     #depth of 12 gave best score 
#     {'max_depth': [6,8,10,12],
#     'criterion': ["gini", "entropy", "log_loss"],
#     'min_weight_fraction_leaf' : [0,0.025,0.05],
#     'min_samples_split': [2,3,5],
#     'max_features': [1,2,3,4,5,6],
#     'splitter' : ["best","random"],
#     'max_leaf_nodes' : [10,20,30]
#     }],scoring='accuracy', cv=5)

# scores = cross_val_score(gs, X_train_std, y_train, scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# gs = gs.fit(X_train, y_train)
# #print("best score: ", gs.best_score_)
# #print("best parameter: ",gs.best_params_)
# clf = gs.best_estimator_
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))


# param_range=[6,7,8,9,10,11,12]


#Did this to get the graph of max depth against accuracy
# train_scores, test_scores = validation_curve(
#                 estimator=tree_model, 
#                 X=X_train, 
#                 y=y_train, 
#                 param_name='max_depth', 
#                 param_range=[4,5,6,7,8,9,10],
#                 cv=10)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(param_range, train_mean, 
#          color='blue', marker='o', 
#          markersize=5, label='Training accuracy')
# plt.fill_between(param_range, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15,
#                  color='blue')
# plt.plot(param_range, test_mean, 
#          color='green', linestyle='--', 
#          marker='s', markersize=5, 
#          label='Validation accuracy')
# plt.fill_between(param_range, 
#                  test_mean + test_std,
#                  test_mean - test_std, 
#                  alpha=0.15, color='green')
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Max_depth')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1.0])
# plt.tight_layout()
# #plt.savefig('images/06_06.png', dpi=300)
# plt.show()
#graph.write_png('tree.png') 


#Building a random forest
#found these to be the best hyperparameters for the random forest after gridsearch
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=100, 
                                max_depth = 10,
                                max_features = 5,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
y_pred = forest.predict(X_test)


# gs = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=[
#     {"n_estimators" : [50,75,100], 
#     'criterion': ["gini", "entropy", "log_loss"],
#     'max_depth': [7,10,12],
#     'max_features': [3,4,5,6],
#     }],scoring='accuracy', cv=5)

# scores = cross_val_score(gs, X_train_std, y_train, scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# gs = gs.fit(X_train, y_train)
# print("best score: ", gs.best_score_)
# print("best parameter: ",gs.best_params_)
# clf = gs.best_estimator_
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
# print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average= "weighted"))
# print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,average= "weighted"))
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,average= "weighted"))