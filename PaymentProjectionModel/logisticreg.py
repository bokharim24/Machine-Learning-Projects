from sklearn import datasets
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from distutils.version import LooseVersion
from eda import EDA

eda = EDA()
X_train, X_test, y_train, y_test = eda.train_test_split()

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs',penalty='l2')
pipeline = make_pipeline(MaxAbsScaler(), lr)
pipeline.fit(X_train, y_train) 

# Predict
y_pred = pipeline.predict(X_test)
# Probability of predicting true label
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]


# Print confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
# Print evaluation metrics
print("\n\Logistic Regression Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(
    y_test, y_pred, average="weighted"))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Recall (weighted):", recall_score(y_test, y_pred, average="weighted"))

#Grid search
# param_grids = {"penalty": ['l1','l2'],
#               "C": [0.001,0.1,1,100],
#               "solver": ["lbfgs","liblinear"],
#               "multi_class":["auto", "ovr","multinomial"]}


# gs = GridSearchCV(estimator=LogisticRegression(random_state=0),param_grid = param_grids)

# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# gs = gs.fit(X_train, y_train)
# print("best score: ", gs.best_score_)
# print("best parameter: ",gs.best_params_)
# clf = gs.best_estimator_
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))