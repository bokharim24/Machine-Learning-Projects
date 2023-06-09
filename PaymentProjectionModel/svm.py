from eda import EDA
from sklearn import datasets
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, precision_score, recall_score, RocCurveDisplay
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

# Get data from eda
eda = EDA()
X_train, X_test, y_train, y_test = eda.train_test_split()
# print(np.count_nonzero(y_train)/len(y_train))
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# Create pipeline
# svcm = SVC(kernel="linear") # When not using ROC AUC
svcm = SVC(kernel="rbf",C=125, gamma='scale', decision_function_shape='ovo', probability=True)
pipeline = make_pipeline(MaxAbsScaler(), svcm)
 

# Fit data
pipeline.fit(X_train, y_train.ravel()) 

# Predict
y_pred = pipeline.predict(X_test)
# Probability of predicting true label
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Print confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Print evaluation metrics
print("\n\nSVM Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(
    y_test, y_pred, average="weighted"))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Recall (weighted):", recall_score(y_test, y_pred, average="weighted"))

# Rand search for optimal hyperparameters
# Other Potential hyperparameters: coef0: float, class_weight: dict or ‘balanced’, tol: float(default=1e-3), probability: bool, shrinking: bool
param_grids = {"C": [0.001, 0.1, 1, 100],
              "kernel": ["linear", "poly", "rbf" ]}

gs = GridSearchCV(estimator=SVC(random_state=0),param_grid = param_grids)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
gs = gs.fit(X_train, y_train)
print("best score: ", gs.best_score_)
print("best parameter: ",gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# # use pipeline for random/grid search
# svcm_gs = SVC(kernel ='rbf', probability=True)
# pipeline_gs = make_pipeline(MaxAbsScaler(), svcm_gs)

# param_grid = {"svc__C": [50, 75, 100, 125],
#             #   "svc__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
#             #   "svc__degree": [1, 2, 3, 4, 5],  # Only for 'poly' kernel
#               "svc__gamma": ["scale", "auto"],
#               "svc__decision_function_shape": ["ovo", "ovr"]}

# grid = GridSearchCV(estimator=pipeline_gs, param_grid=param_grid, verbose=3, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train)
# grid_predictions = grid.predict(X_test)

# print(classification_report(y_test, grid_predictions))
# print(grid.best_params_)
