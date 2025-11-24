import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load dataset
data = datasets.load_breast_cancer()
X = data.data[:, :2]  # use 2 features for visualization
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# RBF SVM with tuning
params = {'C':[0.1,1,10], 'gamma':[0.01,0.1,1]}
rbf_svm = GridSearchCV(SVC(kernel='rbf'), params, cv=3)
rbf_svm.fit(X_train, y_train)

print("Linear SVM accuracy:", linear_svm.score(X_test, y_test))
print("Best RBF params:", rbf_svm.best_params_)
print("RBF SVM accuracy:", rbf_svm.score(X_test, y_test))
