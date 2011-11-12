import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Memory

from utils import format_data

# Using joblib allows to cache some of the results, in order to gain time on
# computation

mem = Memory(cachedir='.')


################################################################################
# Load the data
X, y = format_data()

################################################################################
# Split data into training set and testing set
print "Splitting the data"
test, train = iter(StratifiedKFold(y, k=15)).next()
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

################################################################################
# Train the SVM classification model
print "Training the classification model"
param_grid = {
  'C': [1, 5, 10, 50, 100, 500]
  }

clf = GridSearchCV(LinearSVC(), param_grid,
                   fit_params={'class_weight': 'auto'})
clf = clf.fit(X_train, y_train)
print "Best estimator found by grid search:"
print clf.best_estimator
clf = clf.best_estimator

w =  clf.coef_[0]
w = w.reshape((24, 24))

################################################################################
# Let's test the classifier
print "Testing the classifier"
y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred)
print confusion_matrix(y_test, y_pred)



