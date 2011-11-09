import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Memory

import utils

# Using joblib allows to cache some of the results, in order to gain time on
# computation

mem = Memory(cachedir='.')


################################################################################
# Load the data
def format_data():
    pos, neg = utils.load_data()

    pos_norm = utils.normalise(pos)
    neg_norm = utils.normalise(neg)

    svm_pos = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    svm_neg = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

    # Let's format the data in a nicer way
    X = np.concatenate((svm_pos, svm_neg), axis=1).T
    y = np.concatenate((np.zeros((svm_pos.shape[1])),
                        np.ones((svm_neg.shape[1]))),
                    axis=1).T
    return X, y

X, y = mem.cache(format_data)()

################################################################################
# Split data into training set and testing set
print "Splitting the data"
train, test = iter(StratifiedKFold(y, k=4)).next()
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

################################################################################
# Train the SVM classification model
print "Training the classification model"
param_grid = {
 'C': [1, 5, 10, 50, 100],
  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
  }

clf = mem.cache(GridSearchCV)(SVC(kernel='rbf'), param_grid,
                   fit_params={'class_weight': 'auto'})
clf = mem.cache(clf.fit)(X_train, y_train)
print "Best estimator found by grid search:"
print clf.best_estimator

################################################################################
# Let's test the classifier

y_pred = mem.cache(clf.predict)(X_test)
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))



