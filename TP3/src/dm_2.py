import numpy as np

from matplotlib.pyplot import imread

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Memory

from utils import format_data, generate_bounding_boxes, normalise


# Using joblib allows to cache some of the results, in order to gain time on
# computation

mem = Memory(cachedir='.')

################################################################################
# Load the training data and fit the classifier
print "fitting the classifier"
X, y = format_data()

clf = LinearSVC(C=50)
clf = mem.cache(clf.fit)(X, y)

################################################################################
# Generate all possibles patches, and predict the classifier on them
print "predicting on the images"
im = imread('../data/img2.jpg')[::-1].mean(axis=2)
boxes = mem.cache(generate_bounding_boxes)(im)

labels = clf.predict(boxes)

