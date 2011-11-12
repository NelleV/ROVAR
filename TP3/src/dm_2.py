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
im1 = imread('../data/img1.jpg')[::-1].mean(axis=2)
boxes1 = mem.cache(generate_bounding_boxes)(im1)

labels1 = clf.predict(boxes1)

im2 = imread('../data/img2.jpg')[::-1].mean(axis=2)
boxes2 = mem.cache(generate_bounding_boxes)(im2)

labels2 = clf.predict(boxes2)

im3 = imread('../data/img3.jpg')[::-1].mean(axis=2)
boxes3 = mem.cache(generate_bounding_boxes)(im3)

labels3 = clf.predict(boxes3)


