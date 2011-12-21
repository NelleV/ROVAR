import numpy as np

from matplotlib.pyplot import imread, matshow
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Memory

from utils import format_data, generate_bounding_boxes, normalise
from utils import show_positive_boxes, create_heat_map, merge_bounding_boxes
from utils import find_centroids, make_new_bounding_boxes

# Using joblib allows to cache some of the results, in order to gain time on
# computation

mem = Memory(cachedir='.')

################################################################################
# Load the training data and fit the classifier
print "fitting the classifier"
def classifier():
    X, y = format_data()

    clf = LinearSVC(C=0.005)
    clf = mem.cache(clf.fit)(X, y)
    return clf

def classifier_rbf():
    X, y = format_data()

    clf = SVC(C=10, gamma=0.002)
    clf = mem.cache(clf.fit)(X, y)
    return clf


################################################################################
# Generate all possibles patches, and predict the classifier on them
print "predicting on the images"

def predict_on_image(file_path, thres=0.5):
    clf = classifier()
    im1 = imread(file_path)[::-1].mean(axis=2)
    positions1, boxes1 = generate_bounding_boxes(im1, pix=1)

    #labels1 = clf.predict(boxes1)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    scores1 = np.dot(boxes1, w) + b
    #image1 = show_positive_boxes(im1, labels1, positions1)
    hmap = create_heat_map(im1, scores1, positions1)

    centroids, centroids_scores = find_centroids(positions1, scores1,
                                                 min_dist=35)
    sc = centroids_scores > thres
    im = show_positive_boxes(im1, sc, centroids - 12)

    return hmap, im, centroids_scores


hmap1, im1, scores1 = mem.cache(predict_on_image)('../data/img1.jpg',
                                                  thres=0)
hmap2, im2, scores2 = mem.cache(predict_on_image)('../data/img2.jpg',
                                                  thres=0)
hmap3, im3, scores3 = mem.cache(predict_on_image)('../data/img3.jpg',
                                                  thres=0)
hmap4, im4, scores4 = mem.cache(predict_on_image)('../data/img4.jpg',
                                                  thres=0)

fig = plt.figure()
ax = fig.add_subplot(4, 2, 1)
ax.matshow(im1, cmap=cm.gray)
ax = fig.add_subplot(4, 2, 2)
ax.matshow(hmap1)

ax = fig.add_subplot(4, 2, 3)
ax.matshow(im2, cmap=cm.gray)
ax = fig.add_subplot(4, 2, 4)
ax.matshow(hmap2)

ax = fig.add_subplot(4, 2, 5)
ax.matshow(im3, cmap=cm.gray)
ax = fig.add_subplot(4, 2, 6)
ax.matshow(hmap3)

ax = fig.add_subplot(4, 2, 7)
ax.matshow(im4, cmap=cm.gray)
ax = fig.add_subplot(4, 2, 8)
ax.matshow(hmap4)



#im2 = imread('../data/img2.jpg')[::-1].mean(axis=2)
#positions2, boxes2 = mem.cache(generate_bounding_boxes)(im2, pix=4)
#
#labels2 = mem.cache(clf.predict)(boxes2)
#image2 = mem.cache(show_positive_boxes)(im2, labels2, positions2)
#
#
#im3 = imread('../data/img3.jpg')[::-1].mean(axis=2)
#positions3, boxes3 = mem.cache(generate_bounding_boxes)(im3, pix=4)
#
#labels3 = mem.cache(clf.predict)(boxes3)
#image3 = mem.cache(show_positive_boxes)(im3, labels3, positions3)
#
