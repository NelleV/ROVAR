import numpy as np

from matplotlib import pyplot

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
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
X_train, X_test = X[0:3000], X[3000:6000]
y_train, y_test = y[0:3000], y[3000:6000]

################################################################################
# Train the SVM classification model
print "Training the classification model"
cs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
csx = range(len(cs))
precisions = []
recalls = []
for c in cs:
    print c
    clf = LinearSVC(C=float(c))
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

    # print classification_report(y_test, y_pred)

fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.plot(csx, precisions)
ax.plot(csx, recalls)

leg = ax.legend(('Precisions', 'Recalls'), shadow=True)

for t in leg.get_texts():
    t.set_fontsize('small')    # the legend text fontsize

# matplotlib.lines.Line2D instances
for l in leg.get_lines():
    l.set_linewidth(1.5)  # the legend line width


