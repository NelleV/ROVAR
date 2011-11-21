================================================================================
Assignment 3: Simple Face Detector
================================================================================


1. Choosing C
================================================================================

We train a linear SVM on a small set of images, and compute the precision,
recall and f1 score.


.. figure:: images/01_recalls_precisions.png
  :scale: 50 %

  Recall and precision curves for 1000 training data


For ``C=0.005``


.. +=============+=============+========+===========+==========+

+-------------+-------------+--------+-----------+----------+
|             | precision   | recall | f1-score  | support  |
+-------------+-------------+--------+-----------+----------+
|          0  |      0.90   |   0.95 |     0.93  |    2426  |
+-------------+-------------+--------+-----------+----------+
|          1  |     0.73    |  0.56  |    0.64   |    574   |
+-------------+-------------+--------+-----------+----------+
| avg / total |      0.87   |   0.88 |     0.87  |    3000  |
+-------------+-------------+--------+-----------+----------+


For ``C=0.01``

+-------------+-------------+--------+-----------+----------+
|             | precision   | recall | f1-score  | support  |
+=============+=============+========+===========+==========+
|          0  |     0.90    |  0.95  |    0.92   |   2426   |
+-------------+-------------+--------+-----------+----------+
|          1  |     0.72    |  0.56  |    0.63   |    574   |
+-------------+-------------+--------+-----------+----------+
| avg / total |      0.87   |   0.87 |     0.87  |    3000  |
+-------------+-------------+--------+-----------+----------+

I choose to set ``C=0.005`` for the rest of the project, as the results are
slightly better with this configuration.


2. The Hyperplane
================================================================================

Once the LinearSVM is fitted, we can visualise the obtained results by
reshaping the coefficient ``w`` to the format of the patches provided to the
classifier.

.. figure:: images/02_hyperplan.png
  :scale: 10 %

  Linear hyperlane

We can recognise a face.


3. High-confident scanning-window detections on test images in Step 3
================================================================================

.. figure:: images/03_image1.png
  :scale: 65 %

  All windows recognised by the code



4. Merging bounding boxes
================================================================================

.. figure:: images/04_image1.png
  :scale: 65 %

As we can observe, most of the faces are detected correctly. On the other
hand, the  classifier also detects a lot of false matches. If we observe the
patches detected as face normalise, we can see that it looks a lot like a
face.

We can improve the results of the classification by using a series of
classifier or retrain the linearSVM with false positive matches. Here, we will
refit the matching patches on an RBF kernel SVM, in order to improve the
results.


5. (Optional) improved detection results using non-linear classifier.
================================================================================


