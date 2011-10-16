import numpy as np
from tp1 import calculate_homography
from tp1 import cal_homography, homography
from tp1 import match_descriptors, show_matched_desc

from pylab import *

def test_homo():
    points = np.array([[1, 1, 4, 4],
                       [-1, 0, 2, 3],
                       [5, 5, 8, 8],
                       [0, 3, 3, 6]])

    fp = np.array([[1, 1, 1],
                   [0, 0, 1],
                   [5, 5, 1],
                   [0, 3, 1]])

    tp = np.array([[4, 4, 1],
                   [3, 3, 1],
                   [8, 8, 1],
                   [3, 6, 1]])

    H1 = calculate_homography(points)
    H2 = cal_homography(fp.transpose(), tp.transpose())
    H3 = homography(fp.transpose(), tp.transpose())
    return H1, H2, H3

def test_matched_points():
    d1 = np.array([[1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]])

    d2 = np.array([[1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4],
                   [150, 150]])

    f1 = np.array([[1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]])

    f2 = np.array([[2, 2],
                   [3, 3],
                   [4, 4],
                   [5, 5],
                   [1, 3]])

    es = match_descriptors(d1, d2, f1, f2)
    return es

if __name__ == "__main__":
    image1 = np.zeros((9, 9))
    image2 = np.zeros((9, 9))
    es = np.array(test_matched_points())
    image =  show_matched_desc(image1, image2, es)
    imshow(image)


