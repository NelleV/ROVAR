import os

import numpy as np
import scipy as sp
import itertools

from pylab import imread, imsave, imshow, mean

from sklearn.metrics.pairwise import euclidean_distances

from PIL import Image

from harris import harris

import sift
from vlfeat import vl_sift

import utils
import ransac
import ransac_model

THRESHOLD = 0.65

ITERATIONS = 50

def detect_harris_detector(image):
    """
    Detects harris points

    params
    -------
        image: numpy array

    returns
    -------
        coords: list of coords

    """
    # The Harris implementation we have only works on n*n matrix. Let's fake
    # it
    length = image.shape[1] - image.shape[0]

    # First compute Harris on the first half of the image
    # im1 = image[:, 0:length]
    im1 = image
    harrisim1 = harris.compute_harris_response(im1)
    points1 = harris.get_harris_points(harrisim1,
                                       min_distance=5,
                                       threshold=0.1)

    # Then the second half of the image
#    im2 = image[:, 152:]
#    harrisim2 = harris.compute_harris_response(im2)
#    points2 = harris.get_harris_points(harrisim2, min_distance=10, threshold=0.1)
#    # the second set of points have been translated of image.shape[0] - length
#    # and image.shape[1] - length
#    harris_points = set((x[0],
#                         x[1] + 152) for x in points2)


    # harris_points = harris_points.union(set(x for x in points1))
    return set(points1)
    # harris_points = harris_points.union(set(length + x for x in points2))
    # And merge the results


def extract_coord(sift_array):
    coords = set()
    for element in sift_array:
        coords.add((int(element[0]), int(element[1])))
    return coords


def euclidean_distances(d1, d2):
    distances = np.zeros((d1.shape[0], d2.shape[0]), dtype=np.int16)
    for i, x in enumerate(d1):
        for j, y in enumerate(d2):
            d = np.zeros(y.shape, dtype=np.int16)
            d = (x - y)**2
            distances[i, j] = d.sum(axis=0)
    return distances


def match_descriptors(d1, d2, f1, f2):
    """
    Match descriptors

    returns
    -------
        A n*4 array of coordinates
    """
    distances = euclidean_distances(d1, d2)
    # the nearest neighbour is the one for which the euclidean distance is the
    # smallest
    N1 = np.array([[x, y] for x, y in enumerate(distances.argmin(axis=1))])
    distances_N1 = distances.min(axis=1)
    for X in N1:
        distances[X[0], X[1]] = 25000
    distances_N2 = distances.min(axis=1)

    # TODO thresholding
    eps = np.zeros(distances_N1.shape, dtype=np.float64)
    eps += distances_N1
    eps /= distances_N2

    eps = eps < THRESHOLD

    matches = []
    matches_d = []
    for i, element in enumerate(eps):
        if element:
            matches.append((f1[N1[i][0], 0],
                            f1[N1[i][0], 1],
                            f2[N1[i][1], 0],
                            f2[N1[i][1], 1]))
            matches_d.append(N1[i])

    return matches, matches_d


def show_descriptors(im, l):
    # im and l are two matrix. im is a greyscale image, and l the list of
    # descriptors and their coord.
    image = im

    for element in l:
        draw_point(image, *element)

    return image

def show_sift_desc(im, f):
    image = im

    for element in f:
        draw_point(image, *element[0:2])

    return image


def draw_point(image, x, y):
    image[x][y] = 0
    # image[x][y][1] = 0
    for i in range(3):
        for j in range(3):
            try:
                image[x - i][y - j] = 0
                image[x - i][y + j] = 0
                image[x + i][y - j] = 0
                image[x + i][y + j] = 0
            except IndexError:
                pass


def calculate_homography(points):
    """
    Calculate the homography

    params
    -------
        points_1: 2*n matrix, n being 4.
            points on image 1

        points_2: 2*n matric, (here n=4)
            points on image2

    returns
    -------
        H, the homography matrix
    """
    points_1 = points
    a = []
    for X in points:
        x1 = X[0]
        y1 = X[1]
        x2 = X[2]
        y2 = X[3]

        a_x = np.array([-x1, -y1, -1, 0, 0 , 0, x2 * x1, x2 * y1, x2])
        a_y = np.array([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1,
                        y2])
        a.append(a_x)
        a.append(a_y)
    A = np.array(a)
    H = np.linalg.svd(A)[-1][:, -1]
    H.shape = (3, 3)
    return H


def homography(fp, tp):
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i],
                  tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i],
                    tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)

    H = V[8].reshape((3,3))
    return H


def cal_homography(fp, tp):
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1))
    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.dot(C1,fp)

    #--to points--
    m = np.mean(tp[:2], axis=1)
    #C2 = C1.copy() #must use same scaling for both point sets
    maxstd = np.max(np.std(tp[:2], axis=1))
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2,tp)

    #create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i],
                  tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i],
                    tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)

    H = V[8].reshape((3,3))
    #decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H,C1))

    #normalize and return
    return H / H[2][2]


def RANSAC(desc):
    """
    Estimates the homography between the two images
    """
    # Reformat desc
    redesc = np.ones((desc.shape[0], 6))
    redesc[:, :2] = desc[:, :2]
    redesc[:, 3:5] = desc[:, 2:]
    result = ransac.ransac(redesc, ransac_model.ransac_model(), 4, 150, 50, 3)
    return result


def show_matched_desc(image1, image2, matched_desc):
    image = np.zeros((image1.shape[0], image2.shape[1] + 10 + image2.shape[1]))
    image[:, :image1.shape[1]] = image1
    image[:, 10 + image2.shape[1]:] = image2
    placed_desc = matched_desc
    placed_desc[:, -1] = matched_desc[:, -1] + 10 + image2.shape[1]
    from skimage.draw import bresenham
    for el in placed_desc:
        try:
            image[bresenham(el[0], el[1], el[2], el[3])] = 0
        except:
            pass
    return image



def calculate_inertia(element):
    """Calculates inertia"""
    inertia = []
    count = 0
    for i, j in itertools.permutations(element):
        count += 1
        if count % (len(element) - 1) == 0:
            total = 0
            if count != 0:
                inertia.append(total)
        total += inertia.append(euclidean_distances(d1, d2).min(axis=0).sum())
    return inertia.argmin()


if __name__ == "__main__":
    # Sift descriptor doesn't work with color images. Let's stick with grey
    # images
    image1 = mean(imread('keble_a.jpg'), 2)[::-1]
    image2 = mean(imread('keble_a.jpg'), 2)[::-1]
    image1 = image1[:400, :400]
    image2 = image2[:400, 50:450]
    # image2 = mean(imread('keble_b.jpg'), 2)[::-1]
    #FIXME we assume that image1.shape = image2.shape

    # the image is rotated
    coords1 = detect_harris_detector(image1)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = detect_harris_detector(image2)
    key_points2 = utils.create_frames_from_harris_points(coords2)


    # Get sift descriptors
    f1, d1 = vl_sift(np.array(image1, 'f', order='F'),
                     frames=key_points1,
                     orientations=False)
    import pdb; pdb.set_trace()
    f1, d1 = f1.transpose(), d1.transpose()

    nd1 = np.zeros(d1.shape)
    nd1 += d1
    nd1 /= d1.max()

    # Get sift descriptors
    f2, d2 = vl_sift(np.array(image2, 'f', order='F'),
                     frames=key_points2,
                     orientations=False)
    f2, d2 = f2.transpose(), d2.transpose()
    nd2 = np.zeros(d2.shape)
    nd2 += d2
    nd2 /= d2.max()


    matched_desc, matches_d = match_descriptors(nd1, nd2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
    image  = show_matched_desc(image1, image2, matched_desc)
    # imshow(image)


    # Get the sift descriptors of the image using the pgm converted version of
    # the image
    # l1, d1 = get_sift_descriptors('keble_a.pgm')

    # l2, d2 = get_sift_descriptors('keble_b.pgm')
    # l3, d3 = get_sift_descriptors('keble_c.pgm')


