import os

import numpy as np
from scipy import ndimage
from scipy import linalg
import itertools

from pylab import imread, imsave, imshow, mean

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals.joblib import Memory

from skimage import transform

from PIL import Image

from harris import harris

import sift
from vlfeat import vl_sift

import utils
import ransac
import ransac_model

THRESHOLD = 0.50

ITERATIONS = 50

def detect_harris_detector(image, threshold=0.1, min_distance=5):
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
                                       min_distance=min_distance,
                                       threshold=threshold)

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


#def euclidean_distances(d1, d2):
#    distances = np.zeros((d1.shape[0], d2.shape[0]), dtype=np.int16)
#    for i, x in enumerate(d1):
#        for j, y in enumerate(d2):
#            d = np.zeros(y.shape, dtype=np.int16)
#            d = (x - y)**2
#            distances[i, j] = d.sum(axis=0)
#    return distances
#

def match_descriptors(d1, d2, f1, f2):
    """
    Match descriptors

    returns
    -------
        A n*4 array of coordinates
    """
    distances = euclidean_distances(d1, d2)
    mean_norm = float((d1**2).sum() +
                      (d2**2).sum()) / (d2.shape[0] * d2.shape[1] +
                                        d1.shape[0] * d2.shape[1])
    distances /= mean_norm
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

    eps1 = (distances_N1 < 0.1)
    # import pdb; pdb.set_trace()

    eps = np.logical_or(eps1, eps)

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
    H = linalg.svd(A)[-1].T[:, -1]
    H.shape = (3, 3)
    H /= H[2, 2]
    return H


def homography(fp, tp):
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i],
                  tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i],
                    tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = linalg.svd(A)

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


def nelle_desc(image, frames):
    size_x = 8
    size_4
    for frame in frames:
        desc = image[frame[0] - 4:x + 4, frame[1] - 4:frame[1] + 4]

def stitch(image1, image2, points):
    """
    Stitch image1 to image2, with points points
    """
    # image 2 is base image. ie, we need to translate image1
    image1b = np.zeros((image1.shape[0], image1.shape[1]+500))
    image1b[:, 500:] = image1
    image1 = image1b
    image2b = np.zeros((image2.shape[0], image2.shape[1] + 500))
    image2b[:, 500:] = image2
    image2 = image2b


    H1 = calculate_homography(
                np.array([[51, 23, 340, 38],
                          [62, 337, 359, 340],
                          [395, 286, 691, 286],
                          [367, 128, 655, 116]]))
    points = np.array([[340, 38, 51, 23],
                       [359, 340, 62, 337],
                       [691, 286, 395, 286],
                       [655, 116, 367, 128]])
    points[:, 0] += 500
    points[:,2] += 500
    # We translate the points

    H2 = calculate_homography(points)

    image1H = transform.homography(image1, H2)
    em1 = em.copy()
    em1[:, 500:] = image2[:, 500:]
    return em1


if __name__ == "__main__":
    # Sift descriptor doesn't work with color images. Let's stick with grey
    # images
    mem = Memory(cachedir='.')
    if 1:
        image1 = mean(imread('keble_a.jpg'), 2)[::-1].astype(np.float)
        image2 = mean(imread('keble_b.jpg'), 2)[::-1].astype(np.float)
        image1 = image1
        # image2 = image1 + 10
        image2 = image2
    else:
        image = np.zeros((300, 400))
        image += 30
        np.random.seed(0)
        image += np.random.random(size=image.shape)
        image[125:175, 125:175] = 125
        image[100:150, 150:200] = 200
        #image = ndimage.gaussian_filter(image, 3)
        image1 = image.copy()[50:250, 70:270]
        image2 = image.copy()[50:250, 50:250]


    # image2 = mean(imread('keble_b.jpg'), 2)[::-1]
    #FIXME we assume that image1.shape = image2.shape

    # the image is rotated
    coords1 = mem.cache(detect_harris_detector)(image1, threshold=.999)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = mem.cache(detect_harris_detector)(image2, threshold=.999)
    key_points2 = utils.create_frames_from_harris_points(coords2)

    # Rearrange the keypoints to be close
    if 0:
        import hungarian
        dist = euclidean_distances(key_points1.T, key_points2.T)
        ordering = mem.cache(hungarian.hungarian)(dist)[:, 1]
        key_points2 = key_points2[:, ordering]

    # Get sift descriptors
    f1, d1 = mem.cache(vl_sift)(np.array(image1, 'f', order='F'),
                     frames=key_points1,
                     orientations=False)

    #import pdb; pdb.set_trace()
    f1, d1 = f1.transpose(), d1.transpose()

    # Get sift descriptors
    f2, d2 = mem.cache(vl_sift)(np.array(image2, 'f', order='F'),
                     frames=key_points2,
                     orientations=False)
    f2, d2 = f2.transpose(), d2.transpose()

    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
#    image  = show_matched_desc(image1, image2, matched_desc)
#    imshow(image)

    # image 2 is base image. ie, we need to translate image1
    image1b = np.zeros((image1.shape[0], image1.shape[1]+500))
    image1b[:, 500:] = image1
    image1 = image1b
    image2b = np.zeros((image2.shape[0], image2.shape[1] + 500))
    image2b[:, 500:] = image2
    image2 = image2b


    H1 = calculate_homography(
                np.array([[51, 23, 340, 38],
                          [62, 337, 359, 340],
                          [395, 286, 691, 286],
                          [367, 128, 655, 116]]))
    points = np.array([[340, 38, 51, 23],
                       [359, 340, 62, 337],
                       [691, 286, 395, 286],
                       [655, 116, 367, 128]])
    points[:, 0] += 500
    points[:,2] += 500

    H2 = calculate_homography(points)

    Hr = H2.copy()
    Hr[0, 2] = 0
    Hr[1, 2] = 0
    image1H = transform.homography(image1, H2)
#    image2Hr = np.zeros((image2.shape[0], 1500))
#    image2Hr[:image2.shape[0], :image2.shape[1]] = image2
#    image2Hr = transform.homography(image2Hr, Ht)
#    em = image2Hr.copy()
#    em[:image1.shape[0],:image1.shape[1]] = image1H
    em = image1H + image2
    em1 = em.copy()
    em1[:, 500:] = image2[:, 500:]

