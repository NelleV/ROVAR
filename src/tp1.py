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

THRESHOLD = 0.6

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

    eps = np.logical_and(eps1, eps)

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


def RANSAC(desc):
    """
    Estimates the homography between the two images
    """
    # Reformat desc
    redesc = np.ones((desc.shape[0], 6))
    redesc[:, :2] = desc[:, :2]
    redesc[:, 3:5] = desc[:, 2:]

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
            image[bresenham(el[0]+1, el[1]+1, el[2]+1, el[3]+1)] = 0

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
    descs = []
    for frame in frames:
        desc = image[frame[0] - 5:frame[0] + 5, frame[1] - 5:frame[1] + 5]
        edesc = []
        for i in desc:
            for j in i:
                edesc.append(j)
        descs.append(np.array(edesc))
    return frames, np.array(descs)


def stitchLR(image1, image2, points):
    """
    Stitch image1 to image2, with points points
    """
    # image 2 is base image. ie, we need to translate image1
    image1b = np.zeros((image1.shape[0], image1.shape[1] + 500))
    image1b[:, 500:] = image1
    image1 = image1b
    image2b = np.zeros((image2.shape[0], image2.shape[1] + 500))
    image2b[:, 500:] = image2
    image2 = image2b

    points[:, 0] += 500
    points[:, 2] += 500

    H2 = calculate_homography(points)

    Hr = H2.copy()
    # Hr[0, 2] = 0
    # Hr[1, 2] = 0
    image1H = transform.homography(image1, H2)
#    image2Hr = np.zeros((image2.shape[0], 1500))
#    image2Hr[:image2.shape[0], :image2.shape[1]] = image2
#    image2Hr = transform.homography(image2Hr, Ht)
#    em = image2Hr.copy()
#    em[:image1.shape[0],:image1.shape[1]] = image1H
    em = image1H + image2
    em[:, 500:] = image2[:, 500:]
    return em

def stitchRL(image2, image3, points23):
    """
    Stitch image1 to image2, with points points
    """
    s = image2.shape[1]
    image2b = np.zeros((image2.shape[0], image2.shape[1] + 500))
    image2b[:, :image2.shape[1]] = image2
    image2 = image2b
    image3b = np.zeros((image3.shape[0], image3.shape[1] + 500))
    image3b[:, :image3.shape[1]] = image3
    image3 = image3b

    #points23[:, 0] += 500
    #points23[:, 2] += 500

    H2 = calculate_homography(points23)

    Hr = H2.copy()
    # Hr[0, 2] = 0
    # Hr[1, 2] = 0
    image3H = transform.homography(image3, H2)
#    image2Hr = np.zeros((image2.shape[0], 1500))
#    image2Hr[:image2.shape[0], :image2.shape[1]] = image2
#    image2Hr = transform.homography(image2Hr, Ht)
#    em = image2Hr.copy()
#    em[:image1.shape[0],:image1.shape[1]] = image1H
    em = image2 + image3H
    em[:,:s] = image2[:, :s]
    return em


def oxford():
    mem = Memory(cachedir='.')
    image1 = mean(imread('keble_a.jpg'), 2)[::-1].astype(np.float)
    image2 = mean(imread('keble_b.jpg'), 2)[::-1].astype(np.float)
    image3 = mean(imread('keble_c.jpg'), 2)[::-1].astype(np.float)

    image1 = image1
    # image2 = image1 + 10
    image2 = image2

    # image2 = mean(imread('keble_b.jpg'), 2)[::-1]
    #FIXME we assume that image1.shape = image2.shape

    # the image is rotated
    coords1 = mem.cache(detect_harris_detector)(image1, threshold=.999)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = mem.cache(detect_harris_detector)(image2, threshold=.999)
    key_points2 = utils.create_frames_from_harris_points(coords2)

    coords3 = mem.cache(detect_harris_detector)(image3, threshold=.999)
    key_points3 = utils.create_frames_from_harris_points(coords3)


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
    f1, d1 = f1.transpose(), d1.transpose()

    # Get sift descriptors
    f2, d2 = mem.cache(vl_sift)(np.array(image2, 'f', order='F'),
                     frames=key_points2,
                     orientations=False)
    f2, d2 = f2.transpose(), d2.transpose()

    f3, d3 = mem.cache(vl_sift)(np.array(image3, 'f', order='F'),
                     frames=key_points3,
                     orientations=False)
    f3, d3 = f3.transpose(), d3.transpose()


    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
    image3 =  show_sift_desc(image3, f3)

    points12 = np.array([[340, 38, 51, 23],
                       [359, 340, 62, 337],
                       [691, 286, 395, 286],
                       [655, 116, 367, 128]])
    em12 = stitchLR(image1, image2, points12)

    points32 = np.array([[621, 18, 323, 35],
                         [323, 44, 11, 22],
                         [435, 398, 125, 400],
                         [349, 360, 34, 363],
                         [653, 336, 344, 340]])

    points23 = points32.copy()
    points23[:, :2] = points32[:, 2:]
    points23[:, 2:] = points32[:, :2]
    em23 = stitchRL(image2, image3, points23)
    em = np.zeros((image2.shape[0], image2.shape[1] + 1000))
    em[:, :image2.shape[1] + 500] = em12
    em[:, 500:] = em23

    imsave( "oxford.eps", em)
    return em

def breteuil():
#if __name__ == "__main__":

    mem = Memory(cachedir='.')
    if 1:
        image1 = mean(imread('breteuil_1.jpg'), 2)[::-1].astype(np.float)
        image2 = mean(imread('breteuil_2.jpg'), 2)[::-1].astype(np.float)
        image3 = mean(imread('breteuil_3.jpg'), 2)[::-1].astype(np.float)

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

    coords3 = mem.cache(detect_harris_detector)(image3, threshold=.999)
    key_points3 = utils.create_frames_from_harris_points(coords3)


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
    f1, d1 = f1.transpose(), d1.transpose()

    # Get sift descriptors
    f2, d2 = mem.cache(vl_sift)(np.array(image2, 'f', order='F'),
                     frames=key_points2,
                     orientations=False)
    f2, d2 = f2.transpose(), d2.transpose()

    f3, d3 = mem.cache(vl_sift)(np.array(image3, 'f', order='F'),
                     frames=key_points3,
                     orientations=False)
    f3, d3 = f3.transpose(), d3.transpose()


    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
    image3 =  show_sift_desc(image3, f3)
    match_image = show_matched_desc(image1, image2, matched_desc) 

    points12 = np.array([[386, 143, 157, 136],
                         [327, 261, 87, 255],
                         [738, 196, 489, 209],
                         [584, 246, 364, 251]])
    em12 = stitchLR(image1, image2, points12)

    points32 = np.array([[431, 178, 65, 180],
                         [545, 175, 190, 185],
                         [445, 307, 77, 318],
                         [720, 258, 347, 268],
                         [543, 206, 188, 213],
                         [740, 248, 367, 256]])

    points23 = points32.copy()
    points23[:, :2] = points32[:, 2:]
    points23[:, 2:] = points32[:, :2]
    em23 = stitchRL(image2, image3, points23)
    em = np.zeros((image2.shape[0], image2.shape[1] + 1000))
    em[:, :image2.shape[1] + 500] = em12
    em[:, 500:] = em23
    
    imsave("breteuil.png", em)
    return em

def random_partition(n, n_data):
    idxs = np.arange(n_data)
    np.random.shuffle(idxs)
    idxs1 = idxs[:n]
    idxs2 = idxs[n:]
    return idxs1, idxs2

def error_homography(H, data):
    X = np.ones((data.shape[0], 3))
    Y = X.copy()
    X[:, :2] = data[:, :2]
    Y[:, :2] = data[:, 2:]
    tX = np.dot(X, H)
    e = np.sqrt((tX - Y)**2)
    e = e.sum(axis=1)
    return e


#def test():
if __name__ == "__main__":

    mem = Memory(cachedir='.')
    if 1:
        image1 = mean(imread('keble_a.jpg'), 2)[::-1].astype(np.float)
        image2 = mean(imread('keble_b.jpg'), 2)[::-1].astype(np.float)
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
    coords1 = mem.cache(detect_harris_detector)(image1, threshold=.995)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = mem.cache(detect_harris_detector)(image2, threshold=.995)
    key_points2 = utils.create_frames_from_harris_points(coords2)

    # Rearrange the keypoints to be close
    if 0:
        import hungarian
        dist = euclidean_distances(key_points1.T, key_points2.T)
        ordering = mem.cache(hungarian.hungarian)(dist)[:, 1]
        key_points2 = key_points2[:, ordering]

    f1, d1 = mem.cache(nelle_desc)(image1,
                                   key_points1.T)

    f2, d2 = mem.cache(nelle_desc)(image2,
                                   key_points2.T)



    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
    match_image = show_matched_desc(image1, image2, matched_desc.copy()) 
    data = matched_desc

    t = 1000
    bestfit = None
    besterr = 10000000000000000
    best_inliners = None
    d = 14
    for iterations in range(50000):
        fit_data, test_data = random_partition(4, data.shape[0])
        fit_data = data[fit_data,:]
        test_data = data[test_data]
        
        fit_H = calculate_homography(fit_data)
        error = error_homography(fit_H, test_data)
        inliners = test_data[error < t]
        if 1:
            if len(inliners) > d:
                print error.min(), len(inliners)

        if len(inliners) > d:
            err = np.mean(error)
            if err < besterr:
                besterrr = err
                bestfit = fit_H
                best_inliners = np.concatenate((fit_data, inliners))

    match_image2 = show_matched_desc(image1, image2, best_inliners.copy())
    H2 = calculate_homography(best_inliners)
    en = stitchLR(image1, image2, best_inliners)
    # fit = calculate_homography(best_inliners)

#if __name__ == "__main__":
#    em_oxford = oxford()
#    em_breteuil = breteuil()
#
