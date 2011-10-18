import os

import numpy as np
from scipy import ndimage
from scipy import linalg
import itertools

from pylab import imread, imsave, imshow, mean

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals.joblib import Memory

from skimage import transform
from skimage import feature

from PIL import Image

from harris import harris

import sift
from vlfeat import vl_sift

import utils
import ransac
import ransac_model

THRESHOLD = 0.70

ITERATIONS = 50

def detect_harris_detector(image, threshold=0.1, min_distance=10):
    """
    Detects harris points

    params
    -------
        image: numpy array

    returns
    -------
        coords: list of coords

    """
    im1 = image
    harrisim1 = harris.compute_harris_response(im1)
    points1 = harris.get_harris_points(harrisim1,
                                       min_distance=min_distance,
                                       threshold=threshold)

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
        distances[X[0], X[1]] = distances.max()
    distances_N2 = distances.min(axis=1)

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
    image = im.copy()
    for element in l:
        draw_point(image, *element)

    return image


def show_sift_desc(im, f):
    image = im.copy()

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
        x1 = X[1]
        y1 = X[0]
        x2 = X[3]
        y2 = X[2]

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


def show_matched_desc(image1, image2, matched_desc):
    image = np.zeros((image1.shape[0], image2.shape[1] + 10 + image2.shape[1]))
    image[:, :image1.shape[1]] = image1
    image[:, 10 + image2.shape[1]:] = image2
    placed_desc = matched_desc
    placed_desc[:, -1] = matched_desc[:, -1] + 10 + image2.shape[1]
    from skimage.draw import bresenham
    for el in placed_desc:
        try:
            draw_point(image, el[0], el[1])
            draw_point(image, el[2], el[3])
            image[bresenham(el[0], el[1], el[2], el[3])] = 0
        except IndexError:
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
    descs = []
    image_g = ndimage.gaussian_filter(image, 1)
    #image_g = image
    for frame in frames:
        desc = image_g[frame[0] - 4:frame[0] + 4, frame[1] - 4:frame[1] + 4]
        edesc = []
        for x, i in enumerate(desc):
            if x in [0, 8]:
                wx = 1.
            elif x in [1, 7]:
                wx = 1.1
            elif x in [2, 6]:
                wx = 1.3
            elif x in [3, 5]:
                wx = 1.7
            else:
                wx = 2.5
            for y, j in enumerate(i):
                if y in [0, 8]:
                    wy = 1.
                elif y in [1, 7]:
                    wy = 1.1
                elif y in [2, 6]:
                    wy = 1.3
                elif y in [3, 5]:
                    wy = 1.7
                else:
                    wy = 2.5
                edesc.append((wx * wy * j) / 10)
        descs.append(np.array(edesc))
    return frames, np.array(descs)


def hog_desc(image, frames):
    desc = feature.hog(image)


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

    points[:, 1] += 500
    points[:, 3] += 500

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
    coords1 = mem.cache(detect_harris_detector)(image1, threshold=.995)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = mem.cache(detect_harris_detector)(image2, threshold=.995)
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
    f1, d1 = mem.cache(vl_sift)(np.array(image1, 'f', order='F'))
                     #frames=key_points1,
                     #orientations=False)
    f1, d1 = f1.transpose(), d1.transpose()

    # Get sift descriptors
    f2, d2 = mem.cache(vl_sift)(np.array(image2, 'f', order='F'))
                     #frames=key_points2,
                     #orientations=False)
    f2, d2 = f2.transpose(), d2.transpose()

#    f3, d3 = mem.cache(vl_sift)(np.array(image3, 'f', order='F'),
#                     frames=key_points3,
#                     orientations=False)
#    f3, d3 = f3.transpose(), d3.transpose()
#

    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc = np.array(matched_desc)
    matches_d = np.array(matches_d)
    match_image = show_matched_desc(image1.copy(), image2.copy(), matched_desc.copy())
    return match_image

    image1 =  show_sift_desc(image1, f1)
    image2 =  show_sift_desc(image2, f2)
    image3 =  show_sift_desc(image3, f3)

#    points12 = np.array([[340, 38, 51, 23],
#                         [359, 340, 62, 337],
#                         [691, 286, 395, 286],
#                         [655, 116, 367, 128]])
    points12 = np.array([[38, 340, 23, 51],
                         [340, 359, 337, 62],
                         [286, 691, 286, 395],
                         [116, 655, 128, 367]])

    em12 = stitchLR(image1, image2, points12)
    #return em12

    points32 = np.array([[621, 18, 323, 35],
                         [323, 44, 11, 22],
                         [435, 398, 125, 400],
                         [349, 360, 34, 363],
                         [653, 336, 344, 340]])

    points23 = np.array([[35, 323, 18, 621],
                         [22, 11, 44, 323],
                         [400, 125, 398, 435],
                         [363, 34, 360, 349],
                         [340, 344, 336, 653]])


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
    coords1 = mem.cache(detect_harris_detector)(image1, threshold=.995)
    key_points1 = utils.create_frames_from_harris_points(coords1)

    # the image is rotated
    coords2 = mem.cache(detect_harris_detector)(image2, threshold=.995)
    key_points2 = utils.create_frames_from_harris_points(coords2)

    coords3 = mem.cache(detect_harris_detector)(image3, threshold=.995)
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


def ransac(data):
    t = 500

    bestfit = None
    besterr = 10000000000000
    best_inliners = None
    d = 2
    max_d = 2
    for iterations in range(100000):
        fit_data, test_data = random_partition(4, data.shape[0])
        fit_data = data[fit_data,:]
        test_data = data[test_data]
        fit_H = calculate_homography(fit_data)
        error = error_homography(fit_H, test_data)
        inliners = test_data[error < t]
        if 1:
            if len(inliners) > d:
                print error.min(), len(inliners), besterr

        err = np.mean(error) / len(inliners)
        if len(inliners) > max_d:
            besterr = err
            bestfit = fit_H
            max_d = len(inliners)
            best_inliners = np.concatenate((fit_data, inliners))
    return best_inliners, bestfit


#def test():
if __name__ == "__main__":

    mem = Memory(cachedir='.')
    if 1:
        image1 = mean(imread('keble_a.jpg'), 2)[::-1].astype(np.float)
        image2 = mean(imread('keble_b.jpg'), 2)[::-1].astype(np.float)
        image3 = mean(imread('keble_c.jpg'), 2)[::-1].astype(np.float)

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

    coords3 = mem.cache(detect_harris_detector)(image3, threshold=.995)
    key_points3 = utils.create_frames_from_harris_points(coords3)


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

    f3, d3 = mem.cache(nelle_desc)(image3,
                                   key_points3.T)


    matched_desc, matches_d = match_descriptors(d1, d2, f1, f2)
    matched_desc1, matches_d = match_descriptors(d2, d3, f2, f3)

    matched_desc = np.array(matched_desc)
    matched_desc1 = np.array(matched_desc1)

    #image1 =  show_sift_desc(image1, f1)
    #image2 =  show_sift_desc(image2, f2)
    match_image = show_matched_desc(image1.copy(), image2.copy(), matched_desc.copy()) 
    data = matched_desc.copy()
    best_inliners, fit_H = mem.cache(ransac)(data)
    match_inliners = show_matched_desc(image1.copy(), image2.copy(), best_inliners.copy())
    en = stitchLR(image1, image2, best_inliners.copy())

#    data = matched_desc1.copy()
#    # We want 3 to 2
#    data[:, :2] = matched_desc1[:, 2:]
#    data[:, :2] = matched_desc1[:, 2:]
#
#    best_inliners, fit_H = mem.cache(ransac)(matched_desc1)
#    em = stitchRL(image2, image3, best_inliners.copy())
#    panorama = np.zeros((image2.shape[0], image2.shape[1] + 1000))
#    panorama[:, :image2.shape[1] + 500] = en
#    panorama[:, 500:] = em
#

    # fit = calculate_homography(best_inliners)

#if __name__ == "__main__":
#    em_oxford = oxford()
#    em_breteuil = breteuil()
#
