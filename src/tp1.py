import os

import numpy as np
import scipy as sp
import itertools

from pylab import imread, imsave, imshow, mean

from scikits.learn.metrics.pairwise import euclidean_distances

from PIL import Image

from harris import harris

import sift


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
    length = min(image.shape) - 1

    # First compute Harris on the first half of the image
    im1 = image[0:length, 0:length]
    harrisim1 = harris.compute_harris_response(im1)
    points1 = harris.get_harris_points(harrisim1)

    # Then the second half of the image
    im2 = image[image.shape[0] - length:, image.shape[1] - length:]
    harrisim2 = harris.compute_harris_response(im2)
    points2 = harris.get_harris_points(harrisim2)
    # the second set of points have been translated of image.shape[0] - length
    # and image.shape[1] - length
    harris_points = set((image.shape[0] - length + x[0],
                         image.shape[1] - length + x[1]) for x in points2)


    harris_points = harris_points.union(set(x for x in points1))
    return harris_points
    # harris_points = harris_points.union(set(length + x for x in points2))
    # And merge the results

def get_sift_descriptors(image_path):
    """
    get the sift descriptors of the image
    """
    key_path = image_path.split('.')[0] + '.key'
    sift.process_image(image_path, key_path)
    l1, d1 = sift.read_features_from_file(key_path)
    return l1, d1

def match_descriptors(d1, d2):
    distances = euclidean_distances(d1, d2)
    # the nearest neighbour is the one for which the euclidean distance is the
    # smallest
    return distances.argmin(axis=0)

def show_descriptors(im, l):
    """ Show the image with overlaying descriptors"""
    # im and l are two matrix. im is a greyscale image, and l the list of
    # descriptors and their coord.
    pass

def show_matches(im1, im2, l1, l2, d):
    pass
    

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
    image = mean(imread('keble_a.jpg'), 2)[::-1]
    # the image is rotated
    coords = detect_harris_detector(image)

    # Get the sift descriptors of the image using the pgm converted version of
    # the image
    # l1, d1 = get_sift_descriptors('keble_a.pgm')
    # l2, d2 = get_sift_descriptors('keble_b.pgm')
    # l3, d3 = get_sift_descriptors('keble_c.pgm')


