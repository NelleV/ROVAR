#
# Harris detector
#
# http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html

import numpy as np
from scipy import stats, ndimage

from matplotlib import pyplot


def compute_harris_response(image, eps=1e-6):
    """ compute the Harris corner detector response function
        for each pixel in the image"""

    # derivatives
    image = ndimage.gaussian_filter(image, 1)
    imx = ndimage.sobel(image, axis=0, mode='constant')
    imy = ndimage.sobel(image, axis=1, mode='constant')

    Wxx = ndimage.gaussian_filter(imx * imx, 1.5, mode='constant')
    Wxy = ndimage.gaussian_filter(imx * imy, 1.5, mode='constant')
    Wyy = ndimage.gaussian_filter(imy * imy, 1.5, mode='constant')

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    harris = Wdet / (Wtr + eps)

    # Non maximum filter of size 3
    harris_max = ndimage.maximum_filter(harris, 3, mode='constant')
    harris *= harris == harris_max
    # Remove the image corners
    harris[:3] = 0
    harris[-3:] = 0
    harris[:, :3] = 0
    harris[:, -3:] = 0

    return harris


def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating
        corners and image boundary"""

    corner_threshold = np.max(harrisim.ravel()) * threshold
    # find top corner candidates above a threshold
    # corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim >= corner_threshold) * 1

    # get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [(candidates[0][c], candidates[1][c]) for c
               in range(len(candidates[0]))]
    # ...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,
                      min_distance:-min_distance] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
              (coords[i][0] - min_distance):(coords[i][0] + min_distance),
              (coords[i][1] - min_distance):(coords[i][1] + min_distance)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    pyplot.subplot(111)
    pyplot.imshow(image)
    pyplot.plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],
                '*')
    pyplot.axis('off')
    pyplot.show()


if __name__ == '__main__':
    import scipy as sp
    im = sp.lena().astype(float)
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim, 6)
    plot_harris_points(im, filtered_coords)
