import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import random

from skimage.draw import bresenham

negsamples = '../data/negsamples.mat'
possamples = '../data/possamples.mat'


def load_data():
    """
    Loads sample data from two matlab files

    Returns a tuple of numpy arrays, the first one containing positive
    samples, the other one negative samples. Each image is a grey scale 24*24
    array.

    Returns
        (positive samples, negative samples)
    """
    pos = io.matlab.mio.loadmat(possamples)
    neg = io.matlab.mio.loadmat(negsamples)
    return pos['possamples'], neg['negsamples']


def normalise(images):
    """
    Normalises the images

    Params
    --------
        images: array of images
    """

    mean = images.mean(axis=0).mean(axis=0)
    cov = np.sqrt(
        ((images - mean)** 2).sum(axis=0).sum(axis=0) / (images.shape[0] * \
                                                         images.shape[1]))
    norm_images = (images - mean) / cov
    return norm_images


def plot_gallery(images, title, h, w, n_row=3, n_col=4):
    """
    Plots a gallery

    Code stolen from sklearn - http://tinyurl.com/cmked78
    """

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subpltots_adjust(bottom=0, left=.01, right=.99, top=.90,
        hspace=.35)
    for i in range(n_row * n_col):
        plt.subpltot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)),
                  cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def format_data(shuffle=True):
    pos, neg = load_data()

    pos_norm = normalise(pos)
    neg_norm = normalise(neg)

    svm_pos = pos_norm.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    svm_neg = neg_norm.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

    # Let's format the data in a nicer way
    X = np.concatenate((svm_pos, svm_neg), axis=1).T
    y = np.concatenate((np.ones((svm_pos.shape[1])),
                        np.zeros((svm_neg.shape[1]))),
                    axis=1).T
    if shuffle:
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)
    return X[idxs,:], y[idxs,:]


def generate_bounding_boxes(image, pix=4):
    """
    Generates bounding boxes from a given image

    params
    ------
        image
        pix

    returns
    -------
        (ndarray, ndarray)
    """

    w, h = image.shape
    boxes = []
    positions = []
    for i in range((w - 24) / pix):
        for j in range((h - 24) / pix):
            boxes.append(image[i * pix:i * pix + 24, j * pix:j * pix + 24])
            positions.append([i * pix, j * pix])
    # FIXME boxes are not in the correct order
    boxes = np.array(boxes).transpose((1, 2, 0))
    norm_boxes = normalise(boxes)
    reshaped_boxes = norm_boxes.reshape(
                        (norm_boxes.shape[0] * norm_boxes.shape[1],
                         norm_boxes.shape[2]))
    return np.array(positions), reshaped_boxes.T


def show_positive_boxes(image, labels, positions):
    """
    Returns an image with the positive bounding boxes drawn

    params
    -------
        image
        labels

    returns
    -------
        image
    """
    image = image.copy()
    for i, label in enumerate(labels):
        if label:
            x0 = positions[i, 0]
            y0 = positions[i, 1]
            image[bresenham(x0, y0, x0 + 24, y0)] = 0
            image[bresenham(x0, y0, x0, y0 + 24)] = 0
            image[bresenham(x0 + 24, y0, x0 + 24, y0 + 24)] = 0
            image[bresenham(x0, y0 + 24, x0 + 24, y0 + 24)] = 0
    return image


def merge_bounding_boxes(image, scores, positions, threshold=0.5, dist=5):
    """
    Merges bounding boxes
    """
    image = image.copy()
    num = 0
    calc = scores > threshold
    correct_positions = positions[calc]
    correct_scores = scores[calc]
    for position, score in zip(correct_positions, correct_scores):
        num += 1
        x0 = position[0]
        y0 = position[1]
        image[bresenham(x0, y0, x0 + 24, y0)] = 0
        image[bresenham(x0, y0, x0, y0 + 24)] = 0
        image[bresenham(x0 + 24, y0, x0 + 24, y0 + 24)] = 0
        image[bresenham(x0, y0 + 24, x0 + 24, y0 + 24)] = 0
    print "detected %d faces" % num
    return image, correct_positions, correct_scores


def create_heat_map(image, scores, positions):
    """
    Create a heatmap

    params
    -------
        image
        scores
        positions
    """
    image = np.zeros(image.shape)
    for position, score in zip(positions, scores):
        x0 = position[0] + 12
        y0 = position[1] + 12
        image[x0, y0] = score
    return image


def find_centroids(positions, scores, min_dist=25, direc=0):
    """
    Find centroids, and merge boxes
    """
    from sklearn.metrics.pairwise import euclidean_distances
    centroids = []
    positions += 12
    centroid_scores = []
    for position, score in zip(positions, scores):
        if not centroids:
            # initialise first centroid
            centroids.append(position)
            centroid_scores.append(score)
        dist = euclidean_distances(position, np.array(centroids))
        best_centroid = dist.argmin()
        best_dist = dist.min()
        if best_dist < min_dist:
            if score > centroid_scores[best_centroid]:
                centroid_scores[best_centroid] = score
                centroids[best_centroid] = position
        else:
            # add a new centroid
            centroids.append(position)
            centroid_scores.append(score)
    return np.array(centroids), np.array(centroid_scores)




if __name__ == "__main__":
    pos, neg = load_data()
    pos = normalise(pos)
    neg = normalise(neg)
    pos_reshaped = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    pos_reshaped = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))
    from matplotlib.pyplot import imread
    image = imread('../data/img1.jpg')[::-1].mean(axis=2)
    positions, boxes = generate_bounding_boxes(image)

