import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import random


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

    svm_pos = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    svm_neg = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

    # Let's format the data in a nicer way
    X = np.concatenate((svm_pos, svm_neg), axis=1).T
    y = np.concatenate((np.ones((svm_pos.shape[1])),
                        np.zeros((svm_neg.shape[1]))),
                    axis=1).T
    if shuffle:
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)
    return X[idxs,:], y[idxs,:]


def generate_bounding_boxes(image):
    """
    Generates all bounding boxes from a given image
    """
    w, h = image.shape
    boxes = []
    for i in range(w - 24):
        for j in range(h - 24):
            boxes.append(image[i:i + 24, j:j + 24])
    boxes = np.array(boxes).T
    norm_boxes = normalise(boxes)
    reshaped_boxes = norm_boxes.reshape(
                        (norm_boxes.shape[0] * norm_boxes.shape[1],
                         norm_boxes.shape[2]))
    return reshaped_boxes.T


def merge_bounding_boxes(image, boxes, labels):
    """
    Merges bounding boxes
    """
    print "" 

if __name__ == "__main__":
    pos, neg = load_data()
    pos = normalise(pos)
    neg = normalise(neg)
    pos_reshaped = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    pos_reshaped = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

