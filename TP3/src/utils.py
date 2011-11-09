import numpy as np
from scipy import io
from matplotlib import pyplot as plt

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


if __name__ == "__main__":
    pos, neg = load_data()
    pos = normalise(pos)
    neg = normalise(neg)
    pos_reshaped = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    pos_reshaped = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

