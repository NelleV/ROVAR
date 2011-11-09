import numpy as np
from scipy import io

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


if __name__ == "__main__":
    pos, neg = load_data()
    pos = normalise(pos)
    neg = normalise(neg)
    pos_reshaped = pos.reshape((pos.shape[0] * pos.shape[1], pos.shape[2]))
    pos_reshaped = neg.reshape((neg.shape[0] * neg.shape[1], neg.shape[2]))

