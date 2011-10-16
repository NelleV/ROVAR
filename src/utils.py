import numpy as np

def create_frames(image, sample=5):
    ur = np.arange(0, image.shape[1], 5)
    vr = np.arange(0, image.shape[0], 5)

    [u,v] = np.meshgrid(ur, vr) 

    f = np.array([u.ravel(), v.ravel()])
    K = f.shape[1]
    f = np.vstack([f, 2 * np.ones([1, K]), 0 * np.ones([1, K])])

    return np.array(f, order='F')


def create_frames_from_harris_points(key_points):
    """
    key_points are set of (x, y)
    """

    # FIXME this computes an n*n array, and not exactly what we want...

    ur = np.zeros(len(key_points))
    vr = np.zeros(len(key_points))
    for i, point in enumerate(key_points):
        ur[i] = point[0]
        vr[i] = point[1]
    # [u,v] = np.meshgrid(ur, vr)

    f = np.array([ur.ravel(), vr.ravel()])
    K = f.shape[1]
    f = np.vstack([f, 2 * np.ones([1, K]), 0 * np.ones([1, K])])

    return np.array(f, order='F')

