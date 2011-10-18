import numpy as np
import scipy as sp

import harris

im = sp.lena()
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim, 6)
harris.plot_harris_points(im, filtered_coords)

