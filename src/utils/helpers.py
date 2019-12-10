import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter


def thermal_preprocess(thermal_mat):
    """ Convert the raw data from thermal sensor to grayscale image data for visualization
            Steps:
            1) Keep only data in range of 30 to 40 degree Celsius
            2) Set values out of this range to 30 degree Celsius
            3) Scale 30 - 40 to 0 - 1
        """
    # base value to convert from Kelvin to Celsius
    base_temp = 27315
    _min = 30 * 100
    _max = 40 * 100

    thermal_mat -= base_temp
    thermal_mat[(thermal_mat < _min) | (thermal_mat > _max)] = _min
    thermal_mat = (thermal_mat - _min) / (10.0 * 100)

    return thermal_mat


def read_thermal_data(path):
    """ Read thermal frame and its meta-data from HDF5 format

        Return: thermal mat, bounding box centre, width, height
        Note: all meta-data are ratios
    """
    im_key = 'image'
    label_key = 'label'

    with h5py.File(path, 'r') as f:
        thermal_mat = f[im_key].value
        label_str = f[im_key].attrs[label_key]

    thermal_mat = thermal_preprocess(thermal_mat)
    # convert to float32 format (compatible with TF2.0)
    thermal_mat = thermal_mat.astype(np.float32)

    c_x, c_y, bb_w, bb_h = [float(s) for s in label_str.split()[1:]]

    return thermal_mat, (c_x, c_y), (bb_w, bb_h)


def point_to_heatmap(point, bb_size, map_shape):
    """ Generate the heat map from the centroid
    Note: coordinates of centroid must be normalised by x and y

    G = exp(-((x - x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))))
    """
    h, w = map_shape
    c_x, c_y = point

    bb_w, bb_h = bb_size

    sigma_x = bb_w/2
    sigma_y = bb_h/2

    x_grid, y_grid = np.meshgrid(np.arange(w)/(w-1), np.arange(h)/(h-1))

    heat_map = np.exp(
        -(
            ((x_grid.reshape(-1) - np.ones(w * h) * c_x)**2) / (2 * (np.ones(w * h) * sigma_x)**2) +
            ((y_grid.reshape(-1) - np.ones(w * h) * c_y)**2) / (2 * (np.ones(w * h) * sigma_y)**2)
        )
    )

    # convert to float32 format (compatible with TF2.0)
    heat_map = heat_map.reshape(map_shape).astype(np.float32)

    # # normalise the heat map to 0 -> 1
    # _min = np.min(heat_map)
    # _max = np.max(heat_map)
    # heat_map = (heat_map - _min)/(_max - _min)

    return heat_map


def _log(x):
    """ Avoid x = 1 -> log(x) = 0 """
    if x == 1:
        return np.log(0.999)

    return np.log(x)


def heatmap_to_point(heat_map):
    # Use a 3x3 kernel to smooth the heat map
    heat_map = gaussian_filter(heat_map, sigma=3)
    # key_point = np.unravel_index(np.argmax(heat_map), heat_map.shape)

    max_x = np.argmax(np.sum(heat_map, axis=0))
    max_y = np.argmax(np.sum(heat_map, axis=1))

    prob = heat_map[max_y, max_x]

    # find sigmas
    h, w = heat_map.shape
    bb_h = np.mean([np.sqrt(abs(-((i - max_y)/(h - 1))**2 / (2 * _log(heat_map[i, max_x])))) for i in range(h)]) * 2 * h
    bb_w = np.mean([np.sqrt(abs(-((i - max_x)/(w - 1))**2 / (2 * _log(heat_map[max_y, i])))) for i in range(w)]) * 2 * w

    return (max_y, max_x), (bb_h, bb_w), prob


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    heat_map = point_to_heatmap((0.5427, 0.6757), (0.2181, 0.1458), (120, 160))

