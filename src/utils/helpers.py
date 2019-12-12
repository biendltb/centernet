import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf


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

    return heat_map


def _log(x):
    """ Avoid x = 1 -> log(x) = 0 """
    if x == 1:
        return np.log(0.999)

    return np.log(x)


def heatmap_to_point_tf(heatmap):
    """ Convert the heat map to point and bounding box in Tensorflow
        Input tensor shape: batch_size * h * w * channel
    """
    # heatmap = tf.expand_dims(heatmap, axis=0)
    heatmap = np.array([heatmap for i in range(32)])
    heatmap = tf.expand_dims(heatmap, axis=-1)

    batch_size = heatmap.shape[0]
    h, w = heatmap.shape[1:3]

    gaussian_kernel = tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=tf.float32) / 16.0

    filters = gaussian_kernel[:, :, tf.newaxis, tf.newaxis]

    heatmaps_tensor = heatmap

    original_tensor = heatmaps_tensor

    heatmaps_tensor = tf.nn.conv2d(heatmaps_tensor, filters, strides=1, padding="SAME")

    h, w = heatmaps_tensor.shape[1:3]

    max_x = tf.math.argmax(tf.math.reduce_sum(heatmaps_tensor, axis=1), axis=1, output_type=tf.int32)[:, 0]
    max_y = tf.math.argmax(tf.math.reduce_sum(heatmaps_tensor, axis=2), axis=1, output_type=tf.int32)[:, 0]

    probs = tf.gather_nd(heatmaps_tensor, tf.stack([tf.range(batch_size), max_y, max_x, tf.zeros_like(max_y)], axis=-1))
    # probs = tf.stack([tf.reduce_max(tf.slice(original_tensor, [i, max_y[i]-2, max_x[i]-2, 0], [1, 5, 5, 1])) for i in range(batch_size)])

    pos_diff_h = tf.cast(
        tf.math.square(
            (tf.tile(tf.expand_dims(tf.range(h), axis=0), [batch_size, 1]) - tf.tile(tf.expand_dims(max_y, -1),
                                                                                     [1, h])) / (h - 1)
        ),
        tf.float32
    )
    bb_h = tf.reduce_mean(tf.sqrt(
        abs((pos_diff_h / (
                    2.0 * tf.math.log(tf.stack([heatmaps_tensor[i, :, max_x[i], 0] for i in range(batch_size)])))))),
        axis=1) * 2 * h

    pos_diff_w = tf.cast(
        tf.math.square(
            (tf.tile(tf.expand_dims(tf.range(w), axis=0), [batch_size, 1]) - tf.tile(tf.expand_dims(max_x, -1),
                                                                                     [1, w])) / (w - 1)
        ),
        tf.float32
    )
    bb_w = tf.reduce_mean(tf.sqrt(
        abs((pos_diff_w / (
                    2.0 * tf.math.log(tf.stack([heatmaps_tensor[i, max_y[i], :, 0] for i in range(batch_size)])))))),
        axis=1) * 2 * w

    out = tf.stack([tf.cast(max_y, tf.float32), tf.cast(max_x, tf.float32), bb_h, bb_w, probs], axis=-1)

    return max_y[0].numpy(), max_x[0].numpy(), bb_h[0].numpy(), bb_w[0].numpy(), probs[0].numpy()


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

