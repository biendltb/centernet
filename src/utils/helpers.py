import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from src.datasets import image_transformer


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

    if sigma_x <= 0 or sigma_y <=0:
        return np.zeros(map_shape)

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

    # original_tensor = heatmaps_tensor

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

    # set width is the minimum of width and height
    bb_w = tf.math.minimum(bb_h, bb_w)

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


# PARSE HEAT MAP TO BOXES
def heatmap_to_boxes(heat_map, min_power=0.95, min_dist=10):
    """ Non-maxima suppression strategy:
        1) Remove all pixels which power < 0.95
        2) Repeat until all pixels are zero
        3) Get the max power point in the map
        4) Set all points which distance to the max-power point is less than min_dist to zero
        5) Keep the max_point and go back to 2)

    """
    candidates = []
    h, w = heat_map.shape[:2]

    max_points, mp_powers = _generate_center_candidate(heat_map, min_power=min_power, min_dist=min_dist)

    if len(max_points) == 0:
        return candidates

    diff_maps = []

    while len(max_points) > 0:
        pnt, _ = max_points.pop(0), mp_powers.pop(0)
        _can, _diff_map, _curr_hmap = _find_gau(pnt, heat_map)
        max_points, mp_powers = _refine_maxpoints(_curr_hmap, max_points, mp_powers)
        candidates.append(_can)
        diff_maps.append(_diff_map)

    # assign pixel for each heat map
    assign_map = np.argmin(diff_maps, axis=0)
    unique, counts = np.unique(assign_map, return_counts=True)

    # calculate scores for each Gaussian distribution based on the number of assigned pixels
    scores = []
    for i, can in enumerate(candidates):
        c_x, c_y, bb_w, bb_h = can[:4]
        # find the intersection area of bounding box with the image frame
        x1, y1 = max(0, c_x - bb_w/2), max(0, c_y - bb_h/2)
        x2, y2 = min(w, c_x + bb_w/2), min(h, c_y + bb_h/2)
        inter_w = x2 - x1
        inter_h = y2 - y1
        scores.append(counts[i] / (inter_w * inter_h))

    candidates, scores = np.array(candidates), np.array(scores)

    candidates = candidates[scores > 0.1]

    # use non-maxima suppression to edge out the duplicated candidates
    if len(candidates) > 1:
        tf_boxes = []
        tf_scores = []
        for can in candidates:
            c_x, c_y, bb_w, bb_h, score_x, score_y, prob = can

            x1, x2 = int(c_x - bb_w / 2), int(c_x + bb_w / 2)
            y1, y2 = int(c_y - bb_h / 2), int(c_y + bb_h / 2)

            tf_boxes.append([y1, x1, y2, x2])
            tf_scores.append(prob)

        # apply IoU non-maxima suppression
        selected_ids = tf.image.non_max_suppression(tf_boxes, tf_scores, len(tf_boxes))
        candidates = candidates[selected_ids.numpy()]


    # parse bounding box coordinates to ratios
    candidates = _parse_coordinates(candidates, heat_map.shape)

    return candidates


def _generate_center_candidate(heat_map, min_power, min_dist):
    """ Generate the center candidates which meet the min power
        Minimum distance is the minimum centroid distance
    """
    tmp_heat_map = heat_map.copy()
    max_points = []
    mp_powers = []
    h, w = tmp_heat_map.shape[:2]

    # remove all pixels which power < 0.95
    tmp_heat_map[tmp_heat_map < min_power] = 0

    while np.sum(tmp_heat_map) > 0:
        # get the max power point in the map
        max_y, max_x = np.unravel_index(tmp_heat_map.argmax(), tmp_heat_map.shape)

        # remove points that's in the min distance
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        dist_map = np.sqrt((x_grid - np.ones_like(x_grid) * max_x) ** 2 + (y_grid - np.ones_like(y_grid) * max_y) ** 2)
        tmp_heat_map[dist_map < min_dist] = 0

        # keep the max_point
        max_points.append((max_y, max_x))
        mp_powers.append(heat_map[max_y, max_x])

    return max_points, mp_powers


def _find_gau(pnt, heat_map):
    """ Find the best gaussian estimation for the point
        Return:
            + Gaussian params: (center_x, center_y), (sigma_x, sigma_y), mask
            + Mask of coverage area
    """
    hmap = heat_map.copy()

    # take horizontal and vertical which go through the max point
    max_y, max_x = pnt
    h_line = hmap[max_y, :]
    v_line = hmap[:, max_x]

    # to avoid log(n) = 0
    h_line[h_line == 1] = 0.9999
    v_line[v_line == 1] = 0.9999
    h_line[h_line == 0] = 0.0001
    v_line[v_line == 0] = 0.0001

    # find sigmas
    h, w = hmap.shape
    bb_hs = np.sqrt(abs(-((np.arange(h) - np.ones(h) * max_y) / (h - 1)) ** 2 / (2 * np.log(v_line)))) * 2 * h
    bb_ws = np.sqrt(abs(-((np.arange(w) - np.ones(w) * max_x) / (w - 1)) ** 2 / (2 * np.log(h_line)))) * 2 * w

    bb_w, bb_w_score = _get_mean_median(bb_ws, bins=100)
    bb_h, bb_h_score = _get_mean_median(bb_hs, bins=100)

    # probability is the power at the center
    prob = heat_map[max_y, max_x]

    # Find the power difference on every pixels, later on this information will be used to assign pixels to
    # the Gaussian distribution which has least difference
    _c_x, _bb_w = max_x / w, bb_w / w
    _c_y, _bb_h = max_y / h, bb_h / h
    curr_hmap = point_to_heatmap((_c_x, _c_y), (_bb_w, _bb_h), heat_map.shape)

    diff_map = np.abs(curr_hmap - heat_map)

    return [max_x, max_y, bb_w, bb_h, bb_w_score, bb_h_score, prob], diff_map, curr_hmap


def _refine_maxpoints(curr_hmap, max_points, mp_powers):
    """ Remove proposed key points which is possibly a point of the current distribution
        If it is, the power should be lower than the power at that point
    """
    passed_mp = []
    passed_mp_powers = []
    for i, pnt in enumerate(max_points):
        if curr_hmap[pnt] < mp_powers[i]:
            passed_mp.append(pnt)
            passed_mp_powers.append(mp_powers[i])

    return passed_mp, passed_mp_powers


def _get_mean_median(vector1d, bins=100):
    """ Get the most frequent bin from the histogram, then take the mean of values fall in that bin
    """
    # allow the size of the bounding box to be more than 100% of the image size
    hist, bin_edges = np.histogram(vector1d, bins=bins, range=(5, int(len(vector1d)*1.1)))
    max_id = np.argmax(hist)
    score = hist[max_id]

    low_edge = bin_edges[max_id]
    high_stop = bin_edges[max_id+1]

    edge_len = np.mean(vector1d[(low_edge <= vector1d) & (vector1d < high_stop)])
    score /= edge_len

    return edge_len, score


def _parse_coordinates(candidates, im_shape):
    """ Parse coordinates of the bounding boxes to ratio
    """
    outputs = []

    h, w = im_shape[:2]

    for _can in candidates:
        c_x, c_y, bb_w, bb_h, score_x, score_y, prob = _can

        outputs.append([c_x/w, c_y/h, bb_w/w, bb_h/h, score_x, score_y, prob])

    return outputs


def draw_bb_on_im(heat_map, vis_im, add_hmap=False):
    """ Take boxes from heat map and draw the bounding boxes
    """
    boxes = heatmap_to_boxes(heat_map)

    h, w = vis_im.shape[:2]

    for box in boxes:
        c_x, c_y, bb_w, bb_h, score_x, score_y, prob = box[:7]
        c_x, c_y, bb_w, bb_h = c_x * w, c_y * h, bb_w * w, bb_h * h

        x1, x2 = int(c_x - bb_w / 2), int(c_x + bb_w / 2)
        y1, y2 = int(c_y - bb_h / 2), int(c_y + bb_h / 2)

        vis_im = cv2.rectangle(vis_im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        vis_im = cv2.circle(vis_im, (int(c_x), int(c_y)), 1, (0, 255, 0), 2)

    if add_hmap:
        # add heat map to the visualisation image
        hmap_im = cvt_hmap_to_im(heat_map)
        vis_im = tf.concat([vis_im, hmap_im], axis=1)

    return vis_im


def denorm_im(im):
    """ Denormalise the image
    """
    return np.array(im * 127.5 + 127.5).astype(np.uint8)


def load_im(im, hmap):
    """ Load image and pre-process
    """

    im = im_preprocess(im)

    if hmap is not None:
        hmap = hmap[:, :, tf.newaxis]

    return im, hmap


def load_im_from_path(im_path, hmap):
    """ Load image from image path and resize
    """
    im = read_im_from_path(im_path)

    im = im_preprocess(im)

    if hmap is not None:
        hmap = hmap[:, :, tf.newaxis]

    return im, hmap


def read_im_from_path(im_path):
    # load im
    im = tf.io.read_file(im_path)
    im = tf.image.decode_png(im, channels=3)
    im = tf.image.resize(im, (320, 320))

    return im


def im_preprocess(im):
    # IM_SHAPE = (224, 224)
    # im = tf.image.resize(im, IM_SHAPE[:2])
    im = tf.cast(im, tf.float32)
    im = (im - 127.5) / 127.5

    return im


def cvt_hmap_to_im(hmap):
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=hmap.min(), vmax=hmap.max())
    im = cmap(norm(hmap))

    im = (im[:, :, :3] * 255).astype(np.uint8)

    return im


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # heat_map = point_to_heatmap((0.5427, 0.6757), (0.2181, 0.1458), (120, 160))

    pass

