import tensorflow as tf
import os
import numpy as np

from src.utils import path_cvt
from src.utils import helpers

# flag to control the auto tune
AUTOTUNE = tf.data.experimental.AUTOTUNE

IM_SHAPE = (128, 128)


def _get_paths(file_list_path):

    with open(file_list_path, 'r') as f:
        lines = f.readlines()

    file_names = [l.strip() for l in lines]

    ds_dir = os.path.dirname(file_list_path)

    paths = [os.path.join(ds_dir, f_name) for f_name in file_names]

    return paths


def _load_ds(batch_size, im_paths, shuffle=True):
    # read all thermal data and load to memory
    thermal_ims = []
    heat_maps = []

    for path in im_paths:
        thermal_mat, centroid, bb_size = helpers.read_thermal_data(path)
        h_map = helpers.point_to_heatmap(centroid, bb_size, thermal_mat.shape)

        thermal_mat = np.expand_dims(thermal_mat, axis=-1)
        h_map = np.expand_dims(h_map, axis=-1)

        thermal_ims.append(thermal_mat)
        heat_maps.append(h_map)

    thermal_ds = tf.data.Dataset.from_tensor_slices((thermal_ims, heat_maps))

    ds = thermal_ds

    im_cnt = len(im_paths)
    if shuffle is True:
        ds = ds.shuffle(buffer_size=im_cnt)

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds, im_cnt


def load_all_ds(batch_size=1):
    train_list_path = path_cvt.get_path_to_train_dataset()
    train_paths = _get_paths(train_list_path)

    eval_list_path = path_cvt.get_path_to_eval_dataset()
    eval_paths = _get_paths(eval_list_path)
    
    train_ds, train_len = _load_ds(batch_size, train_paths)
    eval_ds, eval_len = _load_ds(batch_size, eval_paths, shuffle=False)

    print('Sucessfully load the dataset. TRAIN: {} | EVAL: {}'.format(train_len, eval_len))

    return train_ds, eval_ds


def load_vis_data(n=9):
    eval_list_path = path_cvt.get_path_to_eval_dataset()
    eval_paths = _get_paths(eval_list_path)

    # read all thermal data and load to memory
    thermal_ims = []
    heat_maps = []

    # take n images only
    for path in eval_paths[:n]:
        thermal_mat, centroid, bb_size = helpers.read_thermal_data(path)
        h_map = helpers.point_to_heatmap(centroid, bb_size, thermal_mat.shape)

        thermal_mat = np.expand_dims(thermal_mat, axis=-1)
        h_map = np.expand_dims(h_map, axis=-1)

        thermal_ims.append(thermal_mat)
        heat_maps.append(h_map)

    return np.array(thermal_ims), np.array(heat_maps)


if __name__ == '__main__':
    # TEST
    load_all_ds()
