""" Load images and labels from datasets
"""

import tensorflow as tf
import numpy as np

from src.datasets.fddb import FDDB

IM_SHAPE = (224, 224, 3)

# flag to control the auto tune
AUTOTUNE = tf.data.experimental.AUTOTUNE


def _load_im(im_path, hmap):
    """ Load image from image path and resize
    """
    # load im
    im = tf.io.read_file(im_path)
    im = tf.image.decode_png(im, channels=3)
    im = tf.image.resize(im, IM_SHAPE[:2])
    im = (im - 127.5) / 127.5

    if hmap is not None:
        hmap = hmap[:, :, tf.newaxis]

    return im, hmap


def create_tf_ds(data, batch_size=1):
    im_cnt = len(data[0])
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(_load_im, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=im_cnt)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def load_ds(batch_size=1):
    fddb = FDDB(eval_set=9)
    # load data and labels as tuples
    train_data, eval_data = fddb.load_ds()

    train_ds = create_tf_ds(train_data, batch_size=batch_size)
    eval_ds = create_tf_ds(eval_data, batch_size=batch_size)

    return train_ds, eval_ds


def load_vis_data(n=9):
    fddb = FDDB()

    im_paths, heat_maps = fddb.load_by_fold_id(fold_id=9)

    ids = np.arange(len(im_paths))
    np.random.seed(17)
    np.random.shuffle(ids)

    vis_im_paths = np.array(im_paths)[ids][:n]
    vis_hmaps = np.array(heat_maps)[ids][:n]

    # load image
    vis_ims = []
    for im_path in vis_im_paths:
        im, _ = _load_im(im_path, None)
        # use batch_size=1 for visualisation
        im = im[tf.newaxis, :, :, :]
        vis_ims.append(im)

    return vis_ims, vis_hmaps


if __name__ == '__main__':
    # train_ds, eval_ds = load_ds()
    # for im, label in train_ds:
    #     print(im.shape)

    vis_ims, hmaps = load_vis_data(n=9)

    pass