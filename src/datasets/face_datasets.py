""" Load images and labels from datasets
"""

import tensorflow as tf
import numpy as np

from src.datasets.fddb import FDDB
from src.datasets.widerface import WIDER
from src.utils import helpers

# flag to control the auto tune
AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_tf_ds(data, load_func, batch_size=1):
    im_cnt = len(data[0])
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(load_func, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=im_cnt)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    print('Create dataset with {} elements'.format(im_cnt))

    return ds


def load_ds(batch_size=1):
    data_augmentation = True
    use_path = False
    load_func = helpers.load_im

    # # FDDB
    # fddb = FDDB(eval_set=9)
    # # load data and labels as tuples
    # train_data, eval_data = fddb.load_ds(train_augm=False, use_path=True)
    # # _, eval_data = fddb.load_ds()

    # WIDER FACE DATASET
    wider_face = WIDER(im_shape=(128, 128, 3))
    train_data, eval_data = wider_face.load_ds(augmentation=data_augmentation, use_path=use_path)

    print('====== TRAIN DATA: {} | VALIDATION DATA: {} ======'.format(len(train_data[0]), len(eval_data[0])))

    train_ds = create_tf_ds(train_data, load_func=load_func, batch_size=batch_size)
    eval_ds = create_tf_ds(eval_data, load_func=load_func, batch_size=batch_size)

    return train_ds, eval_ds


def load_vis_data(n=9):
    # FDDB
    fddb = FDDB()
    im_paths, heat_maps = fddb.load_by_fold_id(fold_id=9)

    # # WIDER FACE DATASET
    # wider_face = WIDER()
    # im_paths, heat_maps = wider_face.load_val_ds()

    ids = np.arange(len(im_paths))
    np.random.seed(17)
    np.random.shuffle(ids)

    vis_im_paths = np.array(im_paths)[ids][:n]
    vis_hmaps = np.array(heat_maps)[ids][:n]

    # load image
    vis_ims = []
    for im_path in vis_im_paths:
        im, _ = helpers.load_im_from_path(im_path, None)
        # use batch_size=1 for visualisation
        im = im[tf.newaxis, :, :, :]
        vis_ims.append(im)

    return vis_ims, vis_hmaps


if __name__ == '__main__':
    train_ds, eval_ds = load_ds()
    for im, label in train_ds:
        im = helpers.denorm_im(im[0])
        print(im.shape)

    # # TEST dataset
    # fddb = FDDB(eval_set=9)
    # train_data, eval_data = fddb.load_ds(train_augm=False, use_path=True)
    # for im, hmap in zip(*train_data):
    #     # test_im, _ = helpers.load_im(im_path, hmap, trans_code)
    #     # test_im = tf.cast((im * 127.5 + 127.5), np.uint8)
    #     im, _ = helpers.load_im_from_path(im, hmap)
    #
    #     hmap_im = helpers.cvt_hmap_to_im(hmap)
    #     vis_im = tf.concat([im, hmap_im], axis=1)
    #     pass


    vis_ims, hmaps = load_vis_data(n=9)

    pass
