from typing import Callable, Dict
import tensorflow as tf

from src.networks import dla
from src.datasets.thermal_dataset import load_all_ds


class Centernet:
    def __init__(self, dataset_fn: Callable = load_all_ds, network_fn: Callable = dla.dla_net,
                 lr: float = 1e-4,
                 dataset_args: Dict = None):

        # load dataset
        if dataset_args is None:
            dataset_args = {}
        self.train_ds, self.eval_ds = dataset_fn(**dataset_args)

        self.model = network_fn()

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def forward_pass(self, thermal_mat, heat_map):
        out_map = self.model(thermal_mat)

        # use L2 loss
        loss = tf.reduce_mean(tf.square(out_map - heat_map))

        # calculate the l2 distance to the ground truth centroid
        gt_pnt = tf.unravel_index(tf.math.argmax(tf.reshape(heat_map, [-1])), tf.cast(tf.shape(heat_map), tf.int64))
        out_pnt = tf.unravel_index(tf.math.argmax(tf.reshape(out_map, [-1])), tf.cast(tf.shape(out_map), tf.int64))
        dist = tf.linalg.norm(tf.cast(out_pnt - gt_pnt, tf.float64))

        return loss, dist
