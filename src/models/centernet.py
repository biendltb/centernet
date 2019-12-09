from typing import Callable, Dict
import tensorflow as tf

from src.networks import dla
from src.datasets.thermal_dataset import load_all_ds


class Centernet:
    def __init__(self, dataset_fn: Callable = load_all_ds, network_fn: Callable = dla.dla_lite_net,
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

        # smooth the output heat map
        # median_filter = tf.ones((3, 3, 1, 1)) * 1/9
        # out_map = tf.nn.conv2d(out_map, median_filter, strides=1, padding='SAME')

        # calculate the l2 distance to the ground truth centroid
        gt_pnt = tf.unravel_index(tf.math.argmax(tf.reshape(heat_map, [heat_map.shape[0], -1]), axis=1), tf.cast(tf.shape(heat_map), tf.int64)[1:3])
        out_pnt = tf.unravel_index(tf.math.argmax(tf.reshape(out_map, [out_map.shape[0], -1]), axis=1), tf.cast(tf.shape(out_map), tf.int64)[1:3])
        dist = tf.reduce_mean(tf.math.sqrt(tf.cast(tf.reduce_sum(tf.math.square(out_pnt - gt_pnt), axis=0), tf.float32)))

        return loss, dist
