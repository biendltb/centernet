from typing import Callable, Dict
import tensorflow as tf

from src.networks import gauface_dla, gauface_vnet
from src.datasets.face_datasets import load_ds


class GauFace:
    def __init__(self, dataset_fn: Callable = load_ds, network_fn: Callable = gauface_dla.dla_lite_net,
                 lr: float = 1e-4,
                 dataset_args: Dict = None):

        # load dataset
        if dataset_args is None:
            dataset_args = {}
        self.train_ds, self.eval_ds = dataset_fn(**dataset_args)

        self.model = network_fn()

        self.optimizer = tf.keras.optimizers.Adam(lr=lr)

    def forward_pass(self, im, heat_map):
        out_map = self.model(im)

        # filter the heatmap in the range of 2.5 sigma
        in_mask = tf.greater_equal(heat_map, tf.exp(-25.0/4))
        ex_mask = tf.math.logical_not(in_mask)

        diff_tensor = tf.abs(out_map - heat_map)

        in_loss = tf.reduce_mean(tf.boolean_mask(diff_tensor, in_mask))
        ex_loss = tf.reduce_mean(tf.boolean_mask(diff_tensor, ex_mask))

        # use L2 loss
        # loss = tf.reduce_mean(tf.abs(out_map - heat_map))

        return in_loss + ex_loss
