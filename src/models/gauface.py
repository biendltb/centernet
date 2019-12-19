from typing import Callable, Dict
import tensorflow as tf

from src.networks import gauface_dla
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

        # use L2 loss
        loss = tf.sqrt(tf.reduce_sum(tf.square(out_map - heat_map)))

        return loss