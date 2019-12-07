import os
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm

import tensorflow as tf

from src.models.centernet import Centernet
from src.utils.path_cvt import get_path_to_vis_ims, get_path_to_ckpts
from src.datasets.thermal_dataset import load_vis_data

# -------> For RTX NVIDIA GPU only
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# -------

# wandb for logging
import wandb

# save checkpoint every N epochs
CKPT_INTERVAL = 15


class ModelTrain:
    def __init__(self, epochs=100, batch_size=1, use_wandb=False):
        dataset_args = {
            'batch_size': batch_size
        }
        self.model = Centernet(dataset_args=dataset_args, lr=2e-4)

        self.epochs = epochs
        self.batch_size = batch_size

        # visualisation params
        self.num_examples_to_generate = 9
        self.visualized_eval_images, _ = load_vis_data(n=self.num_examples_to_generate)

        # set up checkpoints
        checkpoint_dir = get_path_to_ckpts()
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer,
                                              model=self.model.model,
                                              )

        self._use_wandb = use_wandb
        if self._use_wandb:
            wandb.init(project='centernet')
            wandb.config.batch_size = batch_size

        print('===== MODEL_SUMMARY =====')
        self.model.model.summary()

        self._start_epoch = 0
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            print('===> Training from scratch')
        else:
            last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
            self.checkpoint.restore(last_ckpt_path)

            ckpt_prefix = os.path.basename(last_ckpt_path)
            self._start_epoch = int(ckpt_prefix[ckpt_prefix.rfind('-')+1:]) * CKPT_INTERVAL + 1
            print('===> Successfully restored checkpoints at {}'.format(last_ckpt_path))

    @tf.function
    def _train_step(self, thermal_mat, heat_map):
        with tf.GradientTape() as tape:
            loss, err = self.model.forward_pass(thermal_mat, heat_map)

        gradients = tape.gradient(loss, self.model.model.trainable_variables)

        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.model.trainable_variables))

        return loss, err

    def _eval_step(self, thermal_mat, heat_map):
        loss, err = self.model.forward_pass(thermal_mat, heat_map)
        return loss, err

    def train(self):
        for epoch in range(self._start_epoch, self.epochs):
            start = time.time()

            losses = []
            errs = []

            for thermal_mat, heat_map in tqdm(self.model.train_ds):
                step_loss, step_err = self._train_step(thermal_mat, heat_map)
                losses.append(step_loss)
                errs.append(step_err)

            # evaluate at the end of each epoch
            eval_losses = []
            eval_errs = []
            for thermal_mat, heat_map in self.model.eval_ds:
                loss, err = self._eval_step(thermal_mat, heat_map)
                eval_losses.append(loss)
                eval_errs.append(err)

            # logging losses at the end of an epoch
            if self._use_wandb:
                wandb.log({
                    'loss': np.mean(losses),
                    'err': np.mean(errs),
                    'eval_loss': np.mean(eval_losses),
                    'eval_err': np.mean(eval_errs)
                })

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generate_and_save_images(
                                     epoch + 1)

            # Save the model every 15 epochs
            if (epoch + 1) % CKPT_INTERVAL == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    def generate_and_save_images(self, epoch):
        """ Visualize the heat map combining with image, preserve the matplotlib color map
            e.g. https://stackoverflow.com/questions/31544130/saving-an-imshow-like-image-while-preserving-resolution
        """
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.model.model.predict(self.visualized_eval_images)

        plt.figure(figsize=(9, 9))

        for i in range(predictions.shape[0]):
            fig = predictions[i, :, :, 0] + self.visualized_eval_images[i, :, :, 0]
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=fig.min(), vmax=fig.max())
            image = cmap(norm(fig))

            plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.axis('off')

        plt.savefig(os.path.join(get_path_to_vis_ims(), 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close()


if __name__ == '__main__':
    trainer = ModelTrain(
        epochs=5000,
        batch_size=2,
        use_wandb=True
    )

    trainer.train()
