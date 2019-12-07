import tensorflow as tf
import h5py
import numpy as np
import cv2

from src.networks.dla import dla_net
from src.utils import helpers


class ThermalEval:
    def __init__(self):
        self.model = dla_net()

    def load_ckpt(self, ckpt_path):
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(ckpt_path)

    def infer_frame(self, thermal_frame):
        heat_map = self.model.predict(thermal_frame)

        _map = heat_map[0, :, :, 0]

        key_point = helpers.heatmap_to_point(_map)

        return key_point

    def infer_video(self, thermal_path):
        with h5py.File(thermal_path) as f:
            # get number of databases - frame count database
            n_thermal_frames = len(f.keys()) - 1

            # grab thermal and visual frames one-by-one for processing
            for i in range(n_thermal_frames):
                key = 'frame{}'.format(i)
                thermal_mat = f[key].value
                thermal_frame = helpers.thermal_preprocess(thermal_mat)
                thermal_gray = (thermal_frame * 255).astype(np.uint8)

                thermal_frame = np.expand_dims(thermal_frame, axis=-1)
                thermal_frame = np.expand_dims(thermal_frame, axis=0)

                key_point = self.infer_frame(thermal_frame)

                vis_frame = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)
                vis_frame[key_point] = (0, 0, 255)
                scale = 4
                vis_frame = cv2.resize(vis_frame, (vis_frame.shape[1] * scale, vis_frame.shape[0] * scale))

                cv2.imshow('test', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 5 ms
                    break


if __name__ == '__main__':
    thermal_video = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/HD/thermal/anhdnt/set1_0.hdf5'
    ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_1/ckpts/ckpt-9'
    thermal_model = ThermalEval()
    thermal_model.load_ckpt(ckpt_path)

    thermal_model.infer_video(thermal_video)

