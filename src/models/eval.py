import tensorflow as tf
import h5py
import numpy as np
import cv2
import time

from src.networks import dla
from src.utils import helpers


class ThermalEval:
    def __init__(self):
        self.model = dla.dla_lite_net(mode='eval')

    def load_ckpt(self, ckpt_path):
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(ckpt_path)

    def infer_frame(self, thermal_frame):
        outs = self.model.predict(thermal_frame, batch_size=1)[0]

        # _map = outs[0, :, :, 0]
        # outs = helpers.heatmap_to_point(_map)

        max_y, max_x, bb_h, bb_w, prob = outs
        max_y, max_x = int(max_y), int(max_x)

        print(prob)

        return (max_y, max_x), (bb_h, bb_w), prob

    def infer_video(self, thermal_path):
        with h5py.File(thermal_path) as f:
            # get number of databases - frame count database
            n_thermal_frames = len(f.keys()) - 1

            exec_time = []

            # grab thermal and visual frames one-by-one for processing
            for i in range(n_thermal_frames):
                key = 'frame{}'.format(i)
                thermal_mat = f[key].value
                thermal_frame = helpers.thermal_preprocess(thermal_mat)
                thermal_gray = (thermal_frame * 255).astype(np.uint8)

                thermal_frame = np.expand_dims(thermal_frame, axis=-1)
                thermal_frame = np.expand_dims(thermal_frame, axis=0)

                start = time.time()
                key_point, bb_size, prob = self.infer_frame(thermal_frame)
                exec_time.append(time.time()-start)

                vis_frame = cv2.cvtColor(thermal_gray, cv2.COLOR_GRAY2BGR)

                if prob > 0.1:
                    vis_frame[key_point] = (0, 255, 0)

                    # draw bounding boxes
                    bb_h, bb_w = bb_size
                    kp_y, kp_x = key_point

                    x1, y1 = int(round(kp_x - bb_w / 2)), int(round(kp_y - bb_h / 2))
                    x2, y2 = int(round(x1 + bb_w)), int(round(y1 + bb_h))
                    vis_frame = cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255))

                scale = 4
                vis_frame = cv2.resize(vis_frame, (vis_frame.shape[1] * scale, vis_frame.shape[0] * scale))

                cv2.imshow('test', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 5 ms
                    break
            print('Average execution time: {:.2f}ms | {:.2f} fps'.format(np.mean(exec_time) * 1000, 1/np.mean(exec_time)))

    def save_model(self, save_path):
        self.model.save(save_path, save_format='tf')


if __name__ == '__main__':
    thermal_video = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/HD/thermal/anhdnt/set1_0.hdf5'
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_1/ckpts/ckpt-9'
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_3/ckpts/ckpt-24'
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_6/ckpts/ckpt-8'
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_7/ckpts/ckpt-20'
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_8/ckpts/ckpt-20'
    ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_9/ckpts/ckpt-12'

    thermal_model = ThermalEval()
    thermal_model.load_ckpt(ckpt_path)

    thermal_model.infer_video(thermal_video)

    # save_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/centernet/model_9/export/'
    # thermal_model.save_model(save_path)