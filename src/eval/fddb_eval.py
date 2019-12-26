""" Export the annotations in FDDB evaluation format
"""
import tensorflow as tf
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


from src.datasets.fddb import FDDB
from src.networks import gauface_dla
from src.utils import helpers


def run_evaluation_export(eval_fold, ckpt_path, export_path):
    # load model
    model = gauface_dla.dla_lite_net()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path)

    fddb = FDDB()
    im_paths = fddb.load_im_paths(fold_id=eval_fold)

    # writer = open(export_path, 'w')

    for path in im_paths:
        im, original_im = helpers.im_preprocess(path)
        im = im[tf.newaxis, :, :, :]
        original_im = np.array(original_im)

        h, w = original_im.shape[:2]

        output = model.predict(im)
        hmap = output[0, :, :, 0]

        boxes = helpers.heatmap_to_boxes(hmap)

        # # WRITE TO FILE
        # im_name = path[path.find('200'):-4]
        # writer.write('{}\n'.format(im_name))
        # writer.write('{}\n'.format(len(boxes)))
        # for bbox in boxes:
        #     c_x, c_y, bb_w, bb_h, score_x, score_y, prob = bbox
        #     c_x, c_y, bb_w, bb_h = c_x * w, c_y * h, bb_w * w, bb_h * h
        #
        #     left_x = max(0, int(round(c_x - bb_w / 2)))
        #     top_y = max(0, int(round(c_y - bb_h / 2)))
        #     bb_w = int(round(bb_w))
        #     bb_h = int(round(bb_h))
        #
        #     writer.write('{} {} {} {} {:.6f}\n'.format(left_x, top_y, bb_w, bb_h, prob))

        # VISUALISATION
        vis_im = helpers.draw_bb_on_im(hmap, original_im)
        cv2.imshow('test', vis_im[:, :, ::-1])

        k = cv2.waitKey(10000)
        if k == 27:  # press ESC key to exit
            continue


def save_model(ckpt_path, save_path):
    # load model
    model = gauface_dla.dla_lite_net()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path)

    model.save(save_path, save_format='tf')


def webcam_eval(ckpt_path):
    # load model
    model = gauface_dla.dla_lite_net()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path)

    # open webcam
    stream = cv2.VideoCapture(1)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # # setup the writer
    # save_path = '_gauface_record.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = 20
    # _res = (1280, 720)
    # video_writer = cv2.VideoWriter(save_path, fourcc, fps, _res)

    exe_time = []

    while stream.isOpened():
        success, frame = stream.read()

        if success:
            im = tf.image.resize(frame[:, :, ::-1], (224, 224))
            im = (im - 127.5) / 127.5
            im = im[tf.newaxis, :, :, :]

            start = time.time()
            output = model.predict(im)
            hmap = output[0, :, :, 0]
            exe_time.append(time.time() - start)

            # boxes = helpers.heatmap_to_boxes(hmap)

            vis_im = helpers.draw_bb_on_im(hmap, frame)

            # visualization
            vis_hm = helpers.cvt_hmap_to_im(hmap)
            vis_hm = cv2.resize(vis_hm[:, :, ::-1], (int(vis_im.shape[1]/5), int(vis_im.shape[0]/5)))
            vis_im[:vis_hm.shape[0], :vis_hm.shape[1], :] = vis_hm
            cv2.imshow('test', vis_im)
            # video_writer.write(vis_im)

            k = cv2.waitKey(1)
            if k == 27:  # press ESC key to exit
                break

    print('Avg. processing time: {:.2f}ms | FPS: {:2.2f}'.format(np.mean(exe_time) * 1000, 1/np.mean(exe_time)))
    # video_writer.release()


if __name__ == '__main__':
    fold_id = 9
    # ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/gauface/model_8/ckpts/ckpt-7'
    # version kernel_7 top models: 12_10, 18_10
    ckpt_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/gauface/model_18/ckpts/ckpt-10'

    # ann_export_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/gauface/model_8/fold-{:02d}-annotatedList.txt'.format(fold_id)
    # run_evaluation_export(fold_id, ckpt_path, ann_export_path)

    # export_path = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/model_gym/gauface/model_12/export'
    # save_model(ckpt_path, export_path)

    webcam_eval(ckpt_path)
