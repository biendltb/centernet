import numpy as np
import os
import cv2

from src.utils import path_cvt, helpers
from src.datasets.image_transformer import TransformerGenerator, tf_apply_trans_codes


class WIDER:
    def __init__(self, im_shape=(224, 224, 3)):
        """
        Load dataset
        :param eval_set: set id for evaluation, the rest used for training (0 -> 9)
        """
        # create path to train and val set
        ds_dir = path_cvt.get_path_to_WIDERFace()

        self.train_ann_path = os.path.join(ds_dir, 'wider_face_split/export/wider_face_train_bbx_gt_cvt.txt')
        self.val_ann_path = os.path.join(ds_dir, 'wider_face_split/export/wider_face_val_bbx_gt_cvt.txt')

        self.train_ims_dir = os.path.join(ds_dir, 'WIDER_train/images')
        self.val_ims_dir = os.path.join(ds_dir, 'WIDER_val/images')

        self.im_shape = im_shape

        self.trans_gen = TransformerGenerator()

    def load_ds(self, augmentation=False, use_path=True):
        assert not (augmentation is True and use_path is True), 'ERROR: Cannot use path for image augmentation.'

        train_ims, train_hmaps = self._read_ann(self.train_ann_path, self.train_ims_dir,
                                                augmentation=augmentation, use_path=use_path)
        eval_ims, eval_hmaps = self._read_ann(self.val_ann_path, self.val_ims_dir,
                                              augmentation=False, use_path=use_path)

        print('WIDER FACE DATASET: Train data: {}\nValidation data: {}'.format(len(train_ims), len(eval_ims)))

        return (train_ims, train_hmaps), (eval_ims, eval_hmaps)

    def load_val_ds(self):
        return self._read_ann(self.val_ann_path, self.val_ims_dir)

    def _read_ann(self, ann_path, ims_dir, augmentation, use_path, augmentation_ratio=0.7):
        """ Load all images paths and annotations
        The corresponding annotations are included in the file
        "wider_face_train_bbx_gt_cvt.txt" in the following
        format:

        ...
        <image name i>
        <number of faces in this image =im>
        <face i1>
        <face i2>
        ...
        <face im>
        ...

        Here, each face is denoted by:
        <center_x center_y bb_w bb_h>
        """

        im_paths = []
        heat_maps = []
        # if augmentation
        ims = []

        with open(ann_path, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]

        while len(lines) > 0:
            im_name = lines.pop(0)
            im_path = os.path.join(ims_dir, im_name)

            bb_cnt = int(lines.pop(0))

            if bb_cnt == 0:
                bb_cnt = 1

            # collect the bounding box list
            bb_list = []
            for i in range(bb_cnt):
                rect_params = lines.pop(0)
                c_x, c_y, bb_w, bb_h = [float(x) for x in rect_params.split()]
                bb_list.append([c_x, c_y, bb_w, bb_h])

            origin_hmap = self._gen_heat_map(bb_list)

            im_paths.append(im_path)
            heat_maps.append(origin_hmap)

            if not use_path:
                im = helpers.read_im_from_path(im_path)
                im = cv2.resize(np.array(im), (self.im_shape[1], self.im_shape[0]))
                ims.append(im)

            if augmentation:
                # roll a dice to see whether we augment this image
                if np.random.rand() < augmentation_ratio:
                    # transform the image for data augmentation
                    gen_trans_code, new_bb_list = self.trans_gen.transformation_gen(bb_list)
                    trans_hmap = self._gen_heat_map(new_bb_list)

                    trans_im = tf_apply_trans_codes(im, gen_trans_code)
                    trans_im = cv2.resize(np.array(trans_im), (self.im_shape[1], self.im_shape[0]))

                    ims.append(trans_im)
                    heat_maps.append(trans_hmap)

        if not use_path:
            return ims, heat_maps

        return im_paths, heat_maps

    def _gen_heat_map(self, bbox_list):
        single_hmaps = []
        for _bbox in bbox_list:
            c_x, c_y, bb_w, bb_h = _bbox
            hmap = helpers.point_to_heatmap((c_x, c_y), (bb_w, bb_h), self.im_shape[:2])
            single_hmaps.append(hmap)

        mixed_hmap = np.max(single_hmaps, axis=0)

        return mixed_hmap


if __name__ == '__main__':
    wider = WIDER()
    wider.load_ds()

    # 12 bounding boxes per image
