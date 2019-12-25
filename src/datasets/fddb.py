import numpy as np
from typing import List
import os
import glob
import cv2

from src.utils import path_cvt, helpers
from src.datasets.image_transformer import TransformerGenerator, tf_apply_trans_codes

IM_SHAPE = (224, 224)

class FDDB:
    def __init__(self, eval_set=9, im_shape=(224, 224, 3)):
        """
        Load dataset
        :param eval_set: set id for evaluation, the rest used for training (0 -> 9)
        """
        self.ds_path = path_cvt.get_path_to_FDDB()
        self.eval_set = eval_set
        self.im_shape = im_shape

        self.trans_gen = TransformerGenerator()

    def load_ds(self, train_augm=True):
        # divide train/eval set
        eval = [self.eval_set]
        train = list(set(np.arange(10)) - set(eval))

        train_ims, train_hmaps = self._read_ann(train, augmentation=train_augm)
        eval_ims, eval_hmaps = self._read_ann(eval, augmentation=False)

        return (train_ims, train_hmaps), (eval_ims, eval_hmaps)

    def load_by_fold_id(self, fold_id, augmentation=False):
        return self._read_ann([fold_id], augmentation=augmentation)

    def load_im_paths(self, fold_id: int):
        """ Load all image paths by fold id
            File: FDDB-fold-xx.txt
        """
        fold_path = 'FDDB-folds/FDDB-fold-{:02d}.txt'.format(fold_id)
        im_path_template = '{}/originalPics/{}.*'
        fold_path = os.path.join(self.ds_path, fold_path)

        im_paths = []

        with open(fold_path) as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]

        while len(lines) > 0:
            im_name = lines.pop(0)
            path = glob.glob(im_path_template.format(self.ds_path, im_name))
            if len(path) != 1:
                print('ERROR: None or more than an image found in {}'.format(path))
                break
            im_paths.append(path[0])

        return im_paths

    def _read_ann(self, fold_ids: List, augmentation):
        """ Load all images paths and annotations
        The corresponding annotations are included in the file
        "FDDB-fold-xx-rectangleList.txt" in the following
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
        ann_path_template = 'FDDB-folds_rect/FDDB-fold-{:02d}-rectangleList.txt'
        im_path_template = '{}/originalPics/{}.*'

        im_paths = []
        ims = []
        heat_maps = []
        trans_codes = []
        for id in fold_ids:
            ann_path = os.path.join(self.ds_path, ann_path_template.format(id + 1))
            with open(ann_path, 'r') as f:
                lines = f.readlines()

            lines = [l.strip() for l in lines]

            while len(lines) > 0:
                im_name = lines.pop(0)

                path = glob.glob(im_path_template.format(self.ds_path, im_name))
                if len(path) != 1:
                    print('ERROR: None or more than an image found in {}'.format(path))
                    break
                im_path = path[0]

                bb_cnt = int(lines.pop(0))

                # collect the bounding box list
                bb_list = []
                for i in range(bb_cnt):
                    rect_params = lines.pop(0)
                    c_x, c_y, bb_w, bb_h = [float(x) for x in rect_params.split()]
                    bb_list.append([c_x, c_y, bb_w, bb_h])

                origin_hmap = self._gen_heat_map(bb_list)

                # im_paths.append(im_path)
                im = helpers.read_im_from_path(im_path)
                im = cv2.resize(np.array(im), (IM_SHAPE[1], IM_SHAPE[0]))
                ims.append(im)
                heat_maps.append(origin_hmap)
                # trans_codes.append(self.trans_gen.get_no_transform_code())

                if augmentation:
                    # transform the image for data augmentation
                    gen_trans_code, new_bb_list = self.trans_gen.transformation_gen(bb_list)
                    trans_hmap = self._gen_heat_map(new_bb_list)

                    # im_paths.append(im_path)
                    trans_im = tf_apply_trans_codes(im, gen_trans_code)
                    trans_im = cv2.resize(np.array(trans_im), (IM_SHAPE[1], IM_SHAPE[0]))
                    ims.append(trans_im)
                    heat_maps.append(trans_hmap)
                    # trans_codes.append(gen_trans_code)

        # return im_paths[:100], heat_maps[:100], trans_codes[:100]
        return ims, heat_maps

    def _gen_heat_map(self, bbox_list):
        single_hmaps = []
        for _bbox in bbox_list:
            c_x, c_y, bb_w, bb_h = _bbox
            hmap = helpers.point_to_heatmap((c_x, c_y), (bb_w, bb_h), self.im_shape[:2])
            single_hmaps.append(hmap)

        mixed_hmap = np.max(single_hmaps, axis=0)

        return mixed_hmap

