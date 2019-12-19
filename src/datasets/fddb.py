import numpy as np
from typing import List
import os
import glob

from src.utils import path_cvt, helpers


class FDDB:
    def __init__(self, eval_set=9, im_shape=(224, 224, 3)):
        """
        Load dataset
        :param eval_set: set id for evaluation, the rest used for training (0 -> 9)
        """
        self.ds_path = path_cvt.get_path_to_FDDB()
        self.eval_set = eval_set
        self.im_shape = im_shape

    def load_ds(self):
        # divide train/eval set
        eval = [self.eval_set]
        train = list(set(np.arange(10)) - set(eval))

        train_ims, train_hmaps = self._read_ann(train)
        eval_ims, eval_hmaps = self._read_ann(eval)

        return (train_ims, train_hmaps), (eval_ims, eval_hmaps)

    def load_by_fold_id(self, fold_id):
        return self._read_ann([fold_id])

    def _read_ann(self, fold_ids: List):
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
        heat_maps = []
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

                # print('{}: {}'.format(im_names, bb_cnt))
                single_hmaps = []

                for i in range(bb_cnt):
                    rect_params = lines.pop(0)
                    c_x, c_y, bb_w, bb_h = [float(x) for x in rect_params.split()]

                    hmap = helpers.point_to_heatmap((c_x, c_y), (bb_w, bb_h), self.im_shape[:2])

                    single_hmaps.append(hmap)

                final_hmap = np.max(single_hmaps, axis=0)

                im_paths.append(im_path)
                heat_maps.append(final_hmap)

        return im_paths, heat_maps