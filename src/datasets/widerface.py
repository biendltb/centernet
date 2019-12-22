import numpy as np
import os

from src.utils import path_cvt, helpers


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

    def load_ds(self):

        train_ims, train_hmaps = self._read_ann(self.train_ann_path, self.train_ims_dir)
        eval_ims, eval_hmaps = self._read_ann(self.val_ann_path, self.val_ims_dir)

        print('WIDER FACE DATASET: Train data: {}\nValidation data: {}'.format(len(train_ims), len(eval_ims)))

        return (train_ims, train_hmaps), (eval_ims, eval_hmaps)

    def load_val_ds(self):
        return self._read_ann(self.val_ann_path, self.val_ims_dir)

    def _read_ann(self, ann_path, ims_dir):
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

        with open(ann_path, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]

        while len(lines) > 0:
            im_name = lines.pop(0)
            im_path = os.path.join(ims_dir, im_name)

            bb_cnt = int(lines.pop(0))

            # print('{}: {}'.format(im_names, bb_cnt))
            single_hmaps = []

            if bb_cnt == 0:
                bb_cnt = 1

            # train only with images which have <= 5 faces per image
            if bb_cnt > 5:
                for i in range(bb_cnt):
                    lines.pop(0)
                continue

            for i in range(bb_cnt):
                rect_params = lines.pop(0)
                c_x, c_y, bb_w, bb_h = [float(x) for x in rect_params.split()]

                hmap = helpers.point_to_heatmap((c_x, c_y), (bb_w, bb_h), self.im_shape[:2])

                single_hmaps.append(hmap)

            final_hmap = np.max(single_hmaps, axis=0)

            im_paths.append(im_path)
            heat_maps.append(final_hmap)

        return im_paths, heat_maps


if __name__ == '__main__':
    wider = WIDER()
    wider.load_ds()

    # 12 bounding boxes per image
