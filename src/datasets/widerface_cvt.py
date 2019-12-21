""" Parse the WIDERFace to the ratio format
"""
import os
import numpy as np
from PIL import Image


from src.utils import path_cvt


def annotation_cvt(annotation_path, export_path, ims_dir):
    """ Parse the annotation file to the ratio-value format
        The purpose is not to read image content every time labels are generated
        Exported file is store under the export folder in the same folder with the label
    """
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]

    with open(export_path, 'w') as f:
        while len(lines) > 0:
            im_name = lines.pop(0)
            bb_cnt = int(lines.pop(0))

            f.write('{}\n{}\n'.format(im_name, bb_cnt))

            # get image width, height
            im_path = '{}/{}'.format(ims_dir, im_name)
            im_shape = np.array(Image.open(im_path)).shape

            h, w = im_shape[:2]

            # print('{}: {}'.format(im_name, bb_cnt))

            if bb_cnt == 0:
                bb_cnt = 1

            for i in range(bb_cnt):
                rect_params = lines.pop(0)
                rect_params = [int(x) for x in rect_params.split()[:4]]

                i_x1, i_y1, i_bb_w, i_bb_h = rect_params

                c_x, bb_w = (i_x1 + i_bb_w/2)/w, i_bb_w/w
                c_y, bb_h = (i_y1 + i_bb_h/2)/h, i_bb_h/h

                f.write('{:.6} {:.6} {:.6} {:.6}\n'.format(c_x, c_y, bb_w, bb_h))


def parse():
    ds_dir = path_cvt.get_path_to_WIDERFace()

    train_path = os.path.join(ds_dir, 'wider_face_split/wider_face_train_bbx_gt.txt')
    train_export_path = os.path.join(ds_dir, 'wider_face_split/export/wider_face_train_bbx_gt_cvt.txt')
    train_ims_dir = os.path.join(ds_dir, 'WIDER_train/images/')
    annotation_cvt(train_path, train_export_path, train_ims_dir)

    val_path = os.path.join(ds_dir, 'wider_face_split/wider_face_val_bbx_gt.txt')
    val_export_path = os.path.join(ds_dir, 'wider_face_split/export/wider_face_val_bbx_gt_cvt.txt')
    val_ims_dir = os.path.join(ds_dir, 'WIDER_val/images/')
    annotation_cvt(val_path, val_export_path, val_ims_dir)


if __name__ == '__main__':
    # parse()
    pass
