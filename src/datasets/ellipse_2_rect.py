""" Convert ellipse annotation to rectangle bounding box
"""
import glob
import os
import numpy as np
from PIL import Image


from src.utils import path_cvt


def ellipse_2_rect():
    """ Load all images paths and annotations
    The corresponding annotations are included in the file
    "FDDB-fold-xx-ellipseList.txt" in the following
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
    <major_axis_radius minor_axis_radius angle center_x center_y 1>.

    """
    ds_path = path_cvt.get_path_to_FDDB()
    e_ann_path_template = 'FDDB-folds/FDDB-fold-*-ellipseList.txt'

    ann_path_list = glob.glob(os.path.join(ds_path, e_ann_path_template))

    for ann_path in ann_path_list:
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]

        rect_ann_path = ann_path.replace('ellipse', 'rectangle')

        with open(rect_ann_path, 'w') as f:
            while len(lines) > 0:
                im_name = lines.pop(0)
                bb_cnt = int(lines.pop(0))

                f.write('{}\n{}\n'.format(im_name, bb_cnt))

                # get image width, height
                im_path = '{}/{}/{}.*'.format(ds_path, 'originalPics', im_name)
                # get full path with extension
                im_path = glob.glob(im_path)[0]
                im_shape = np.array(Image.open(im_path)).shape

                # print('{}: {}'.format(im_name, bb_cnt))

                for i in range(bb_cnt):
                    ellipse_params = lines.pop(0)
                    ellipse_params = [float(x) for x in ellipse_params.split()]

                    (c_x, c_y), (bb_w, bb_h) = ellipse_to_rect(ellipse_params, im_shape)
                    f.write('{:.6} {:.6} {:.6} {:.6}\n'.format(c_x, c_y, bb_w, bb_h))

                    pass


def ellipse_to_rect(ellipse_params, im_shape):
    h, w = im_shape[:2]

    cos = np.cos(np.radians(ellipse_params[2]))
    bb_h = ellipse_params[0] * cos * 2 / h
    bb_w = ellipse_params[1] * cos * 2 / w
    c_x = ellipse_params[3] / w
    c_y = ellipse_params[4] / h

    return (c_x, c_y), (bb_w, bb_h)


if __name__ == '__main__':
    # ellipse_2_rect()
    pass
