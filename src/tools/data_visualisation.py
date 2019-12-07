import glob
import h5py
import numpy as np
import cv2

from src.utils import helpers


def thermal2gray(thermal_mat):
    """ Convert the raw data from thermal sensor to grayscale image data for visualization
    """
    thermal_mat = helpers.thermal_preprocess(thermal_mat)
    thermal_mat = (thermal_mat * 255).astype(np.uint8)

    return thermal_mat


def bb_decode(label_str, im_shape):
    """ Decode the bouding box from string to OpenCV corner points
    """
    h, w = im_shape

    c_x, c_y, bb_w, bb_h = [float(s) for s in label_str.split()[1:]]

    c_y, bb_h = c_y * h, bb_h * h
    c_x, bb_w = c_x * w, bb_w * w

    # calculate top-left and bottom-right points
    y1 = int(c_y - bb_h / 2.0)
    y2 = int(c_y + bb_h / 2.0)
    x1 = int(c_x - bb_w / 2.0)
    x2 = int(c_x + bb_w / 2.0)

    return [(x1, y1), (x2, y2)]


def image_export(scale=2):
    """ Export thermal images for visualisation
        Label in dataset: <class_id> <centre_x> <centre_y> <bb_w> <bb_y>
    """
    ds_dir = '/media/biendltb/6e1ef38e-db2f-4eda-ad11-31252df3b87b/data/Datasets/thermal_face/ds/'
    thermal_paths = glob.glob(ds_dir + '/*.hdf5')

    for path in thermal_paths:
        im_key = 'image'
        label_key = 'label'
        with h5py.File(path, 'r') as f:
            thermal_mat = f[im_key].value
            label_str = f[im_key].attrs[label_key]

        gray_im = thermal2gray(thermal_mat)
        org_shape = gray_im.shape
        pnt_1, pnt_2 = bb_decode(label_str, org_shape)

        color_im = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(color_im, pnt_1, pnt_2, (0, 255, 0), 1)

        color_im = cv2.resize(color_im, (org_shape[1] * scale, org_shape[0] * scale))

        cv2.imshow('test', color_im)

        k = cv2.waitKey(100)
        if k == 27:  # press ESC key to exit
            break


if __name__ == '__main__':
    image_export()
