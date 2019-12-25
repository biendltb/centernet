""" Image transformation for data augmentation
    The generator generates transformation on images based on the location of bounding boxes
    Order of transformation:
    + Horizontal flip: boolean
    + Crop: x1, y1, w, h
    + Shift: shift_x, shift_y
"""

import numpy as np
import tensorflow as tf


class TransformerGenerator:
    def __init__(self, seed=17):
        np.random.seed(seed=seed)

        # max scale for an axis is 1/3
        self.scale_limit = 0.5

    def transformation_gen(self, bbox_list):
        """ Generate list of transformation with the awareness of bounding boxes location in the image
            Get the image shape with a list of bounding boxes in the image
            bbox_list: [c_x, c_y, w, h], ... in the ratio to the image size

            Transformation logic:
            #1: get random choice to determin if flipping the image
            #2: randomly choose crop or pad with random magnitudes in all the directions

            Length of transoformation code is 5:
            + 1st: flip 0 or 1
            + 2nd - 5th: crop or pad magnitude
        """
        transformation_code = []

        bbox_list = np.array(bbox_list)

        assert bbox_list.shape[1] == 4, 'ERROR: Bounding box params is not in the right format (4 params)'

        # horizontal flip
        flip_code = int(round(np.random.rand()))
        transformation_code.append(flip_code)
        bbox_list = self._bbox_hflip(bbox_list, flip_code)

        # crop or expand
        spaces = self._get_avail_spaces(bbox_list)

        for s_max in spaces:
            crop_expand_code = int(round(np.random.rand()))
            # 0 - crop; 1 - pad
            # crop - negative; pad - positive
            _mag = np.random.rand() * s_max
            if crop_expand_code == 0:
                transformation_code.append(-_mag)
            else:
                transformation_code.append(_mag)

        bbox_list = self._bblist_crop_pad(transformation_code, bbox_list)

        return transformation_code, bbox_list
    
    def get_no_transform_code(self):
        return np.zeros(5)

    def _bbox_hflip(self, bbox_list, flip_code):
        """ Flip the bounding box list horizontally
            Formula: 1 - x
        """
        _bbox_list = bbox_list.copy()
        if flip_code == 1:
            _bbox_list[:, 0] = 1 - _bbox_list[:, 0]

        return _bbox_list

    def _get_avail_spaces(self, bbox_list):
        """ Get available space in all directions
        """
        # get min and max of all the bounding boxes
        _x1 = np.max([0, np.min(bbox_list[:, 0] - bbox_list[:, 2] / 2)])
        _y1 = np.max([0, np.min(bbox_list[:, 1] - bbox_list[:, 3] / 2)])
        _x2 = np.min([1, np.max(bbox_list[:, 0] + bbox_list[:, 2] / 2)])
        _y2 = np.min([1, np.max(bbox_list[:, 1] + bbox_list[:, 3] / 2)])

        return _x1, _y1, 1-_x2, 1-_y2

    def _bblist_crop_pad(self, transformation_code, bbox_list):
        _bbox_list = bbox_list.copy()

        _x1_mag, _y1_mag, _x2_mag, _y2_mag = transformation_code[-4:]

        new_w = 1 + _x1_mag + _x2_mag
        new_h = 1 + _y1_mag + _y2_mag
        # update c_x
        _bbox_list[:, 0] = (_bbox_list[:, 0] + _x1_mag) / new_w
        # update c_y
        _bbox_list[:, 1] = (_bbox_list[:, 1] + _y1_mag) / new_h
        # update bb_w
        _bbox_list[:, 2] = _bbox_list[:, 2] / new_w
        # update bb_h
        _bbox_list[:, 3] = _bbox_list[:, 3] / new_h

        return _bbox_list


# apply transoformation codes
def tf_apply_trans_codes(im, trans_code):
    ret_im = im
    h, w = im.shape[:2]

    is_flip = trans_code[0]
    # _x1_mag, _y1_mag, _x2_mag, _y2_mag = trans_code[-4:]
    _x1_mag, _y1_mag, _x2_mag, _y2_mag = trans_code[1], trans_code[2], trans_code[3], trans_code[4]

    if is_flip == 1:
        ret_im = tf.image.flip_left_right(ret_im)

    _x1_px = tf.abs(tf.cast(tf.round(_x1_mag * w), tf.int32))
    _x2_px = tf.abs(tf.cast(tf.round(_x2_mag * w), tf.int32))
    _y1_px = tf.abs(tf.cast(tf.round(_y1_mag * h), tf.int32))
    _y2_px = tf.abs(tf.cast(tf.round(_y2_mag * h), tf.int32))

    # crop
    if _x1_mag < 0:
        ret_im = ret_im[:, _x1_px:, :]

    if _x2_mag < 0:
        w = ret_im.shape[1]
        ret_im = ret_im[:, :w-_x2_px, :]

    if _y1_mag < 0:
        ret_im = ret_im[_y1_px:, :, :]

    if _y2_mag < 0:
        h = ret_im.shape[0]
        ret_im = ret_im[:h-_y2_px, :, :]

    # pad: use symmetric padding
    if _x1_mag > 0 and _x1_px > 0:
        _pad_patch = ret_im[:, :_x1_px, :]
        _pad_patch = tf.image.flip_left_right(_pad_patch)
        ret_im = tf.concat([_pad_patch, ret_im], axis=1)

    if _x2_mag > 0 and _x2_px > 0:
        w = ret_im.shape[1]
        _pad_patch = ret_im[:, w-_x2_px:, :]
        _pad_patch = tf.image.flip_left_right(_pad_patch)
        ret_im = tf.concat([ret_im, _pad_patch], axis=1)

    if _y1_mag > 0 and _y1_px > 0:
        _pad_patch = ret_im[:_y1_px, :, :]
        _pad_patch = tf.image.flip_up_down(_pad_patch)
        ret_im = tf.concat([_pad_patch, ret_im], axis=0)

    if _y2_mag > 0 and _y2_px > 0:
        h = ret_im.shape[0]
        _pad_patch = ret_im[h-_y2_px:, :, :]
        _pad_patch = tf.image.flip_up_down(_pad_patch)
        ret_im = tf.concat([ret_im, _pad_patch], axis=0)

    # ret_im = tf.cast(ret_im, tf.uint8)

    return ret_im









