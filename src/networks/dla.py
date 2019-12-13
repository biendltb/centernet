""" Implementation of DLA basic block
    Reference: PyTorch implementation
    (https://github.com/ucbdrive/dla/blob/d073c1442936023872356714d54207a95e8913c7/dla.py)
"""
import tensorflow as tf


INPUT_SHAPE = (120, 160, 1)
NUM_CLASS = 1


def _conv(x, filters, kernel_size, strides=1):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result(x)


def _dconv(x, filters, kernel_size, strides=1):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result(x)


def _max_pooling(x, pool_size, strides):
    return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)


def _avg_pooling(x, pool_size, strides):
    return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding='same')(x)


class BasicBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.filters = filters

        self._conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1,
                                            padding='same', use_bias=False)

    def call(self, x):
        input_filters = tf.shape(x)[3]
        # if input and the block have different number of filters, use one more conv layer to equalise it
        residual = tf.cond(tf.equal(input_filters, self.filters),
                           lambda: x,
                           # lambda: _conv(x, self.filters, 1, 1))
                           lambda: self._conv(x))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


def _basic_block(x, filters, kernel_size=3, strides=1):
    input_filters = x.shape[3]

    _tmp_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1,
                                                      padding='same', use_bias=False)

    # if input and the block have different number of filters, use one more conv layer to equalise it
    residual = tf.cond(tf.equal(input_filters, filters),
                       lambda: x,
                       lambda: _tmp_conv(x))

    x = _conv(x, filters=filters, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x += residual
    x = tf.keras.layers.ReLU()(x)

    return x


# modified from Stick-To
def _dla_generator(bottom, filters, levels):
    if levels == 1:
        block1 = _basic_block(bottom, filters)  # BasicBlock(filters=filters)(bottom)
        block2 = _basic_block(block1, filters)  # BasicBlock(filters=filters)(block1)
        aggregation = block1 + block2
        aggregation = _conv(aggregation, filters, kernel_size=3)
    else:
        block1 = _dla_generator(bottom, filters, levels-1)
        block2 = _dla_generator(block1, filters, levels-1)
        aggregation = block1 + block2
        aggregation = _conv(aggregation, filters, kernel_size=3)

    return aggregation


def dla_net():
    base_filters = 16
    # channel last; None -> grayscale or color images
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

    x = _conv(inputs, base_filters, 7)
    # x = _conv(x, 16, 3)
    x = _conv(x, base_filters * 2, 3)
    # x = _conv(x, 32, 3, strides=2)

    # stage 3
    dla_stage3 = _dla_generator(x, base_filters * 4, levels=1)
    # dla_stage3 = _max_pooling(dla_stage3, 2, 2)

    # stage 4
    dla_stage4 = _dla_generator(dla_stage3, base_filters * 8, levels=2)
    dla_stage4 = _max_pooling(dla_stage4, 2, 2)
    residual = _conv(dla_stage3, base_filters * 8, 1)
    residual = _avg_pooling(residual, 2, 2)
    dla_stage4 += residual

    # stage 5
    dla_stage5 = _dla_generator(dla_stage4, base_filters * 16, levels=2)
    dla_stage5 = _max_pooling(dla_stage5, 2, 2)
    residual = _conv(dla_stage4, base_filters * 16, 1)
    residual = _avg_pooling(residual, 2, 2)
    dla_stage5 += residual

    # stage 6
    dla_stage6 = _dla_generator(dla_stage5, base_filters * 32, levels=1)
    dla_stage6 = _max_pooling(dla_stage6, 2, 2)
    residual = _conv(dla_stage5, base_filters * 32, 1)
    residual = _avg_pooling(residual, 2, 2)
    dla_stage6 += residual

    # upsampling
    dla_stage6 = _conv(dla_stage6, base_filters * 16, 1)
    dla_stage6_5 = _dconv(dla_stage6, base_filters * 16, 4, 2)
    dla_stage6_4 = _dconv(dla_stage6_5, base_filters * 16, 4, 2)
    dla_stage6_3 = _dconv(dla_stage6_4, base_filters * 16, 4, 2)

    dla_stage5 = _conv(dla_stage5, base_filters * 16, 1)
    dla_stage5_4 = _conv(dla_stage5 + dla_stage6_5, 256, 3)
    dla_stage5_4 = _dconv(dla_stage5_4, base_filters * 16, 4, 2)
    dla_stage5_3 = _dconv(dla_stage5_4, base_filters * 16, 4, 2)

    dla_stage4 = _conv(dla_stage4, base_filters * 16, 1)
    dla_stage4_3 = _conv(dla_stage4+dla_stage5_4, base_filters * 16, 3)
    dla_stage4_3 = _dconv(dla_stage4_3, base_filters * 16, 4, 2)

    features = _conv(dla_stage6_3 + dla_stage5_3 + dla_stage4_3, base_filters * 16, 3)
    features = _conv(features, base_filters * 16, 1)

    # separate to multiple output heads
    keypoints = _conv(features, NUM_CLASS, 3)
    # size = _conv(features, 2, 3, 1)
    
    model = tf.keras.Model(inputs=inputs, outputs=keypoints)

    return model


def heatmap_to_point(heatmaps_tensor, batch_size=1):
    """ Convert the heat map to point and bounding box in Tensorflow
        Input tensor shape: batch_size * h * w * channel
    """
    
    gaussian_kernel = tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=tf.float32) / 16.0

    filters = gaussian_kernel[:, :, tf.newaxis, tf.newaxis]

    original_tensor = heatmaps_tensor

    heatmaps_tensor = tf.nn.conv2d(heatmaps_tensor, filters, strides=1, padding="SAME")

    h, w = heatmaps_tensor.shape[1:3]

    max_x = tf.math.argmax(tf.math.reduce_sum(heatmaps_tensor, axis=1), axis=1, output_type=tf.int32)[:, 0]
    max_y = tf.math.argmax(tf.math.reduce_sum(heatmaps_tensor, axis=2), axis=1, output_type=tf.int32)[:, 0]

    probs = tf.gather_nd(original_tensor, tf.stack([tf.range(batch_size), max_y, max_x, tf.zeros_like(max_y)], axis=-1))
    # probs = tf.stack(
    #     [tf.reduce_max(tf.slice(original_tensor, [i, max_y[i] - 2, max_x[i] - 2, 0], [1, 5, 5, 1])) for i in
    #      range(batch_size)])
    probs = tf.clip_by_value(probs, clip_value_min=0, clip_value_max=0.99999)

    pos_diff_h = tf.cast(
        tf.math.square(
            (tf.tile(tf.expand_dims(tf.range(h), axis=0), [batch_size, 1]) - tf.tile(tf.expand_dims(max_y, -1),
                                                                                     [1, h])) / (h - 1)
        ),
        tf.float32
    )
    bb_h = tf.reduce_mean(tf.sqrt(
        abs((pos_diff_h / (2.0 * tf.math.log(tf.stack([heatmaps_tensor[i, :, max_x[i], 0] for i in range(batch_size)])))))),
        axis=1) * 2 * h

    pos_diff_w = tf.cast(
        tf.math.square(
            (tf.tile(tf.expand_dims(tf.range(w), axis=0), [batch_size, 1]) - tf.tile(tf.expand_dims(max_x, -1),
                                                                                     [1, w])) / (w - 1)
        ),
        tf.float32
    )
    bb_w = tf.reduce_mean(tf.sqrt(
        abs((pos_diff_w / (2.0 * tf.math.log(tf.stack([heatmaps_tensor[i, max_y[i], :, 0] for i in range(batch_size)])))))),
        axis=1) * 2 * w

    # set width is the minimum of width and height
    bb_w = tf.math.minimum(bb_h, bb_w)

    out = tf.stack([tf.cast(max_y, tf.float32), tf.cast(max_x, tf.float32), bb_h, bb_w, probs], axis=-1)

    return out


def dla_lite_net(mode='train'):
    base_filters = 8
    # channel last; None -> grayscale or color images
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name='thermal_frame')

    x = _conv(inputs, base_filters, 7)
    stage1 = _conv(x, base_filters * 2, 3)
    stage2 = _conv(stage1, base_filters * 2, 3, strides=2)  # 1/2

    # stage 3
    dla_stage3 = _dla_generator(stage2, base_filters * 4, levels=1)
    dla_stage3 = _max_pooling(dla_stage3, 2, 2)  # 1/4

    # stage 4
    dla_stage4 = _dla_generator(dla_stage3, base_filters * 8, levels=2)
    dla_stage4 = _max_pooling(dla_stage4, 2, 2)  # 1/8
    residual = _conv(dla_stage3, base_filters * 8, 1)
    residual = _avg_pooling(residual, 2, 2)  # 1/8
    dla_stage4 += residual

    dla_stage4 = _conv(dla_stage4, base_filters * 16, 1)
    dla_stage4_3 = _dconv(dla_stage4, base_filters * 8, 4, 2)  # 1/4

    dla_stage3 = _conv(dla_stage3, base_filters * 8, 1)
    dla_stage3_3 = _conv(dla_stage3 + dla_stage4_3, base_filters * 8, 3)
    dla_stage3_3 = _dconv(dla_stage3_3, base_filters * 4, 4, 2)  # 1/2

    stage2 = _conv(stage2, base_filters * 4, 1)
    stage2 = _conv(stage2 + dla_stage3_3, base_filters * 4, 1)
    stage2 = _dconv(stage2, base_filters * 2, 4, 2)

    stage1 = _conv(stage1, base_filters * 2, 1)
    stage1 = _conv(stage1 + stage2, base_filters * 2, 1)

    features = _conv(stage1, base_filters * 1, 1)

    # separate to multiple output heads
    keypoints = _conv(features, NUM_CLASS, 3)
    # size = _conv(features, 2, 3, 1)

    if mode == 'train':
        model = tf.keras.Model(inputs=inputs, outputs=keypoints)
    else:
        out = tf.keras.layers.Lambda(lambda hmap: heatmap_to_point(hmap), name='thermal_output')(keypoints)
        model = tf.keras.Model(inputs=inputs, outputs=out)

    return model













