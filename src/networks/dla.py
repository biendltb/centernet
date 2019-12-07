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
    return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides)(x)


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


# modified from Stick-To
def _dla_generator(bottom, filters, levels):
    if levels == 1:
        block1 = BasicBlock(filters=filters)(bottom)
        block2 = BasicBlock(filters=filters)(block1)
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













