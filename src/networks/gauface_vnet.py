import tensorflow as tf

INPUT_SHAPE = (320, 320, 3)


def _conv(x, filters, kernel_size=7, strides=1):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result(x)


def _conv_no_relu(x, filters, kernel_size=7, strides=1):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    return result(x)


def _dconv(x, filters, kernel_size, strides=1):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result(x)


def resnext_block(x, filters, cardinality=8, strides=4):
    channel_split = tf.keras.layers.Lambda(lambda _x: tf.split(_x, cardinality, axis=-1))

    splits = channel_split(x)

    agg_out = None
    for s in splits:
        _out = _conv(s, int(filters/cardinality), kernel_size=1, strides=strides)
        _out = _conv(_out, int(filters/cardinality), kernel_size=7)
        _out = _conv_no_relu(_out, filters, kernel_size=1)

        if agg_out is None:
            agg_out = _out
        else:
            agg_out += _out
    
    # downsample x
    x = tf.keras.layers.AveragePooling2D(pool_size=strides, strides=strides)(x)
    x = _conv_no_relu(x, filters, kernel_size=7)

    x = tf.keras.layers.ReLU()(x + agg_out)

    return x


def up_block(x, x_down, filters, up_scale=4):
    x_down = _conv_no_relu(x_down, filters, kernel_size=7)

    x = tf.keras.layers.UpSampling2D(up_scale)(x)
    x = _conv_no_relu(x, filters, kernel_size=7)
    # x = _dconv(x, filters, kernel_size=up_scale, strides=up_scale)

    x = tf.keras.layers.ReLU()(x + x_down)

    return x


def resnext_net():
    base_filters = 8
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input')

    x = _conv(inputs, base_filters, 15)

    stage_1 = resnext_block(x, base_filters * 2, strides=4)

    stage_2 = resnext_block(stage_1, base_filters * 4, strides=4)

    stage_3 = resnext_block(stage_2, base_filters * 8, strides=4)

    # stage_4 = resnext_block(stage_3, base_filters * 8, strides=2)
    #
    # stage_5 = resnext_block(stage_4, base_filters * 8, strides=2)

    bottom = _conv(stage_3, base_filters * 8, kernel_size=1)

    # up_5 = up_block(bottom, stage_4, base_filters * 4, up_scale=2)
    #
    # up_4 = up_block(up_5, stage_3, base_filters * 4, up_scale=2)

    up_3 = up_block(bottom, stage_2, base_filters * 4, up_scale=4)

    up_2 = up_block(up_3, stage_1, base_filters * 2, up_scale=4)

    up_1 = up_block(up_2, x, base_filters, up_scale=4)

    features = _conv(up_1, base_filters)

    keypoints = _conv(features, 1, 3)

    model = tf.keras.Model(inputs=inputs, outputs=keypoints)

    return model



