import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, strides=(1, 1, 1, 1), pad="SAME", bn=True, act=True):
    with tf.variable_scope(name):
        strides = strides
        padding = pad
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if act:
            conv = tf.nn.leaky_relu(conv)

    return conv




def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv0')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv1')

        residual_output = input_data + short_cut

    return residual_output


def global_average_pooling(input_data, h, w, name):
    with tf.variable_scope(name):
        input_data = tf.nn.avg_pool(input_data,strides=[1,h,w,1],ksize=[1,h,w,1],padding="VALID", name='ave_pooling')

    return input_data

