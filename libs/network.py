from libs import loss
from libs import tools
import tensorflow as tf
from libs.dataset import Dataset


class Network(Dataset):
    def __init__(self, args, C):
        super(Network, self).__init__(args)
        with tf.name_scope('define_input'):
            self.input_images = tf.placeholder(tf.float32, shape=[None, C.INPUT_H, 800, 3], name='input_images')            # input images
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

    def _crnn(self, input_data, trainable):
        with tf.name_scope('cnn'):
            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable,
                                             name='conv0', strides=(1, 2, 2, 1))                                                  # h//2, w//2
            for i in range(2):
                input_data = tools.residual_block(input_data, 32, 16, 32, trainable=trainable, name='residual%d' % (i + 0))

            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 32, 64), trainable=trainable,
                                             name='conv3', strides=(1, 2, 2, 1))                                                  # h//4, w//4

            for i in range(3):
                input_data = tools.residual_block(input_data, 64, 32, 64, trainable=trainable, name='residual%d' % (i + 2))

            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 64, 128), trainable=trainable,
                                             name='conv6', strides=(1, 2, 1, 1))                                                  # h//8, w//4

            for i in range(4):
                input_data = tools.residual_block(input_data, 128, 64, 128, trainable=trainable, name='residual%d' % (i + 5))

            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 128, 256), trainable=trainable,
                                             name='conv10', strides=(1, 2, 1, 1))                                                  # h//16, w//4

            for i in range(4):
                input_data = tools.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' % (i + 9))

            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 256, 512), trainable=trainable,
                                             name='conv14', strides=(1, 2, 1, 1))                                                  # h//16, w//4
            for i in range(3):
                input_data = tools.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' % (i + 13))

            input_data = tools.convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable,
                                             name='conv17', strides=(1, 2, 1, 1))                                                  # h//32, w//4

            input_data = tf.squeeze(input_data, [1], name='squeeze')

        with tf.name_scope('rnn'):

            print(1)


