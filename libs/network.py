from libs import loss
from libs import tools
import tensorflow as tf
from libs.dataset import Dataset


class Network(Dataset):
    def __init__(self, args, C):
        super(Network, self).__init__(args)
        with tf.name_scope('define_input'):
            self.input_images = tf.placeholder(tf.float32, shape=[None, C.INPUT_HEIGHT, None, 3], name='input_images')            # input_images
            self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')                                                 # 输入序列长度，也就是图片下采样后的宽
            self.label = tf.sparse_placeholder(tf.int32, name='label')

        self.input_images = Input(shape=(C.INPUT_HEIGHT, None, 1), name='the_input')
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')