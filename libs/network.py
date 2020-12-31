# -*- coding: utf-8 -*-
# @Time    : 20-12-14 上午10:23
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : network.py


from libs import loss
from libs import tools
import tensorflow as tf
from libs.dataset import Dataset


class Network(Dataset):
    def __init__(self, args, C):
        super(Network, self).__init__(args)
        self.input_images = tf.placeholder(tf.float32, shape=[None, C.IMAGE_SIZE_H, C.IMAGE_SIZE_W, 3], name='input_images')
        self.input_masks = tf.placeholder(tf.int32, shape=[None, C.IMAGE_SIZE_W // C.DOWN_SAMPLE], name='input_masks')
        self.input_gts = tf.sparse_placeholder(tf.int32, name='input_gts')
        self.seq_len = tf.placeholder(tf.int32, [None], name='input_seq_length')
        self.trainable = tf.placeholder(dtype=tf.bool, name='training')

    def _embedding(self, C):
        with tf.name_scope('embeddings'):
            with tf.variable_scope('image_embeddings'):
                input_data = tools.convolutional(self.input_images, filters_shape=(3, 3, 3, 32), strides=(1, 2, 2, 1),
                                                 trainable=self.trainable, name='conv0')
                input_data = tools.convolutional(input_data, filters_shape=(3, 3, 32, 64), strides=(1, 2, 1, 1),
                                                 trainable=self.trainable, name='conv1')
                for i in range(3):
                    input_data = tools.residual_block(input_data, 64, 32, 64, trainable=self.trainable,
                                                      name='residual%d' % (i))

                input_data = tools.convolutional(input_data, filters_shape=(3, 3, 64, 128), strides=(1, 2, 2, 1),
                                                 trainable=self.trainable, name='conv2')
                for i in range(4):
                    input_data = tools.residual_block(input_data, 128, 64, 128, trainable=self.trainable,
                                                      name='residual%d' % (i + 3))

                input_data = tools.convolutional(input_data, filters_shape=(3, 3, 128, 256), strides=(1, 2, 1, 1),
                                                 trainable=self.trainable, name='conv3')
                for i in range(5):
                    input_data = tools.residual_block(input_data, 256, 128, 256, trainable=self.trainable,
                                                      name='residual%d' % (i + 7))

                input_data = tools.convolutional(input_data, filters_shape=(2, 3, 256, 512), strides=(1, 2, 1, 1),
                                                 trainable=self.trainable, name='conv4')
                for i in range(3):
                    input_data = tools.residual_block(input_data, 512, 256, 512, trainable=self.trainable,
                                                      name='residual%d' % (i + 12))

                input_data = tools.convolutional(input_data, filters_shape=(1, 3, 512, 768), strides=(1, 1, 1, 1),
                                                 trainable=self.trainable, name='conv5')

                cnn_data = tf.squeeze(input_data, axis=1)

            with tf.variable_scope('position_embeddings'):
                cnn_shape = cnn_data.get_shape()
                seq_length = cnn_shape[1]
                feture_deep = cnn_shape[2]
                full_position_embeddings = tf.get_variable(name='position', dtype=tf.float32, trainable=True,
                                                        shape=[C.IMAGE_SIZE_W, feture_deep],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                position_embeddings = tf.expand_dims(tf.slice(full_position_embeddings, [0, 0], [seq_length, -1]), 0)

            output = cnn_data + position_embeddings
            output = tools.layer_normal(output, 'embedding_layer_normal')

        return output


    def _encoding(self, C, embedding_feature):
        with tf.name_scope('encoding'):
            with tf.variable_scope('attention_mask'):
                mask_shape = tf.shape(self.input_masks)
                batch_size = mask_shape[0]                                                   #这里非常关键 因为这里batch_size是个动态值,要用.shape来获取非.get_shape
                seq_length = mask_shape[1]
                input_mask = tf.cast(tf.reshape(self.input_masks, [batch_size, 1, seq_length]), tf.float32)
                broadcast_ones = tf.ones(shape=[batch_size, seq_length, 1], dtype=tf.float32)
                attention_mask = broadcast_ones * input_mask

            with tf.variable_scope('attention_encoding'):
                attention_head_size = int(C.HIDDEN_SIZE / C.NUM_ATTENTION_HEADS)
                embedding_shape = embedding_feature.get_shape()
                seq_length = embedding_shape[1]
                input_width = embedding_shape[2]

                tensor = embedding_feature
                for idx in range(C.NUM_HIDDEN_LAYERS):
                    attention_head_layer = tools.self_attention(from_tensor = tensor,
                                                                to_tensor = tensor,
                                                                num_attention_heads = C.NUM_ATTENTION_HEADS,
                                                                size_per_head = attention_head_size,
                                                                seq_length = seq_length,
                                                                attention_mask = attention_mask,
                                                                name = 'attention_head_layer_' + str(idx))

                    attention_head_layer = tf.layers.dense(attention_head_layer, input_width,
                                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                    attention_output = tools.layer_normal(attention_head_layer + tensor, 'attention_output_' + str(idx))

                    intermediate_dense = tf.layers.dense(attention_output, C.INTER_DENSE_SIZE,
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                    intermediate_output = tools.gelu(intermediate_dense, 'gelu_' + str(idx))[0]

                    layer_output = tf.layers.dense(intermediate_output, input_width,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
                    layer_output = tools.layer_normal(layer_output + attention_output, 'layer_output_' + str(idx))

                    tensor = layer_output

            tensor = tf.layers.dense(tensor, C.OUT_FEATURE, kernel_initializer=tf.contrib.layers.xavier_initializer())
            tensor = tf.transpose(tensor, (1, 0, 2))

        return tensor


    def _loss_function(self, encoding_feature):
        with tf.variable_scope('diss_loss_layer'):
            total_loss = loss.ctc_loss(self.input_gts, encoding_feature, self.seq_len)
            tf.summary.scalar("loss", total_loss)

        return total_loss


    def _learning_rate(self, C):
        with tf.variable_scope('learning_rate'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=C.LEARNING_RATE,
                                                       global_step=global_step,
                                                       decay_steps=C.STEPS_PER_DECAY,
                                                       decay_rate=C.DECAY_FACTORY,
                                                       staircase=True,
                                                       )
            tf.summary.scalar("learn_rate", learning_rate)

            return global_step, learning_rate


    def _optimizer(self, learning_rate, loss, global_step):
        with tf.variable_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            # train_step = tf.train.AdamOptimizer(learning_rate)

            return train_step














