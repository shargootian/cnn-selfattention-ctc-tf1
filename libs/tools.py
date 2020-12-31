# -*- coding: utf-8 -*-
# @Time    : 20-12-14 下午12:06
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : tools.py

import math
import numpy as np
import tensorflow as tf

def convolutional(input_data, filters_shape, trainable, name, strides=(1,1,1,1), bn=True, act=True):
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.contrib.layers.xavier_initializer())
        padding = "SAME"
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 training=trainable)
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


def self_attention(from_tensor, to_tensor, num_attention_heads, size_per_head, seq_length, attention_mask, name):
    with tf.variable_scope(name):
        query = tf.layers.dense(from_tensor,  num_attention_heads * size_per_head,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), name="query")
        key = tf.layers.dense(to_tensor, num_attention_heads * size_per_head,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="key")
        value = tf.layers.dense(to_tensor, num_attention_heads * size_per_head,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), name="value")

        query = tf.transpose(tf.reshape(query, [-1, seq_length, num_attention_heads, size_per_head]), [0, 2, 1, 3])     #真正的多头q矩阵是在这里被创造出来的
        key = tf.transpose(tf.reshape(key, [-1, seq_length, num_attention_heads, size_per_head]), [0, 2, 1, 3])         #真正的多头k矩阵是在这里被创造出来的
        value = tf.transpose(tf.reshape(value, [-1, seq_length, num_attention_heads, size_per_head]), [0, 2, 1, 3])     #真正的多头v矩阵是在这里被创造出来的

        attention_scores = tf.matmul(query, key, transpose_b=True)                                                      #将k矩阵放倒,做到可以相乘
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))                         #结果要排除序列长度的影响

        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        attention_scores = attention_scores + (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0                    #attention的结果加入mask
        attention_probs = tf.nn.softmax(attention_scores)

        context = tf.matmul(attention_probs, value)
        context = tf.reshape(context, [-1, seq_length, num_attention_heads * size_per_head])

        return  context


def layer_normal(input_data, name):
    with tf.variable_scope(name):
        input_data = tf.contrib.layers.layer_norm(inputs=input_data, begin_norm_axis=-1, begin_params_axis=-1)

    return input_data



def gelu(x, name):
    with tf.variable_scope(name):
        cdf = 0.5 * (1.0 + tf.tanh((tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        x = x * cdf

    return x

