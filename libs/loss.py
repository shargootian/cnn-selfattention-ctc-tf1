# -*- coding: utf-8 -*-
# @Time    : 20-11-5 下午3:50
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : loss.py

import tensorflow as tf

def sigmoid_cross_entropy_with_logits(label, pred):
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label, name='sigmoid')
    loss = tf.reduce_mean(cross_entropy_loss)

    return loss


def softmax_cross_entropy(pred, label, num, mask):
    # softmax
    # loss = -ylog(p)
    pred = tf.clip_by_value(pred, 1e-6, 1)
    label = tf.one_hot(tf.cast(label, tf.int32), num, axis=-1)
    loss = tf.reduce_sum(label * tf.log(pred), axis=-1)
    loss = -tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)

    return loss


def l2_loss(label, pred, mask):
    #l2_loss
    #np.sqrt(np.sum((label - pred) ** 2))
    loss = tf.sqrt(tf.reduce_sum(((label - pred) * mask) ** 2))

    return loss


def smooth_l1_loss(pred, label, mask):
    # smooth_l1
    # if abs(x) > 1:
    #    y = abs(x) - 0.5
    # else:
    #    y = 0.5*x**2
    diff = tf.cast(tf.abs(pred - label) * mask, "float64")
    less_than_one = tf.cast(tf.less(diff, 1.0), "float64")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(mask, tf.float64)) + 1e-6)

    return tf.cast(loss, "float32")


# def focal_loss_softmax(pred, label, num, mask, alpha=0.5, gamma=2.0):
#     # focal_loss_softmax
#     # loss = -alpha*(1-p)**gamma*log(p)
#     pred = tf.clip_by_value(pred, 1e-6, 1)
#     label = tf.one_hot(tf.cast(label, tf.int32), num, axis=-1)
#     loss = -alpha * tf.pow((1 - pred), gamma) * label * tf.log(pred)
#     loss = tf.reduce_sum(label * tf.log(pred), axis=-1)
#     loss = -tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)
#
#     return loss


def focal_loss_sigmoid(pred, label, num, mask, alpha=0.5, gamma=2.0):
    # focal loss_sigmoid
    # if y = 1:
    #   -(1-p)**gamma*log(p)
    # if y = 0:
    #   -p**gamma*log(1-p)
    # loss = -(1-p)**gamma*log(p) + -p**gamma*log(1-p)
    pred = tf.clip_by_value(pred, 1e-6, 1 - 1e-6)
    one_hot = tf.one_hot(tf.cast(label, tf.int32), num, axis=-1)
    pos_part = -alpha * tf.pow(1 - pred, gamma) * one_hot * tf.log(pred)
    neg_part = -(1 - alpha) * tf.pow(pred, gamma) * (1 - one_hot) * tf.log(1 - pred)
    loss = tf.reduce_sum((pos_part + neg_part) * tf.tile(tf.expand_dims(mask, -1), [1, 1, 1, num])) / (
                tf.reduce_sum(mask) + 1e-6)

    return loss


def dice_loss(pred, label, mask):
    intersection = tf.reduce_sum(pred * label * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(label * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union

    return loss


def ctc_loss(label, pred, seql):
    loss = tf.nn.ctc_loss(label, pred, seql)
    loss = tf.reduce_mean(loss)

    return loss