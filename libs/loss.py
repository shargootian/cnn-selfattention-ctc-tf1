import tensorflow as tf

def sigmoid_cross_entropy_with_logits(label, pred):
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label, name='sigmoid')
    loss = tf.reduce_mean(cross_entropy_loss)

    return loss


def l2_loss(label, pred):
    loss = tf.sqrt(tf.reduce_sum((label - pred) ** 2))
    #l2_loss
    #np.sqrt(np.sum((label - pred) ** 2))

    return loss


def smooth_l1_loss(label, pred):
    diff = tf.cast(tf.abs(label - pred), "float64")
    less_than_one = tf.cast(tf.less(diff, 1.0), "float64")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_mean(loss)
    #smooth_l1
    # if abs(x) > 1:
    #    y = abs(x) - 0.5
    # else:
    #    y = 0.5*x**2

    return loss
