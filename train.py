# -*- coding: utf-8 -*-
# @Time    : 20-12-14 上午10:20
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : train.py

import os
import tensorflow as tf
from cfg.args import parse_args
from libs.network import Network
from cfg.config import Config as C


class ocr(Network):
    def __init__(self, args):
        super(ocr, self).__init__(args, C)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = C.GPU_MEMORY
        os.environ["CUDA_VISIBLE_DEVICES"] = C.CUDA_VISIBLE_DEVICES
        self.sess = tf.Session(config=config)

    def train(self):
        embedding = self._embedding(C)
        encoding = self._encoding(C, embedding)
        loss = self._loss_function(encoding)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(encoding, self.seq_len, merge_repeated=False)
        dis = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.input_gts))

        global_step, learning_rate = self._learning_rate(C)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        moving_ave = tf.train.ExponentialMovingAverage(C.MOVING_AVERAGE_DECAY).apply(tf.trainable_variables())

        with tf.control_dependencies(update_ops):
            with tf.control_dependencies([moving_ave]):
                train_op = self._optimizer(learning_rate, loss, global_step)

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        tra_summary_writer = tf.summary.FileWriter(self.args.logs, graph=self.sess.graph)
        try:
            print('=> Restoring weights from: %s ... ' % self.args.initial_weight)
            saver.restore(self.sess, self.args.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.args.initial_weight)

        image_label = self.get_file(self.args.train_data)

        for iter in range(C.ITERATION):
            try:
                images, masks, labels, seql = self.get_data(C, image_label)
                ls, _, = self.sess.run([loss, train_op], feed_dict={self.input_images: images,
                                                                    self.input_masks: masks,
                                                                    self.input_gts: labels,
                                                                    self.seq_len : seql,
                                                                    self.trainable: True})

                if iter % C.ITER_SUMMARY == 0 or (iter + 1) == C.ITERATION:
                    _, lr, summary_str, dis_ = self.sess.run([train_op, learning_rate, summary_op, dis],
                                                       feed_dict={self.input_images: images,
                                                                  self.input_masks: masks,
                                                                  self.input_gts: labels,
                                                                  self.seq_len: seql,
                                                                  self.trainable: True})
                    print('summary_iter: %d, learn_rate: %f, model_loss: %.5f, edit_dis: %.5f' % (iter, lr, ls, dis_))
                    tra_summary_writer.add_summary(summary_str, global_step=iter)

                if iter % C.ITER_SAVE == 0 or (iter + 1) == C.ITERATION:
                    print('save_ckpt, iter: %d, model_loss: %.5f' % (iter, ls))
                    checkpoint_path = os.path.join(self.args.logs, 'model_%.5f.ckpt' % (ls))
                    saver.save(self.sess, checkpoint_path, global_step=iter)

            except Exception as e:
                print(e)

if __name__ == '__main__':
    args = parse_args()
    O = ocr(args)
    O.train()
