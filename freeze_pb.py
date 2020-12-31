# -*- coding: utf-8 -*-
# @Time    : 20-12-30 下午1:38
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : freeze_pb.py


import tensorflow as tf
from cfg.args import parse_args

def freeze_graph():
    output_node_names = 'CTCBeamSearchDecoder'
    # output_node_names = 'rnn/transpose_time_major'
    saver = tf.train.import_meta_graph(args.demo_weight + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, args.demo_weight)                                                           # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(                                # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,                                                             # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))                                             # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(args.pb, "wb") as f:                                                        # 保存模型
            f.write(output_graph_def.SerializeToString())                                               # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))                                # 得到当前图有几个操作节点



if __name__ == '__main__':
    args = parse_args()
    freeze_graph()
