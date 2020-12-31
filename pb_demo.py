# -*- coding: utf-8 -*-
# @Time    : 20-12-30 下午1:52
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : pb_demo.py

import json
import cv2 as cv
import numpy as np
import tensorflow as tf
from cfg.args import parse_args
from cfg.config import Config as C

args = parse_args()

char = json.load(open(args.char, mode='r'))
char_out = []
for key in char.keys():
    char_out.append(key)

def image_demo(image, return_elements):
    txt_pred = ''
    img = cv.imread(image)
    h, w, _ = img.shape
    w = w * C.IMAGE_SIZE_H // h
    if w < C.IMAGE_SIZE_W:
        img = cv.resize(img, (w, C.IMAGE_SIZE_H))
        tail = np.zeros((C.IMAGE_SIZE_H, C.IMAGE_SIZE_W - w, 3))
        img = np.concatenate((img, tail), axis=1)
        mask_len = w // C.DOWN_SAMPLE
    else:
        img = cv.resize(img, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
        mask_len = C.IMAGE_SIZE_W // C.DOWN_SAMPLE
    img = img / 255.0 - 0.5
    mask_head = np.ones((mask_len), dtype=np.int32)
    mask_tail = np.zeros(C.IMAGE_SIZE_W // C.DOWN_SAMPLE - mask_len, dtype=np.int32)
    mask = np.concatenate((mask_head, mask_tail), axis=0)

    img = img[np.newaxis, :, :, :]
    mask = mask[np.newaxis, :]
    seql = np.ones(1, dtype=np.int32) * (C.IMAGE_SIZE_W // C.DOWN_SAMPLE)

    pred = sess.run([return_elements[3], return_elements[4], return_elements[5]],
                     feed_dict={return_elements[0]: img,
                                return_elements[1]: mask,
                                return_elements[2]: seql,
                                return_elements[3]: False})
    # pred = np.argmax(pred[0], axis=-1)
    # for item in pred[0]:
    #     if item < len(char_out):
    #         txt_pred += char_out[item]
    # return  txt_pred

    pred = pred[2]
    for item in pred:
        txt_pred += char_out[item]
    return txt_pred


if __name__ == '__main__':
    files = []
    with open('/home/ubuntu/workspace/ocr3/self_attention_dataset/resx.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            res = line.split(':')
            files.append([res[0],res[1].strip()])

    pb_file = args.pb
    graph = tf.Graph()
    # return_elements = ["input_images:0", "input_masks:0", "training:0", "encoding/dense/BiasAdd:0"]
    return_elements = ["input_images:0", "input_masks:0", "input_seq_length:0", "training:0",
                       "CTCBeamSearchDecoder:0", "CTCBeamSearchDecoder:1", "CTCBeamSearchDecoder:2"]

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())
    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        with tf.Session(graph=graph) as sess:
            path = '/home/ubuntu/workspace/ocr3/self_attention_dataset/test/'
            # for i in os.listdir(path):
            r = 0
            for item in files:
                res = image_demo(path + item[0], return_elements)
                if res == item[1]:
                    r += 1
            print(r)
                # print(time.time()-a)

