# -*- coding: utf-8 -*-
# @Time    : 20-12-14 下午1:38
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : config.py

import json
from cfg.args import parse_args

args = parse_args()

def get_label_length():
    with open(args.char, mode='r', encoding='utf-8') as f:
        res = json.load(f)
        return len(res)


class Config(object):
    BATCH_SIZE = 64

    IMAGE_SIZE_H = 32

    IMAGE_SIZE_W = 512

    DOWN_SAMPLE = 4

    ######encoding parameter########
    HIDDEN_SIZE = 768

    NUM_ATTENTION_HEADS = 12

    NUM_HIDDEN_LAYERS = 8

    INTER_DENSE_SIZE = 3072

    OUT_FEATURE = get_label_length() + 1

    ############
    LEARNING_RATE = 0.000001

    STEPS_PER_DECAY = 200

    DECAY_FACTORY = 1

    CUDA_VISIBLE_DEVICES = '0'

    GPU_MEMORY = 0.9

    MOVING_AVERAGE_DECAY = 0.998

    ITERATION = 20000

    ITER_SUMMARY = 10

    ITER_SAVE = 400


if __name__ == '__main__':
    get_label_length()

