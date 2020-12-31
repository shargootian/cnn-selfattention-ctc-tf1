# -*- coding: utf-8 -*-
# @Time    : 20-12-14 下午1:47
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : args.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SELF-ATTENTION')

    parser.add_argument('--char_json', dest='char', help="char list",
                        default='./cfg/chars.json', type=str)

    parser.add_argument('--train_data_path', dest='train_data', help="train dataset location",
                        default='../self_attention_dataset/train/', type=str)

    parser.add_argument('--logs', dest='logs', help="events logs files saveing path",
                        default='./logs/', type=str)

    parser.add_argument('--initial_weight', dest='initial_weight', help="initial weight for ckpt",
                        default='./logs/pretrain/model_0.05338.ckpt-1200', type=str)

    parser.add_argument('--demo_weight', dest='demo_weight', help="initial weight for ckpt",
                        default='./logs/demo/model_0.00985.ckpt-6000', type=str)

    parser.add_argument('--pb_path', dest='pb', help="pb_model files saveing path",
                        default='./logs/pb/db.pb', type=str)

    args = parser.parse_args()

    return args