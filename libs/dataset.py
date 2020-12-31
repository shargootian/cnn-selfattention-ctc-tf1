# -*- coding: utf-8 -*-
# @Time    : 20-12-14 下午1:34
# @Author  : shargootian
# @Email   : wangwei@shargoodata.com
# @File    : dataset.py

import os
import json
import random
import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa

aug = [iaa.LinearContrast(alpha=2),
       iaa.SigmoidContrast(gain=10),
       iaa.GammaContrast(gamma=2),
       iaa.CLAHE(clip_limit=(1, 5)),
       iaa.Grayscale(alpha=1.0),
       iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
       iaa.BilateralBlur(d=6),
       iaa.MotionBlur(k=7),
       iaa.MedianBlur(k=3),
       iaa.AverageBlur(k=3),
       iaa.AdditiveGaussianNoise(loc=0.8, scale=(0.01, 0.08*255)),
       iaa.ContrastNormalization((0.3, 1.5)),
       iaa.Sharpen(alpha=0, lightness=1)]



class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.char = json.load(open(self.args.char,mode='r'))


    def get_file(self, data_path):
        temp = []
        txts = os.listdir(data_path + 'labels')
        for txt in txts:
            with open(data_path + 'labels/' + txt, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    num = []
                    img_path = data_path + 'images/' + txt.split(".")[0] + '/' + line.split(":")[0]
                    img_lab = line.split(":")[1].strip()
                    for item in img_lab:
                        num.append(self.char[item])
                    temp.append([img_path, img_lab, num])
        random.shuffle(temp)

        return temp

    def get_data(self, C, image_label):
        temp = random.sample(image_label, C.BATCH_SIZE)
        images = np.zeros((C.BATCH_SIZE, C.IMAGE_SIZE_H, C.IMAGE_SIZE_W, 3), dtype=np.float)
        masks = np.zeros((C.BATCH_SIZE, C.IMAGE_SIZE_W // 4), dtype=np.int32)
        seql = np.ones(C.BATCH_SIZE, dtype=np.int32) * (C.IMAGE_SIZE_W // C.DOWN_SAMPLE)

        count = 0
        for item in temp:
            ######数据处理和增强######
            img, mask_len = self.img_aug(item, C)
            images[count] = img
            ######生成mask标签######
            mask = self.gen_mask(mask_len, C)
            masks[count] = mask
            count += 1

        ######生成gt标签######
        labels = self.gen_gt(temp)

        return images, masks, labels, seql


    def img_aug(self, item, C):
        img = cv.imread(item[0])
        h, w, _ = img.shape
        w = w * C.IMAGE_SIZE_H // h
        flag = random.sample([0,], 1)[0]
        if flag != 0:
            method = random.sample(aug, 1)[0]
            img = method.augment_images(img)
        if w < C.IMAGE_SIZE_W:
            img = cv.resize(img, (w, C.IMAGE_SIZE_H))
            tail = np.zeros((32, C.IMAGE_SIZE_W - w, 3))
            img = np.concatenate((img, tail), axis=1)
            mask_len = w // C.DOWN_SAMPLE
        else:
            img = cv.resize(img, (C.IMAGE_SIZE_W, C.IMAGE_SIZE_H))
            mask_len = C.IMAGE_SIZE_W // C.DOWN_SAMPLE

        img = img / 255.0 - 0.5

        return img, mask_len


    def gen_mask(self, mask_len, C):
        mask_head = np.ones((mask_len), dtype=np.int32)
        mask_tail = np.zeros(C.IMAGE_SIZE_W // C.DOWN_SAMPLE - mask_len, dtype=np.int32)
        mask = np.concatenate((mask_head, mask_tail), axis=0)

        return mask


    def gen_gt(self, data):
        indices = []
        values = []

        for n, seq in enumerate(data):
            indices.extend(zip([n] * len(seq[2]), range(len(seq[2]))))
            values.extend(seq[2])

        indices = np.array(indices, dtype=np.int64)
        values = np.array(values, dtype=np.float32)
        shape = np.asarray([len(data), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
        # shape = np.array([C.BATCH_SIZE, C.IMAGE_SIZE_W // C.DOWN_SAMPLE], dtype=np.int64)

        return indices, values, shape


if __name__ == '__main__':
    from cfg.args import parse_args
    from cfg.config import Config as C

    args = parse_args()
    D = Dataset(args)
    temp = D.get_file(args.train_data)
    data = D.get_data(C, temp)