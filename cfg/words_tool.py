import os
from cfg.args import parse_args
args = parse_args()


def gen_word(path):
    char_list = []
    files = os.listdir(path + 'labels')
    for file in files:
        with open(path + 'labels/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    temp = line.split(':')[1]
                    for char in temp.strip():
                        if char not in char_list:
                            char_list.append(char)
                except:
                    print(line)
    char_list = sorted(char_list)
    with open('./words.txt', mode='a', encoding='utf-8') as f:
        for char in char_list:
            f.write(char + '\n')


def check_max_label(path):
    char_length = 0
    files = os.listdir(path + 'labels')
    for file in files:
        with open(path + 'labels/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                length = len(line.split(':')[1].strip())
                if length > char_length:
                    char_length = length
    print(char_length)

def check_old_new():
    with open('./words_total.txt', 'r') as old:
        old_lines = old.readlines()
    with open('./words.txt', 'r') as new:
        new_lines = new.readlines()

    for i in new_lines:
        if i not in old_lines:
            print(i)

if __name__ == '__main__':
    gen_word(args.train_data)
    # check_old_new()
    # check_max_label(args.train_data)
    # import cv2 as cv
    # files = os.listdir('/home/ubuntu/workspace/lb_yolo/data_ocr/train/images/ftsdg/')
    # for file in files:
    #     src = cv.imread('/home/ubuntu/workspace/lb_yolo/data_ocr/train/images/ftsdg/' + file)
    #     src = cv.resize(src,(250, 32))
    #     cv.imwrite('./')

