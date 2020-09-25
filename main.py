import os
import tensorflow as tf
from cfg.args import parse_args
from libs.network import Network
from cfg.config import Config as C


class crnn(Network):
    def __init__(self, args):
        super(crnn, self).__init__(args, C)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = C.GPU_MEMORY
        os.environ["CUDA_VISIBLE_DEVICES"] = C.CUDA_VISIBLE_DEVICES
        self.sess = tf.Session(config=config)
        self.chars_dict = self.load_dict(self.args.char_path)
        self.nclass = len(self.chars_dict) + 1

    def train(self):
        pred = self._crnn(self.input_images, self.trainable)


if __name__ == '__main__':
    args = parse_args()
    C = crnn(args)
    C.train()

