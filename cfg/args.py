import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FACE_DETECTION')

    parser.add_argument('--train_data_path', dest='train_data', help="train dataset lpcation",
                        default='../../crnn_dataset/', type=str)

    parser.add_argument('--logs', dest='logs', help="events logs files saveing path",
                        default='./logs/', type=str)

    parser.add_argument('--initial_weight', dest='initial_weight', help="initial weight for ckpt",
                        default='./logs/pretrain/model_0.9340.ckpt-1600', type=str)

    parser.add_argument('--demo_weight', dest='demo_weight', help="initial weight for ckpt",
                        default='./logs/demo/model_0.3885.ckpt-800', type=str)

    parser.add_argument('--pb_path', dest='pb', help="pb_model files saveing path",
                        default='./logs/pb_model/keypoint.pb', type=str)

    args = parser.parse_args()

    return args
