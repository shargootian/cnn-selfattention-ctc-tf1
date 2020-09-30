class Config(object):
    BATCH_SIZE = 32

    INPUT_H = 32

    CELL_DIM = 1024

    LEARNING_RATE = 0.001

    STEPS_PER_DECAY = 200

    DECAY_FACTORY = 1

    MOVING_AVERAGE_DECAY = 0.998

    CUDA_VISIBLE_DEVICES = '0'

    GPU_MEMORY = 0.9

    ITERATION = 20000

    ITER_SUMMARY = 10

    ITER_SAVE = 400
