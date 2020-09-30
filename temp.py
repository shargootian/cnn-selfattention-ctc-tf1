import numpy as np
# import tensorflow as tf

total_series_length = 50000
echo_step = 3
batch_size = 5


def generateData():
    x = np.array(np.random.choice(2,total_series_length,p=[0.5,0.5]))
    y = np.roll(x,echo_step)
    y[:echo_step] = 0

    x = x.reshape((batch_size,-1))
    y = y.reshape((batch_size,-1))
    return x,y


x,y = generateData()
print(1)