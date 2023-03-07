# show the weight initialization per normal distribution or glorot/xvazier distribution
# from section 9 of book "Deep Learning Illustrated"

from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Activation
from keras.initializers import Zeros, RandomNormal
from keras.initializers import glorot_normal, glorot_uniform

n_input = 784
n_dense = 256

w_init = RandomNormal(stddev=1.0)
# w_init = glorot_normal()
b_init = Zeros()

model = Sequential()
model.add(Dense(n_dense,
          input_dim=n_input,
          kernel_initializer=w_init,
          bias_initializer=b_init))
model.add(Activation('sigmoid'))

x = np.random.random((1, n_input))

a = model.predict(x)
# print (f"{a}")
# _ = plt.hist(np.transpose(a))
_ = plt.hist(a)

plt.show()
