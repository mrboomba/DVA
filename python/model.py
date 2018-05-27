from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import numpy as np



N_img_height = 66
N_img_width = 220
N_img_channels = 3
def nvidia_model():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

    model.add(Conv2D(24, (5, 5),kernel_initializer = 'he_normal',name = 'conv1', padding = 'valid',strides=(2,2),))
    
    model.add(ELU())    
    model.add(Conv2D(36, (5, 5), kernel_initializer = 'he_normal',name = 'conv2', padding = 'valid',strides=(2,2)))
    
    model.add(ELU())    
    model.add(Conv2D(48, (5, 5), kernel_initializer = 'he_normal',name = 'conv3',padding = 'valid',strides=(2,2)))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_normal',name = 'conv4',padding = 'valid',strides = (1,1)))
    
    model.add(ELU())              
    model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_normal',name = 'conv5',padding = 'valid',strides= (1,1)))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')

    return model

