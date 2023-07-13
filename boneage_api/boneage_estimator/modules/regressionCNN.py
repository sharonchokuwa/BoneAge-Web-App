
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K 

class SmallerVGGNet: 
    @staticmethod
    def build(width, height, depth):
        #initialize the model, input shape, and channels dim
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        #CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.5))
        
        #(CONV => RELU) => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        
        #(CONV => RELU) => POOL
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        #first and only FC =>  RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        #output layer
        model.add(Dense(1, activation='sigmoid'))
        # return the constructed network architecture
        return model

