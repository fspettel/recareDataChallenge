from time import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# Network architechture, very simple but proved to work best
def getModel():

  model = Sequential()
  model.add(Dense(64, input_dim=5))
  model.add(Activation('relu'))
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

