import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['accuracy'])
import numpy as np

data = np.random.random((10000, 100))
labels = np.random.randint(2, size=(10000, 1))
model.fit(data, labels, epochs=10, batch_size=32)
