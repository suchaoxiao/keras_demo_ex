import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(100,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import numpy as np

data = np.random.random((10000, 100))
label = np.random.randint(10, size=(10000, 1))
one_hot = to_categorical(label, 10)

model.fit(data, one_hot, batch_size=32,
          epochs=10, verbose=1, validation_split=0.1)
