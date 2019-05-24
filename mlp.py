import keras
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = np.random.random((10000, 100))
label = np.random.randint(10, size=(10000, 1))
one_hot = to_categorical(label)
data_train, data_test, label_train, label_test = train_test_split(data, one_hot, test_size=0.2)
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(100,)))
model.add(Dropout(0.4, seed=3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2, seed=3))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.001, momentum=0.9, decay=0.01, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data_train, label_train, batch_size=16, epochs=
10, verbose=2)
model.evaluate(data_test, label_test, batch_size=16)