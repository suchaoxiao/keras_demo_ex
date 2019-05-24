import keras
from keras.layers import Dense,Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

inputs=Input(shape=(784,))
x=Dense(64,activation='relu')(inputs)
x=Dense(64,activation='relu')(x)
pred=Dense(10,activation='softmax')(x)
model= Model(inputs=inputs,outputs=pred)
inputs=np.random.random((10000,784))
labels=np.random.randint(10,size=(10000,1))
labels=to_categorical(labels,num_classes=10)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(inputs,labels)
