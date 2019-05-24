import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
data=np.random.random((10000,100))
label=np.random.randint(1,size=(10000,1))
one_hot=to_categorical(label)
model=Sequential()
model.add(Dense(64,activation='relu',input_dim=100))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
rmsprop=RMSprop(lr=0.0001,decay=0.9)
model.compile(optimizer=rmsprop,loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(data,one_hot,batch_size=16,epochs=10,validation_split=0.1)