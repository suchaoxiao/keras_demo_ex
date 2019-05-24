import keras
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.optimizers import  SGD
import numpy as np
from keras.models import Sequential

data_image=np.random.random((100,32,32,3))
label_image=np.random.randint(1,size=(100,1))
one_hot=to_categorical(label_image,num_classes=2)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',
                 input_shape=(32,32,3)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu')
          )
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))

sgd=SGD(lr=0.001,momentum=0.9,decay=1e-5,nesterov=True)
model.compile(optimizer=sgd,loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(data_image,one_hot,batch_size=16,epochs=10)