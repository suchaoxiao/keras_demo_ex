import keras
from keras.models import Model
from keras.layers import Dense,Conv2D,Input,MaxPool2D,Flatten

digit_input=Input(shape=(27,27,1))
x=Conv2D(64,(3,3))(digit_input)
x=Conv2D(64,(3,3))(x)
x=MaxPool2D((2,2))(x)
out=Flatten()(x)
vision_model=Model(digit_input,out)

digit_a=Input(shape=(27,27,1))
digit_b=Input(shape=(27,27,1))

out_a=vision_model(digit_a)
out_b=vision_model(digit_b)
concat=keras.layers.concatenate([out_a,out_b])
out=Dense(1,activation='sigmoid')(concat)
class_model=Model(inputs=[digit_a,digit_b],outputs=out)
class_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
import numpy as np
data_a=np.random.random((1000,27,27,1))
data_b=np.random.random((1000,27,27,1))
one_hot=keras.utils.to_categorical(np.random.randint(1,size=(1000,1)),num_classes=1)
class_model.fit([data_a,data_b],one_hot,batch_size=16,epochs=2)