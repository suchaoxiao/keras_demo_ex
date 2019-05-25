import keras
from keras.layers import Conv2D,Flatten,MaxPool2D,Input,Embedding,Dense,LSTM
from keras.models import Model

input_img=Input(shape=(224,224,3))
x=Conv2D(64,(3,3),activation='relu',padding='same')(input_img)
x=Conv2D(64,(3,3),activation='relu')(x)
x=MaxPool2D((2,2))(x)
x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
x=Conv2D(128,(3,3),activation='relu')(x)
x=MaxPool2D((2,2))(x)
x=Conv2D(256,(3,3),activation='relu',padding='same')(x)
x=Conv2D(256,(3,3),activation='relu')(x)
x=MaxPool2D((2,2))(x)
encoded_img=Flatten()(x)
# vision_model=Model(inputs=input_img,outputs=encoded_img)

question_input=Input(shape=(100,),dtype='int32')
embedded_question=Embedding(input_dim=1000,output_dim=256,input_length=100)(question_input)
encoded_question=LSTM(256)(embedded_question)
merged=keras.layers.concatenate([encoded_img,encoded_question])
output=Dense(10,activation='softmax')(merged)
vqa_model=Model([input_img,question_input],output)
vqa_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
import numpy as np
data_img=np.random.random((1000,224,224,3))
data_que=np.random.random((1000,100))
label=np.random.randint(10,size=(1000,1))
onehot=keras.utils.to_categorical(label,num_classes=10)

vqa_model.fit([data_img,data_que],onehot,batch_size=16,epochs=2)

