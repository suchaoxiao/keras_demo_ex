#使用resnet50进行imagenet分类
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
''' resnet做预测
model = ResNet50(weights='imagenet')
img_path='00.jpg'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
preds=model.predict(x)

print('pred',decode_predictions(preds,top=3)[0])
#vgg16 model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
model=VGG16(weights='imagenet')
fea_model=Model(inputs=model.input,outputs=model.get_layer('block2_pool').output)

img_path='00.jpg'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
features=fea_model.predict(x)
print(features.shape)
'''
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D
from keras import backend as K

base_model=InceptionV3(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(200,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
model.fit_generator(...)

for i,layer in enumerate(base_model.layers):
    print(i,layer.name)
for layer in model.layers[:249]:
    layer.trainable=False
for layer in model.layers[249:]:
    layer.trainable=True
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss='categorical_crossentropy')

model.fit_generator(...)