import keras
from keras import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten
from keras.layers import concatenate

input_img = Input(shape=(256, 256, 3))
tower_1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
tower_3 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(tower_3)
output = concatenate([tower_1, tower_2, tower_3], axis=1)
output =Flatten()(output)
output = Dense(1, activation='sigmoid')(output)
model = Model(inputs=input_img, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
import numpy as np

data_img = np.random.random((1000, 256, 256, 3))
label = np.random.randint(1, size=(1000, 1))
onehot_label = keras.utils.to_categorical(label, num_classes=1)
model.fit(data_img, onehot_label, batch_size=16, epochs=10)
