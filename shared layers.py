import keras
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.utils import to_categorical, plot_model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))

shared_lstm = LSTM(64)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

import numpy as np

data_a = np.random.random((1000, 280, 256))
data_b = np.random.random((1000, 280, 256))
label= np.random.randint(1, size=(1000, 1))
label = to_categorical(label, num_classes=1)

model.fit([data_a, data_b], label, batch_size=16, epochs=10)
