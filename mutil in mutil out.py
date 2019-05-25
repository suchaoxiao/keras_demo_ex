import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense
from keras.utils import plot_model, to_categorical
import numpy as np

main_input = Input(shape=(100,), name='main_input', dtype='int32')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

auxiliary_out = Dense(1, activation='sigmoid', name='auxiliary_output')(lstm_out)
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_out])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
model.summary()
plot_model(model, to_file='n.png', show_shapes=True)
headline_data = np.random.random((1000, 100))
label = np.random.randint(1, (1000, 1))
labels = to_categorical(label, num_classes=1)
additional_data = np.random.random((100, 5))


model.fit([headline_data, additional_data], [labels, labels], batch_size=32, epochs=10)
