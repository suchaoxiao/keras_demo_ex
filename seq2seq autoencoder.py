# -- coding: utf-8 --
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Reshape
import numpy as np
from keras.callbacks import ReduceLROnPlateau
import keras
from keras.utils import plot_model
from sklearn.metrics import r2_score

data_x=np.random.random((1000,100,18))

batch_size = 64  # Batch size for training.
epochs = 20 # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

encoder_input_data = data_x
decoder_input_data = data_x
decoder_target_data =data_x
# decoder_target_data =np.concatenate(data_x[:,1:,:],data_x[:,0,:].reshape((-1,1,18)),axis=1)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 18))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 18))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(18)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5)
# Run training
model.compile(optimizer='rmsprop', loss='mae')
plot_model(model,'1.png',show_shapes=True)
model.fit([encoder_input_data,decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,callbacks=[reduce_lr])
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 18))
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        decoded_sentence.extend(output_tokens)

        # decoded_sentence=np.append(decoded_sentence,output_tokens)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) >= 100:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens
        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1,:,:]
    decoded_sentence = decode_sequence(input_seq)
    decoded_sentence = np.array(decoded_sentence).reshape((100,18))
    original=encoder_input_data[seq_index: seq_index + 1,:,:].reshape((100,18))
    print('--------------------------------')
    print(r2_score(original,decoded_sentence))
    # print('Input sentence:', original)
    # print('Decoded sentence:', decoded_sentence)
