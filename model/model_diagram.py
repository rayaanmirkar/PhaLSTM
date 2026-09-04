import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization, GlobalMaxPool1D, Bidirectional
from keras.models import Sequential

max_len = 55000


model = Sequential()
model.add(Input(shape=(max_len,), dtype='int32')) 
model.add(Embedding(input_dim=(29+1), output_dim=32, mask_zero=True))
model.add(Conv1D(filters = 128, kernel_size=15, strides=1, activation= 'relu', padding='same'))
model.add(Bidirectional(LSTM(units=64, dropout=0.2)))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

plot_model(
    model, 
    to_file='model_diagram.png', 
    show_shapes=True, 
    show_layer_names=True,
    show_layer_activations=True
)