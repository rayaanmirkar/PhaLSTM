# 
# 
# 
# 8/15/2026 (Scrapped) ->> Multiheaded Self Attention (MHSA) would be a viable idea, but it does not account for the mosiac makeup of various different bacteriophages. Also, it does not infer motifs sequentially as well as the original bilstm architecture.


import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.layers import Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization, Add
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization
from keras.models import Sequential, Model
from sklearn.metrics import classification_report
#keras.mixed_precision.set_global_policy("mixed_float16")

# loss, accuracy, F1-score, precision, recall, ROC-AUC, and PR-AUC
max_features = 18000
chunk = 2000
stride_size = 2000

def chunk_seq(sequence, chunk_size, stride):

    chunks = []

    for i in range(0, len(sequence), stride):

        chunk = sequence[i:i + chunk_size]

        if len(chunk) == chunk_size:
            chunks.append(chunk)

    return chunks


training_df = pd.read_csv(r'C:\Users\raypi\coding\phager\building_data\training_data.csv')
testing_df = pd.read_csv(r'C:\Users\raypi\coding\phager\building_data\testing_data.csv')
validation_df = pd.read_csv(r"C:\Users\raypi\coding\phager\building_data\validation_data.csv")

training_df = training_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
testing_df = testing_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
validation_df = validation_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])

x_training = []
y_training = []

for seq, label in zip(
    training_df['protein_sentence'].astype(str),
    training_df['Binary Lifestyle'].astype(np.float32)
):

    chunks = chunk_seq(seq, chunk, stride_size)

    x_training.extend(chunks)

    y_training.extend([label] * len(chunks))


x_validation = []
y_validation = []

for seq, label in zip(
    validation_df['protein_sentence'].astype(str),
    validation_df['Binary Lifestyle'].astype(np.float32)
):

    chunks = chunk_seq(seq, chunk, stride_size)

    x_validation.extend(chunks)

    y_validation.extend([label] * len(chunks))


x_testing = []
y_testing = []

for seq, label in zip(
    testing_df['protein_sentence'].astype(str),
    testing_df['Binary Lifestyle'].astype(np.float32)
):

    chunks = chunk_seq(seq, chunk, stride_size)

    x_testing.extend(chunks)

    y_testing.extend([label] * len(chunks))


y_training = np.array(y_training, dtype=np.float32)
y_validation = np.array(y_validation, dtype=np.float32)
y_testing = np.array(y_testing, dtype=np.float32)

vectorization_layer = TextVectorization(
    max_tokens= 26,
    standardize=None,
    output_mode="int",
    output_sequence_length=chunk,
    split="character")

vectorization_layer.adapt(x_training)
dim_size = vectorization_layer.vocabulary_size() # maybe use vocab size method?
    
#old model architecture
'''
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=(dim_size+1), output_dim=32))
model.add(Conv1D(filters = 128, kernel_size=15, strides=1, activation= 'relu'))
model.add(Bidirectional(LSTM(units=64, dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''




inputs = Input(shape=(1,), dtype=tf.string)
x = vectorization_layer(inputs)
x = Embedding(input_dim=(dim_size+1), output_dim=32)(x)
x = Conv1D(filters=128, kernel_size=15, strides=1, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=10)(x)

bilstm = Bidirectional(LSTM(units=64, dropout=0.2, return_sequences=True))(x)



#ATTENTION MECHANISM
attention_out = MultiHeadAttention(num_heads=2, key_dim=32, output_shape=128)(query=bilstm, value=bilstm, key=bilstm)
attention_out = Add()([bilstm, attention_out])
attention_out = LayerNormalization()(attention_out)
attention_out = Dropout(0.2)(attention_out)

pooled_out = GlobalAveragePooling1D()(attention_out)
outputs = Dense(1, activation='sigmoid', dtype ='float32')(pooled_out)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#DATASETS TENSORSLICING
train_ds = tf.data.Dataset.from_tensor_slices((x_training, y_training)).batch(8)

val_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(8)
test_ds = tf.data.Dataset.from_tensor_slices((x_testing, y_testing)).batch(8)

counts = np.bincount(y_training.astype(np.int32))
weights = {0: (len(y_training) / (2.0 * counts[0])), 1: (len(y_training) / (2.0 * counts[1]))}


train = model.fit(
    train_ds,
    epochs=10,
    validation_data= val_ds,
    class_weight=weights

)
model.save("phage-bilstm_SAVE.keras")


test_tensor_inputs = tf.constant(x_testing, dtype=tf.string)
y_pred_probs = model.predict(test_tensor_inputs, batch_size=8)
y_pred_classes = (y_pred_probs>=0.5).astype("int32")

print("--------------Classification Report:--------------------")
print(classification_report(y_testing, y_pred_classes, target_names=['Temperate', 'Virulent']))


'''
test_loss, test_acc = model.evaluate(test_ds, verbose=1)

print(f"Test Loss:  {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
'''





