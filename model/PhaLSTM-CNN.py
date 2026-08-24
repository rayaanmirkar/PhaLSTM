import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization, GlobalMaxPool1D
from keras.models import Sequential
from sklearn.metrics import classification_report, matthews_corrcoef, PrecisionRecallDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt



# loss, accuracy, F1-score, precision, recall, ROC-AUC, and PR-AUC
max_features = 18000
chunk = 5000
stride_size = 5000




training_df = pd.read_csv('building_data/training_data.csv')
testing_df = pd.read_csv('building_data/testing_data.csv')
validation_df = pd.read_csv('building_data/validation_data.csv')

training_df = training_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
testing_df = testing_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
validation_df = validation_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])


x_training = training_df['protein_sentence'].astype(str).values
y_training = training_df['Binary Lifestyle'].astype(np.float32).values

x_validation = validation_df['protein_sentence'].astype(str).values
y_validation = validation_df['Binary Lifestyle'].astype(np.float32).values

x_testing = testing_df['protein_sentence'].astype(str).values
y_testing = testing_df['Binary Lifestyle'].astype(np.float32).values

max_len = max(
    max(len(seq) for seq in x_training),
    max(len(seq) for seq in x_validation),
    max(len(seq) for seq in x_testing)
)






y_training = np.array(y_training, dtype=np.float32)
y_validation = np.array(y_validation, dtype=np.float32)
y_testing = np.array(y_testing, dtype=np.float32)

vectorization_layer = TextVectorization(
    max_tokens= 30,
    standardize=None,
    output_mode="int",
    output_sequence_length=max_len,
    split="character")

vectorization_layer.adapt(x_training)
dim_size = vectorization_layer.vocabulary_size() # maybe use vocab size method?
    
#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=(dim_size+1), output_dim=32, mask_zero=True))
model.add(Conv1D(filters = 128, kernel_size=15, strides=1, activation= 'relu', padding='same'))
model.add(Bidirectional(LSTM(units=64, dropout=0.2)))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((x_training, y_training)).batch(64)

val_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_testing, y_testing)).batch(64)


weights = compute_class_weight('balanced', classes=np.unique(y_training), y=y_training)
class_weight_dict = dict(enumerate(weights))


train = model.fit(
    train_ds,
    epochs=10,
    validation_data= val_ds,
    class_weight=class_weight_dict

)
model.save("phage-bilstm_SAVE.keras")

y_true = y_testing
y_pred_probs = model.predict(x_testing).flatten() 
y_pred_classes = (y_pred_probs>=0.5).astype("int32")

mcc = matthews_corrcoef(y_true, y_pred_classes)


#proteome size vs performance 
lengths = [len(seq) for seq in testing_df['protein_sentence']] 
df = pd.DataFrame({"Length": lengths, "Correct": (y_pred_classes.flatten() == y_testing.flatten())})
df["Size"] = df["Size"] = pd.qcut(df["Length"], q=3, labels=["Small", "Medium", "Large"], duplicates='drop')
df.groupby("Size", observed=False)["Correct"].mean().plot(kind="bar", color="#4C72B0")
plt.ylabel("Accuracy Rate")
plt.savefig("dilution_graph.png")
plt.close()


##PRC GRAPH
PrecisionRecallDisplay.from_predictions(y_true, y_pred_probs, color="#DD8452")
plt.savefig("prc_graph.png")
plt.close()


print("--------------Classification Report:--------------------")
print(classification_report(y_true, y_pred_classes, target_names=['Temperate', 'Virulent']))
print(f"MCC Score: {mcc:.4f}")


'''
test_loss, test_acc = model.evaluate(test_ds, verbose=1)

print(f"Test Loss:  {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
'''

    



