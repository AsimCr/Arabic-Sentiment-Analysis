# IMPORTATIONs
import json
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PARAMETERS
vocab_size = 50000
embedding_dim = 16
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"


# DATA_IMPORTING
with open("poem.json", 'r', encoding='utf-8') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['verse'])
    labels.append(item['is_Gazal'])


# FUNCTION TO SAVE TOKENIZER FOR LATER USE
def Tokenizer_saver(Name,tokenizer):
    tokenizer_json = tokenizer.to_json()
    with io.open((Name+".json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


# PREPROCESSING
# Words to number
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
Tokenizer_saver("tokenizer",tokenizer)
# numbers to sequence
training_sequences = tokenizer.texts_to_sequences(sentences)
# sequence padding
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(labels)


# MODEL CREATION
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


num_epochs = 5
# MODEL TRAINING
history = model.fit(training_padded, training_labels, epochs=num_epochs, verbose=1)
#MODEL SAING FOR LATER USE
model.save('my_model')
