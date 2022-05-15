import json
import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# PARAMETERS
max_length = 20
trunc_type='post'
padding_type='post'


# LOAD MODEL AND TOKENIZER
model = tf.keras.models.load_model('my_model')
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


#PREDICTION FUNCTION
def Pred(vers):
    sequences = tokenizer.texts_to_sequences(vers)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = model.predict(padded)
    if (result[0][0] > 0.4):
        print("This verse is Gazal")
    else:
        print("This verse is NOT Gazal")


#TEST
Verse1 = ["لهفي عليها ولهفي من تذكرها يدنو تذكرها مني وتنآني"] #غزل
Verse2 = ["أخماع لو أصبحت وسط رحالهم عرفت خماعة أنها لا تخفر"] #هجاء
Pred(Verse1)
Pred(Verse2)