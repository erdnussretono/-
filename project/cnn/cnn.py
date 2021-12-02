import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, concatenate
from keras.models import load_model, save_model
from tensorflow.python.ops.gen_math_ops import angle

# 데이터 읽어오기
train_file = 'cnn\senti_vali_train.csv'
data = pd.read_csv(train_file, delimiter=',')
data = data.dropna()
features = data['q'].tolist() # 해당칼럼만 리스트로 만들기
labels = data['sentinum'].tolist()
labels= list(map(int, labels))

corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
# sequences = tokenizer. texts_to_sequences(corpus)
# word_index = tokenizer.word_index


cnnmodel = load_model('./cnn/cnn_chatbot_model.h5')
# model.summary()

# pleasure=0
# panic=0
# angry=0
# unrest=0
# wound=0
# sad=0
label_text = {0: 'pleasure', 1: 'panic', 2: 'angry', 3: 'unrest', 4: 'wound', 5: 'sad'}


def padded_sequence(text):
    text_seq= [preprocessing.text.text_to_word_sequence(text)]
    text_seq = tokenizer.texts_to_sequences(text_seq)

    MAX_SEQ_LEN =32 # 단어 시퀀스 백터 크기
    padded_seqs = preprocessing.sequence.pad_sequences(text_seq, 
                                                      maxlen=MAX_SEQ_LEN,
                                                      padding='post')


    predicrion = cnnmodel.predict(padded_seqs)
    predicrion_class = tf.math.argmax(predicrion, axis=1)
    emotion_predicrion=(label_text[predicrion_class.numpy().item()])
    print(emotion_predicrion)
    return emotion_predicrion



# global pleasure
# global panic
# global angry
# global unrest
# global wound
# global sad

# pleasure=0
# panic=0
# angry=0
# unrest=0
# wound=0
# sad=0

# def count(emotion):
#     global pleasure
#     global panic
#     global angry
#     global unrest
#     global wound
#     global sad
#     if (emotion =='기쁨'):
#         pleasure+=1
#     elif (emotion =='당황'):
#         panic+=1
#     elif (emotion =='분노'):
#         angry+=1
#     elif (emotion =='불안'):
#         unrest+=1
#     elif (emotion =='상처'):
#         wound+=1
#     elif (emotion =='슬픔'):
#         sad+=1
    