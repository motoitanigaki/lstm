from __future__ import print_function
import collections
import os, sys
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import MeCab

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

result_path = 'result'
run_opt = 1 # 2にしたかったらコマンドライン引数に2を追加
args = sys.argv
try:
    run_opt = args[1]
except:
    pass

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def read_words(filename):
    f = open(filename)
    text = f.read()
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(text)
    text = list(map(str, text.split(' ')))
    text = remove_values_from_list(text,"\u3000") 
    return text

def build_vocab(filename):
    """
    サンプルテキスト内の単語を頻度順に並べて全てvocabularyに利用
    """
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1,len(words)+1))) # 

    return word_to_id

def file_to_word_ids(filename, word_to_id):
    """
    Blogのサンプルだと全ての単語を利用しているので word in data not ifは起きない    
    """
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    # get the data paths
    train_path = os.path.join("text_all_articles_train.txt")
    valid_path = os.path.join("text_all_articles_valid.txt")
    test_path = os.path.join("ichimaino_kippu.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id) + 1 # 0パディング用にvocabularyを1つ追加
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print("train_data[:5]",train_data[:5])
    # print(word_to_id) This is 10000 words dictionary
    print("vocabulary:",vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary, word_to_id

train_data, valid_data, test_data, vocabulary, reversed_dictionary, word_to_id = load_data()

def generate_sentence(original_text, word_to_id):
    tagger = MeCab.Tagger("-Owakati")
    text = tagger.parse(original_text)
    text = list(map(str, text.split(' ')))
    text = remove_values_from_list(text,"\n") 
    print(text)
    text = [word_to_id[word] for word in text if word in word_to_id]
    generated_sentence = ""
    text_len = len(text)
    for i in range(num_steps):
        try:
            text[i]
        except:
            text.append(0)
    text_np = np.array([text])
    predicted_sentence = original_text
    for i in range(text_len - 1,num_steps):
        prediction = model.predict(text_np)
        predict_word = np.argmax(prediction[:, num_steps - 1, :]) 
        text_np[0][i] = predict_word
        predicted_sentence += reversed_dictionary[predict_word]
    print(predicted_sentence)
    return text_np
    
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0 # バッチの数
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

num_steps = 30 # modelに与える文章内の単語数
batch_size = 20 # 1バッチあたりいくつの文章を与えるか
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

hidden_size = 500
use_dropout=True
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps, mask_zero=True)) # model.add(BatchNormalization(axis=-1)) があったほうが良いかも
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True)) # model.add(BatchNormalization(axis=-1)) があったほうが良いかも
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=result_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 100
if run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    model.save(result_path + "final_model.hdf5")
    generate_sentence("人工", word_to_id)
elif run_opt == 2:
    model = load_model(result_path + "/model-50.hdf5")
    dummy_iters = 40
    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Training data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    # test data set
    dummy_iters = 40
    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Test data:")
    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        # predictionのshapeは(1, num_steps, vocabulary)なのでnum_stepsの最後のstep=最後の単語を取得 そして一番確率の高いものを取得
        predict_word = np.argmax(prediction[:, num_steps - 1, :]) 
        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)