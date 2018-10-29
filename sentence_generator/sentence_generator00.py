from __future__ import print_function
from keras.layers.core import Activation, Dense, Dropout, Flatten, Masking
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform, uniform, orthogonal, TruncatedNormal
from keras.optimizers import RMSprop
from keras import regularizers
from keras.constraints import maxnorm, non_neg
from keras.utils.data_utils import get_file
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import numpy.random as nr
import sys
import h5py
import keras
import math
import MeCab

######### ニューラルネットワーク本体

class Prediction :
  def __init__(self, maxlen, n_hidden, input_dim, vec_dim, output_dim):
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.input_dim = input_dim
    self.vec_dim = vec_dim
    self.output_dim = output_dim
        
  def create_model(self):
    model = Sequential()
    print('#3')
    model.add(Embedding(self.input_dim, self.vec_dim, input_length=self.maxlen,
          embeddings_initializer=uniform(seed=20170719)))
    model.add(BatchNormalization(axis=-1))
    print('#4')
    model.add(Masking(mask_value=0, input_shape=(self.maxlen, self.vec_dim)))
    model.add(LSTM(self.n_hidden, batch_input_shape=(None, self.maxlen, self.vec_dim),
             activation='tanh', recurrent_activation='hard_sigmoid', 
             kernel_initializer=glorot_uniform(seed=20170719), 
             recurrent_initializer=orthogonal(gain=1.0, seed=20170719), 
             dropout=0.5, 
             recurrent_dropout=0.5))
    print('#5')
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    print('#6')
    model.add(Dense(self.output_dim, activation=None, use_bias=True, 
            kernel_initializer=glorot_uniform(seed=20170719), 
            ))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['categorical_accuracy'])
    return model
  
  # 学習
  def train(self, x_train, t_train, batch_size, epochs) :
    early_stopping = EarlyStopping(patience=1, verbose=1)
    print('#2', t_train.shape)
    model = self.create_model()
    print('#7')
    model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
          shuffle=True, callbacks=[early_stopping], validation_split=0.1)
    return model

######### 辞書データ作成

f = open('akai_kabutomushi.txt')
text = f.read() 
tagger = MeCab.Tagger("-Owakati")
mat = tagger.parse(text)
mat = list(map(str, mat.split(' ')))
words = sorted(list(set(mat)))
cnt = np.zeros(len(words))

print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

# 単語の出現数をカウント
for j in range(0, len(mat)):
  cnt[word_indices[mat[j]]] += 1

# 出現頻度の少ない単語を「UNK」で置き換え
words_unk = []                # 未知語一覧

for k in range(0, len(words)):
  if cnt[k]<=3 :
    words_unk.append(words[k])
    words[k] = 'UNK'

print('低頻度語数:', len(words_unk))           # words_unkはUNKに変換された単語のリスト

words = sorted(list(set(words)))
print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

########## 訓練データ作成

maxlen = 40                # 入力語数

mat_urtext = np.zeros((len(mat), 1), dtype=int)
for i in range(0, len(mat)):
  #row = np.zeros(len(words), dtype=np.float32)
  if mat[i] in word_indices :       # 出現頻度の低い単語のインデックスをUNKのそれに置き換え
    if word_indices[mat[i]] != 0 :  # 0パディング対策
      mat_urtext[i, 0] = word_indices[mat[i]]
    else :
      mat_urtext[i, 0] = len(words)
  else:
    mat_urtext[i, 0] = word_indices['UNK']

print(mat_urtext.shape)

# 単語の出現数をもう一度カウント：UNK置き換えでwords_indeicesが変わっているため
cnt = np.zeros(len(words)+1)
for j in range(0, len(mat)):
  cnt[mat_urtext[j, 0]] += 1

print(cnt.shape)

len_seq = len(mat_urtext)-maxlen
data = []
target = []
for i in range(0, len_seq):
  data.append(mat_urtext[i:i+maxlen, :])
  target.append(mat_urtext[i+maxlen, :])

x = np.array(data).reshape(len(data), maxlen, 1)
t = np.array(target).reshape(len(data), 1)

z = list(zip(x, t))
nr.shuffle(z)                 # シャッフル
x, t = zip(*z)
x = np.array(x).reshape(len(data), maxlen, 1)
t = np.array(t).reshape(len(data), 1)
print(x.shape, t.shape)

n_train = int(len(data)*0.9)                     # 訓練データ長
x_train, x_validation = np.vsplit(x, [n_train])  # 元データを訓練用と評価用に分割
t_train, t_validation = np.vsplit(t, [n_train])  # targetを訓練用と評価用に分割

########## メイン処理

vec_dim = 400 
epochs = 100
batch_size = 200
input_dim = len(words)+1
output_dim = input_dim
n_hidden = int(vec_dim*1.5)  # 隠れ層の次元

prediction = Prediction(maxlen, n_hidden, input_dim, vec_dim, output_dim)

emb_param = 'param_1.hdf5'             # パラメーターファイル名
row = x_train.shape[0]
x_train = x_train.reshape(row, maxlen)
model = prediction.train(x_train, np_utils.to_categorical(t_train, output_dim), batch_size, epochs)

model.save_weights(emb_param)          # 学習済みパラメーターセーブ

row2 = x_validation.shape[0]
score = model.evaluate(x_validation.reshape(row2, maxlen), 
             np_utils.to_categorical(t_validation, output_dim), batch_size=batch_size)

print("score:", score)
