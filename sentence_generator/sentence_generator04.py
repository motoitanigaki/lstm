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

######### リスト3　辞書データ作成

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
  if cnt[k]<=1 :
    words_unk.append(words[k])
    words[k] = 'UNK'

print('低頻度語数:', len(words_unk))           # words_unkはUNKに変換された単語のリスト

words = sorted(list(set(words)))
print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

########## リスト4-2　単語推定訓練データ作成（出力次元削減）

maxlen = 40      # 入力語数
n_upper = 10     # 学習対象単語の出現頻度上限
n_lower = 0      # 学習対象単語の出現頻度下限

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
for j in range (0, len(mat)):
  cnt[mat_urtext[j, 0]] += 1

print(cnt.shape)

# 頻度対象内単語のリスト
words_0 = []
for i in range(0, len(words)+1) :
  if cnt[i] >= n_lower and cnt[i] < n_upper:
    words_0.append(i)
    
words_0 = sorted(list(set(words_0)))
w0_indices = dict((w, i) for i, w in enumerate(words_0))  # 単語をキーにインデックス検索
indices_w0 = dict((i, w) for i, w in enumerate(words_0))  # インデックスをキーに単語を検索

print(len(words_0))
len_seq = len(mat_urtext) - maxlen

data = []
target = []

for i in range(0, len_seq):
  # 答えの単語の出現頻度がlower_limit以上でかつnum_fleq 未満の場合を学習対象にする
  if cnt[mat_urtext[i+maxlen, :]] >= n_lower and cnt[mat_urtext[i+maxlen, :]] < n_upper:
    #print(mat_urtext[i+maxlen, :])
    data.append(mat_urtext[i:i+maxlen, :])
    target.append(w0_indices[mat_urtext[i+maxlen, :][0]])

x_train = np.array(data).reshape(len(data), maxlen, 1)
t_train = np.array(target).reshape(len(data), 1)

z = list(zip(x_train, t_train))
nr.seed(12345)
nr.shuffle(z)                 # シャッフル
x_train, t_train = zip(*z)

x = np.array(x_train).reshape(len(data), maxlen, 1)
t = np.array(t_train).reshape(len(data), 1)

for i in range(0, maxlen):
  print(x[2, i, :], indices_word[x[2, i, 0]])
print()
print(t[2, :], indices_word[indices_w0[t[2, 0]]])

x_train = x
t_train = t

print(x_train.shape, t_train.shape)

########## リスト4-3　単語分類ごとのインデックス付け

maxlen = 40                # 入力語数

mat_urtext = np.zeros((len(mat), 1), dtype=int)
for i in range(0, len(mat)):
  #row = np.zeros(len(words), dtype=np.float32)
  if mat[i] in word_indices :       # 出現頻度の低い単語のインデックスをunkのそれに置き換え
    if word_indices[mat[i]] != 0 :  # 0パディング対策
      mat_urtext[i, 0] = word_indices[mat[i]]
    else :
      mat_urtext[i, 0] = len(words)
  else:
    mat_urtext[i, 0] = word_indices['UNK']

print(mat_urtext.shape)


# 単語の出現数をもう一度カウント：UNK置き換えでwords_indeicesが変わっているため
cnt = np.zeros(len(words)+1)
for j in range (0, len(mat)):
  cnt[mat_urtext[j, 0]] += 1

print(cnt.shape)

data = []
target = []

len_seq = len(mat_urtext)-maxlen

#for i in range(0, 10):
for i in range(0, len_seq):
  data.append(mat_urtext[i:i+maxlen, :])
  target.append(mat_urtext[i+maxlen, :])

x_train = np.array(data).reshape(len(data), maxlen, 1)
t_train = np.array(target).reshape(len(data), 1)

print(x_train.shape, t_train.shape)

# 頻度対象内単語のリスト
words_0 = []
w0_indices = []
indices_w0 = []
n_upper = [10, 28, 100, 300, 2000, 15000, 400000]
n_lower = [0, 10, 28, 100, 300, 2000, 15000]
for j in range(0, 7) :
  wk = []
  for i in range(0, len(words)+1) :
    if cnt[i] >= n_lower[j] and cnt[i] < n_upper[j]:
      wk.append(i)
  words_0.append(wk)
    
  words_0[j] = sorted(list(set(words_0[j])))
  wi = dict((w, i) for i, w in enumerate(words_0[j]))  # 単語をキーにインデックス検索
  iw = dict((i, w) for i, w in enumerate(words_0[j]))  # インデックスをキーに単語を検索
  w0_indices.append(wi)
  indices_w0.append(iw)

########## リスト5-1　ニューラルネットワーク本体（全データ学習）

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
    early_stopping = EarlyStopping(monitor='loss', patience=1, verbose=1)
    print('#2', t_train.shape)
    model = self.create_model()
    print('#7')
    model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
          shuffle=True, callbacks=[early_stopping], validation_split=0.0)
    return model

########## リスト5-2　単語出現頻度分類用ニューラルネット

class Prediction :
  def __init__(self, maxlen, n_hidden, input_dim, vec_dim, output_dim):
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.vec_dim = vec_dim

  def create_model(self):
    model = Sequential()
    print('#3')
    model.add(Embedding(self.input_dim, self.vec_dim, input_length=self.maxlen, trainable=True,
              embeddings_initializer=uniform(seed=20170719)))
    model.add(BatchNormalization(axis=-1))
    print('#4')
    model.add(Masking(mask_value=0, input_shape=(self.maxlen, self.vec_dim)))
    model.add(LSTM(self.n_hidden, batch_input_shape=(None, self.maxlen, self.vec_dim),
              kernel_initializer=glorot_uniform(seed=20170719),
              recurrent_initializer=orthogonal(gain=1.0, seed=20170719)
              ))
    print('#5')
    model.add(BatchNormalization(axis=-1))
    print('#6')
    model.add(Dense(self.output_dim, activation='sigmoid', use_bias=True,
              kernel_initializer=glorot_uniform(seed=20170719)))
    model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=['binary_accuracy'])
    return model

  # 学習
  def train(self, x_train, t_train, batch_size, epochs, emb_param) :
    early_stopping = EarlyStopping(monitor='loss', patience=4, verbose=1)
    print('#2', t_train.shape)
    model = self.create_model()
    print('#7')
    model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
              shuffle=True, callbacks=[early_stopping], validation_split=0.0)
    return model

########## リスト6-2　単語推定メイン処理（出力次元削減）

n_pattern = 0

vec_dim = 400
epochs = 100
batch_size = 200
input_dim = len(words)+1
output_dim = len(words_0)
n_hidden = int(vec_dim*1.5)  # 隠れ層の次元

prediction = Prediction(maxlen, n_hidden, input_dim, vec_dim, output_dim)
emb_param = 'param_words_'+str(n_pattern)+'_'+str(n_lower)+'_'+str(n_upper)+'.hdf5'  # 学習済みパラメーターファイル名の定義
print (emb_param)
row = x_train.shape[0]
x_train = x_train.reshape(row, maxlen)
model = prediction.train(x_train, np_utils.to_categorical(t_train, output_dim), batch_size, epochs)

model.save_weights(emb_param)          # 学習済みパラメーターセーブ

score = model.evaluate(x_train.reshape(row, maxlen),
             np_utils.to_categorical(t_train, output_dim), batch_size=batch_size, verbose=1)

print("score:", score)
print()