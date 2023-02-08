import numpy as np
import tensorflow as tf
import sys
import gc
import pickle
import os
#from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_get import make_alldata_file
from data_get import make_alldata_file_withBOW
#from read_skipgram import get_skip_vec
#Skip_gram_dim = 200 # 国語研

'''
3layer neural network
'''

#np.random.seed(0)
#tf.set_random_seed(1234)

#---------------------------
# main
#--------------------------
if __name__ == '__main__':
# 学習  bow train.feature (長さそのまま) + 長さ10
# テスト bow

    train_pdfile = "../data/feature/train.pdfeature"
    test_pdfile = "../data/feature/test.pdfeature"

    train_bwfile = "../data/feature/train.feature"
    test_bwfile = "../data/feature/test.feature"

    id2wd_file =  "../data/feature/id2wd.pickle"
    wd2id_file =  "../data/feature/wd2id.pickle"
    epochs = 10
    batch_size = 10
    hidden_state_num = 512

    '''
    データの生成
    '''
    X_tr, T_tr, X_test, T_test, X_tr_txt, T_tr_txt, X_test_txt, T_test_txt, valid_inputs, valid_targets, X_max_len, T_max_len =  make_alldata_file(train_pdfile,test_pdfile)
    #bag-of-words vector
    X_trbw, X_testbw =  make_alldata_file_withBOW(train_bwfile,test_bwfile,id2wd_file)
    X_trbw = np.array(X_trbw, dtype=np.float)
    X_testbw = np.array(X_testbw, dtype=np.float)
    bow_vocabulary = X_trbw.shape[1]
    print("bow_vocabulary",bow_vocabulary)


    #from tensorflow.keras import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense
    #from tensorflow.keras.utils.np_utils import to_categorical
    from tensorflow.keras.utils import to_categorical
    #from keras.utils import np_utils
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    T_trbw = np.array(T_tr).flatten()
    T_trbw_oh = to_categorical(T_trbw)
    T_testbw = np.array(T_test).flatten()
    T_testbw_oh = to_categorical(T_testbw)
    #print("num of X_trbw",len(X_trbw[0]))
    print("num of T_tr",len(T_trbw))
    print("num of X_test_txt", len(X_test_txt))
    print("num of T_test_txt", len(T_test_txt))
    print("num of X_test", len(X_test))
    #print("num of T_test", T_testbw)

    # 学習のためのモデルを作る
    model = Sequential()
    # 全結合
    #model.add(Dense(input_dim=bow_vocabulary,output_dim=hidden_state_num))
    model.add(Dense(hidden_state_num))
    # 非線形関数にRelu
    model.add(Activation("relu"))
    # 最終出力層はクラスの数
    model.add(Dense(3))
    # 最終出力の関数(softmax関数)
    model.add(Activation("softmax"))
    # モデルをコンパイル
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # 悪い方のモデル
    #model.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])

    # 学習を実行
    model_filename = '../data/model/layer_bow.model'
    history= model.fit(X_trbw, T_trbw_oh,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(X_testbw, T_testbw_oh),
             callbacks = [
                  EarlyStopping(patience=5, monitor='val_acc', mode='max'),
                  ModelCheckpoint(monitor='val_acc', filepath=model_filename, mode='max', save_best_only=True)
              ]) #validation_split=0.15)

    #results = model.predict_proba(X_testbw) #確率で出力
    results = model.predict(X_testbw, batch_size=2) #クラスで出力
    # 結果を表示
    print("Predict:\n", results)
    loss, metrics = model.evaluate(X_testbw, T_testbw_oh)
    print("Xtest",X_test_txt)
    for crrct, out, inp in zip(T_testbw, results, X_test_txt):
        output_string = " ".join(inp)
        vector = ",".join([str(crrct),str(out),output_string])
        print(vector)
    print("accuracy",metrics)
