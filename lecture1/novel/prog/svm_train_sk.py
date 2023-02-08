"""
単語のskip-gramの平均ベクトルで学習

"""
# basic
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import accuracy_score

def svm_exe(input_vectors,labels):
    #特徴量 [事例 x ベクトル]
    X = input_vectors
    #ラベル [事例 x ラベル]
    y = labels

    #print("x",X)
    #print("y",y)

    # 線形SVMのモデル
    model = SVC(kernel='linear', random_state=None)

    # データによる学習はfit関数を使う
    model.fit(X, y)

    pred_x = model.predict(X)
    accuracy = accuracy_score(y, pred_x)
    print('acccuracy= %.3f' % accuracy)
    return model


def read_feature_file(input_file):
    #ファイルの読み込み
    df = pd.read_csv(input_file,sep='\s+') #DataFrame型
    #X  csvで0から数えて1から200列目まで
    x = df.iloc[:,1:]
    #listに変換
    listx = x.values.tolist()
    #print('xx',listx[0])

    #y ラベルの取り出し
    y = df.iloc[:,0]
    #listに変換
    listy = y.values.tolist()
    #print('y',listy)

    return df, listx, listy


if __name__ == "__main__":
    """
    ここがmain 関数
    """
    #入力のファイル
    train_file = "../data/skfeature/train.skfeature"
    #出力のファイル
    df, listx, listy = read_feature_file(train_file)

    model = svm_exe(listx, listy)

    #学習したモデルを保存
    filename = "../data/model/sk_svm_model.dat"
    print("学習したモデルを保存",filename)
    pickle.dump(model, open(filename, 'wb'))
