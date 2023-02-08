"""
イディオムの前後3単語の特徴量で学習したモデルでテスト

input_file のパス ../data/sfeature/
model のパス    ../data/model/

"""
# basic
import sys
import pickle
# Data & Machine learning
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import accuracy_score

#ベクトルの最大値
max_axis = 0

def svm_apply(model, input_vectors,labels,source_lines):
    #特徴量 [事例 x ベクトル]
    X = input_vectors
    #ラベル [事例 x ラベル]
    y = labels

    #print("x",X)

    #modelの適用
    pred_y = model.predict(X)
    print("correct, estimated")
    for correct, estimated, source in zip (y, pred_y, source_lines):
        print(correct, "\t", estimated, "\t", source)
    accuracy = accuracy_score(y, pred_y)
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

def get_num_axis(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    num_axis_str = lines[0].strip()
    num_axis = int(num_axis_str)
    return num_axis

if __name__ == "__main__":
    """
    ここがmain 関数
    """
    #入力のイディオムファイル
    #id2wd_file = "../data/feature/id2wd.pickle"
    input_file = "../data/skfeature/test.skfeature"
    input_source = "../data/test.csv"

    #データセット
    df, listx, listy = read_feature_file(input_file)
    #feature_vector = make_feature_space(listx,id2wd_file)
    #print("shape",feature_vector.shape)

    #モデルの読み込み
    model_file = "../data/model/sk_svm_model.dat"
    model = pickle.load(open(model_file, 'rb'))

    #特徴量を見るためのもとデータの確認
    f = open(input_source, 'r')
    source_lines = f.readlines()
    source_lines = [x.strip() for x in source_lines]

    #svmの適用
    svm_apply(model, listx, listy, source_lines)
