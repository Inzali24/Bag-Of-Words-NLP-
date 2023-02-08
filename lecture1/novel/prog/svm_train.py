"""
単語のbag of words 特徴量で学習

"""
# basic
import sys
import pickle
#import pandas as pd
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

def make_feature_space(listx,id2wd_file):
    """
    1436:1 のような座標軸を
    配列で表現
    """
    #最大座標値を辞書から取り出す
    id2wd = {}
    with open(id2wd_file,'rb') as f:
      id2wd = pickle.load(f)
    keys_list = id2wd.keys()
    max_axis = 0
    for id in keys_list:
        axis = int(id)
        if (axis > max_axis):
            max_axis = axis
    if(max_axis == 0):
        print("error in max_axis=0")
        sys.exit(0)
    num_axis = max_axis + 1
    #print(keys_list)
    print("num_axis=",num_axis)

    output_vectors = []
    for line in listx:
        bag_of_words = [0] * num_axis
        axis_values = [x.split(':') for x in line]
        for axvl in axis_values:
            axis = int(axvl[0])  #座標軸
            value = int(axvl[1]) #そのときの値
            #print("ax, value", axis,value)
            bag_of_words[axis] = value
        output_vectors.append(bag_of_words)
    #print ('output',output_vectors[0])
    return np.array(output_vectors)

def read_feature_file(input_file):
    #ファイルの読み込み
    # 長さが不揃いだと pandasでは読めない
    #普通は下記のような感じ (列が揃ってる必要がある)
    #df = pd.read_csv(input_file,sep='\s+') #DataFrame型
    fp = open(input_file, "r")
    lines = fp.readlines()  #全行を読み込む list型
    fp.close()
    #改行削除
    out_lines = [x.rstrip() for x in lines]
    #半角スペース区切り
    outputs = [x.split(" ") for x in out_lines]

    listx = [x[1:] for x in outputs] #ベクトル部分だけ取り出す
    #print('xx',listx)

    #y ラベルの取り出し
    listy = [x[0] for x in outputs]
    #余計なカッコを削除
    listy = np.array(listy)
    listy = np.squeeze(listy)
    #print('yy',listy)

    return outputs, listx, listy


if __name__ == "__main__":
    """
    ここがmain 関数
    """
    #入力のファイル
    train_file = "../data/feature/train.feature"
    id2wd_file = "../data/feature/id2wd.pickle"
    #出力のファイル
    df, listx, listy = read_feature_file(train_file)

    feature_vector = make_feature_space(listx,id2wd_file)
    print("shape",feature_vector.shape)
    model = svm_exe(feature_vector, listy)

    #学習したモデルを保存
    filename = "../data/model/svm_model.dat"
    print("学習したモデルを保存",filename)
    pickle.dump(model, open(filename, 'wb'))
