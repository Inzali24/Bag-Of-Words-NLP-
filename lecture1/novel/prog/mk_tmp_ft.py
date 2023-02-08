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

word_length = 10 #10単語に揃える

#長さ揃える
def my_padding(inarr,length):
    if len(inarr) >= length:
        return inarr[:length]
    else:
        for aa in range(length - len(inarr)):
          inarr.append('。')
        return inarr

def make_feature_space(listx,id2wd_file):
    """
    形を変える
    """
    #辞書を読む
    id2wd = {}
    with open(id2wd_file,'rb') as f:
      id2wd = pickle.load(f)
    keys_list = id2wd.keys()

    output_vectors = []
    for line in listx:
        axis_values = [x.split(':') for x in line]
        out_wds = []
        for axvl in axis_values:
            axis = int(axvl[0])  #座標軸
            value = int(axvl[1]) #そのときの値
            #単語に変換
            out_wds.append(id2wd[axis])
        #長さをword_lengthに揃える
        out_wd_pd = my_padding(out_wds,word_length)
        output_vectors.append(out_wd_pd)
        #print ('output',out_wd_pd)
    return output_vectors


def read_feature_file(input_file):
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
    test_file = "../data/feature/test.feature"
    id2wd_file = "../data/feature/id2wd.pickle"
    #出力のファイル
    train_ofile = "../data/feature/train.pdfeature"
    test_ofile = "../data/feature/test.pdfeature"


    #学習データベクトル
    df, listx, listy = read_feature_file(train_file)
    feature_vector = make_feature_space(listx,id2wd_file)
    with open(train_ofile,'wt') as fout:
        for label, feature in zip(listy,feature_vector):
            outst = str(label) + "," + ",".join(feature) + "\n"
            fout.write(outst)

    #テストデータベクトル
    df, listx, listy = read_feature_file(test_file)
    feature_vector = make_feature_space(listx,id2wd_file)
    with open(test_ofile,'wt') as fout:
        for label, feature in zip(listy,feature_vector):
            outst = str(label) + "," + ",".join(feature) + "\n"
            fout.write(outst)
