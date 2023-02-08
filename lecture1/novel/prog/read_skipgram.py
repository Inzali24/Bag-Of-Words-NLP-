# skipgramを読みこんで，必要な単語ベクトルだけ取り出して保存

import numpy as np
import sys
import os
import pickle

skip_data = '/qnap6v1/qhome/rsc2/nwjc2vec/skip_fasttext/nwjc_word_0_200_8_25_0_1e4_6_1_0_15/nwjc_word_0_200_8_25_0_1e4_6_1_0_15.txt.vec'
#skip_data = '../data/skfeature/skip_base.vec'

allwd_vec = {}

def get_skip_vec(skip_data):
    vect = {}
    print("read skip data...",file=sys.stderr)
    fp = open(skip_data, "r")
    for line in fp:
        line = line.strip()
        splitted = line.split(' ')
        if(len(splitted) < 3): #最初の行なので飛ばす
            continue
        else:
           word = splitted[0] #String
           skip_vect = splitted[1:] # Array[String]
           vect[word]=skip_vect
    fp.close()
    print("finish reading vector file num=",len(vect),file=sys.stderr)
    return vect

def get_vector(allwds,skip_vect):
    all_vec = {}
    for word in allwds:
        if word in skip_vect:
          #ここで文字列の'-0.21108'を数値に変える
          all_vec[word]=[float(x) for x in skip_vect[word]]
    return all_vec



# 単語登録されているskipだけ取り出して記録する
if __name__ == '__main__':
    wd2id_file = "../data/feature/wd2id.pickle" #計算対象のdir
    #保存先
    serialize_file = "../data/skfeature/all_skip_dict.pickle"
    text_file = "../data/skfeature/all_skip_dict.txt"
    #全nwjc2vecを読み込む
    skip_vect = get_skip_vec(skip_data)
    #辞書を読み込む
    wd2id = {}
    with open(wd2id_file,'rb') as f:
      wd2id = pickle.load(f)
    allwords_list = wd2id.keys()
    #各単語のベクトルを取り出す (辞書)
    allwords_vec = get_vector(allwords_list,skip_vect)

    #分散表現ベクトルの量
    '''
    print("keynum=",len(allwords_vec.keys()))
    #保存
    with open(serialize_file, mode='wb') as f:
        pickle.dump(allwords_vec,f)
    print("saving is finished",serialize_file, file=sys.stderr)
    '''
    #テキスト版の保存
    with open(text_file, mode='w') as f:
       words = allwords_vec.keys()
       for word in words:
           vect_str = [str(x) for x in allwords_vec[word]]
           out_text = word + ' ' + ' '.join(vect_str) + "\n"
           f.write(out_text)
    print("saving is finished",serialize_file, file=sys.stderr)
