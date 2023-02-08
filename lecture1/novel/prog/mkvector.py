"""
データを読んで，単語を特徴量として取り出す

"""
# Standard Library
import os
import sys
#import re
import pickle
#import pandas as pd
import MeCab

# カテゴリのid
#芥川:0  江戸川:1　森:2
cat2id = {"a":0, "e":1, "m":2}

m = MeCab.Tagger()# ("-d ../mecab-ipadic-neologd")

def make_vector(base_data,id2wd,wd2id):
    out_vectors = []
    for line in base_data:
        label = line[0] #正解ラベル
        sentence = line[1] #入力文
        mecab_output = m.parse(sentence)
        #単語のリスト
        base_wd = get_base_word(mecab_output)

        print(base_wd)
        wdids = [wd2id[x] for x in base_wd]
        cat_id = cat2id[label] #正解ラベルを数字に変換
        # axis:1 の形式に変換
        wd_vectors = [str(x) + ":1" for x in wdids] #文字連結
        tmp_array = []
        tmp_array.append(str(cat_id))
        tmp_array.extend(wd_vectors) #vector の追加
        out_vec = " ".join(tmp_array) #文字列
        print(out_vec)
        out_vectors.append(out_vec)
    return out_vectors


def read_file(filepath):
    """ 著者,テキスト
        output: [example x 2]
    """
    #df = pd.read_csv('input_file', sep=',')
    fp = open(filepath, "r")
    lines = fp.readlines()  #全行を読み込む list型
    fp.close()

    out_lines = [x.rstrip() for x in lines]
    outputs = [x.split(",") for x in out_lines]

    return outputs

def make_dictionary(input_files):
    all_wds = set()  #全単語集合
    all_wds.add("unk") #未登録語用
    for file in input_files:
      for line in file:
        label = line[0]
        sentence = line[1]
        mecab_output = m.parse(sentence)
        #単語のリストを得る
        base_wd = get_base_word(mecab_output)
        base_wd_set = set(base_wd) #重複を消す
        all_wds = all_wds | base_wd_set #辞書に追加
    #print(all_wds)
    #単語にidをつけて辞書にする
    id2wd = {}
    wd2id = {}
    for id, word in enumerate(list(all_wds)):
        id2wd[id]=word
        wd2id[word]=id
    #print(id2wd)
    return id2wd, wd2id

#mecabの出力から基本形を取り出す
def get_base_word(mecab_output):
    mecab_output = mecab_output.split('\n')
    mecab_output = mecab_output[:-2] #最後のEOSを消す
    #print(mecab_output)
    mecab_wds = [x.split('\t') for x in mecab_output]
    #print(mecab_wds)
    #基本形があるときは基本形をとりだし，ないときは表層形
    base_wds = []
    for surface, feature_st in mecab_wds:
        atts = feature_st.split(",")
        out = ""
        if (atts[6] == "*"):  #基本形が存在しない
            out = surface  #なので見出し語を入れる（辞書にない語）
        else:
            out = atts[6] #基本形
        base_wds.append(out)
    #print(base_wds)
    return base_wds #[list_of_words]

if __name__ == "__main__":
    """
    ここがmain 関数
    """
    #入力のファイル
    tain_csv = "../data/train.csv"
    test_csv = "../data/test.csv"
    #出力のファイル
    train_feature = "../data/feature/train.feature"
    test_feature = "../data/feature/test.feature"
    id2wd_outfile = "../data/feature/id2wd.pickle"
    wd2id_outfile = "../data/feature/wd2id.pickle"

    #mecab_pexpect = start_mecab()
    #ファイルの読み込み
    train_base = read_file(tain_csv) #[example x 2]
    test_base = read_file(test_csv)
    id2wd, wd2id = make_dictionary([train_base,test_base])

    # 各ファイルを保存
    with open(wd2id_outfile, 'wb') as fp:
        pickle.dump(wd2id, fp)
    with open(id2wd_outfile, 'wb') as fp:
        pickle.dump(id2wd, fp)

    # vectorを作成
    train_vec = make_vector(train_base,id2wd,wd2id)
    test_vec = make_vector(test_base,id2wd,wd2id)

    with open(train_feature,'wt') as fout:
        for line in train_vec:
            outline = line + "\n"
            fout.write(outline)

    with open(test_feature,'wt') as fout:
        for line in test_vec:
            outline = line + "\n"
            fout.write(outline)
