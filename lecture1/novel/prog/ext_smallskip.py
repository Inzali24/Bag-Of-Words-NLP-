import numpy as np
import sys
import os
import gc
import pickle
Skip_gram_dim = 200 # 国語研

def extract_vectors(wd2id_pk, wd_vect):
    """最小限のskip vectors のみ取り出す

    """
    sm_dict = {}
    with open(wd2id_pk, mode='rb') as fp:
        wd2id = pickle.load(fp)
        for wd in wd2id.keys():
            print("pickle_wrd=",wd)
            #nwjc2vecにあれば持って来る。無いときはなし
            if wd in wd_vect: #skip 辞書にある
                sm_dict.update({wd:wd_vect[wd]})
                print("wdok=",wd)
    return sm_dict # ["単語":[0.00234, -0.0024...]]

def load_skip_file(skip_file):
    #load nwjc
    fp = open(skip_file, 'r')
    wd_vect = {}
    # ここで巨大な辞書を作成する
    for line in fp:
        wd_lv = line.strip().split(" ")
        if(len(wd_lv) < 3):
            continue
        head_wd = wd_lv[0]
        vector = wd_lv[1:]
        wd_vect.update({head_wd:vector})

    fp.close()
    return wd_vect

# main
if __name__ == "__main__":
    """
    ここがmain 関数
    """
    #skip path
    skip_file = '/qnap6v1/qhome/rsc2/nwjc2vec/skip_fasttext/nwjc_word_0_200_8_25_0_1e4_6_1_0_15/nwjc_word_0_200_8_25_0_1e4_6_1_0_15.txt.vec'
    wd2id_pk = '../data/feature/wd2id.pickle'

    #outputfile
    output_file = "../data/skfeature/skip_base.vec"

    #skip-gram全部
    wd_vect = load_skip_file(skip_file)
    small_wdskip = extract_vectors(wd2id_pk, wd_vect)
    #ファイルに出力
    fout = open(output_file,'wt')
    for key, value in small_wdskip.items():
        value_st = " ".join(value)
        outline = " ".join([key,value_st]) + "\n"
        fout.write(outline)
        print(outline)
    fout.close()

