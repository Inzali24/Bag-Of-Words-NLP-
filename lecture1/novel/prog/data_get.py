# dataを取り出す。

import numpy as np
import sys
import os
import pickle

from read_skipgram import get_skip_vec


'''
データを読む
'''
def load_data_file(train_file,test_file):
    #必要なパラメータの整理
    # division 番号
    input_train_file = train_file  #= input_file_path + 'learn' + division
    input_test_file  = test_file   #= input_file_path + 'test' + division

    #ファイルの読み込み
    return load_data_core(input_train_file,input_test_file)

def load_data_core(train_file,test_file):
    train_X, train_T  = extract_data_from_file(train_file)
    test_X, test_T = extract_data_from_file(test_file)
    return train_X, train_T, test_X, test_T

#not used
'''
データを読んでvocabularyを作る
'''
def load_data(input_file_path,division):  ###input_data_dir,input_data_idiom_num):
    #必要なパラメータの整理
    # division 番号
    input_train_file = input_file_path + 'learn' + division
    input_test_file = input_file_path + 'test' + division

    #ファイルの読み込み
    return load_data_core(input_train_file,input_test_file)

def load_data_core(train_file,test_file):
    train_X, train_T  = extract_data_from_file(train_file)
    test_X, test_T = extract_data_from_file(test_file)
    return train_X, train_T, test_X, test_T


'''
データを取り出す(defaultカンマ区切り)
'''
def extract_data_from_file(file,delimiter=","):
    input_X = []
    target_T = []
    for line in open(file, encoding="utf8"):
        line = line.strip()
        line_contents = line.split(delimiter)
        label = line_contents[0]
        wdsX = line_contents[1:]
        #print("label=",label)
        #print("wdsX=",wdsX)
        input_X.append(wdsX)
        target_T.append([label])

    return input_X, target_T

def make_vocabulary_from_stored_data(train_input_X,train_target_T,test_input_X,test_target_T):
    # 入力単語ベクトルの全要素集合作成
    xx = []
    xx.append('unk') # for unknown wordsつまり uknownは０番
    for ss in  train_input_X + test_input_X:  #unk does not exit
        xx.extend(ss)

    # 出力タグの全集合作成
    # 学習とテストデータのタグ集合全部入れる．
    # 知らないタグを出力することは無い
    tt = []

    for ss in  train_target_T + test_target_T:
        tt.extend(ss)
    return xx, tt

#あとで使うために学習時のvocabを保存
#testの読み込み
def read_vocabulary_from_file_or_data(input_file_path,train_X, train_T, test_X, test_T, division):
    ###########################
    # read vocabulary of inputs
    ###########################
    #### read or renew the vocab & tag data
    read_flag = False
    choice = input("Do you want to read vocab file (R) or rewrite vocab? (W)")
    if choice in ['R','r']:  #read
        read_flag = True
    else:   # write the vocab
        read_flag = False

    valid_inputs_fileout = input_file_path + "vocaball" + division + ".txt"
    if(os.path.exists(valid_inputs_fileout) and read_flag):
        print("read vocaball.txt from",validinputs_fileout)
        for line in open(valid_inputs_fileout):
            valid_inputs.append(line.strip())
    else:
        if read_flag:
            print("vocab file does not exit, then write vocab file")
        print("write vocab_file.txt to=",valid_inputs_fileout)
        xx, tt = make_vocabulary_from_stored_data(train_X, train_T, test_X, test_T)
        valid_inputs = list(set(xx))
        with open(valid_inputs_fileout,'w') as fw:
            for xx in valid_inputs:
                fw.write(xx)
                fw.write('\n')
            fw.close

    ############################
    #  read target tags from file if the file exits
    ############################
    valid_targets_fileout = input_file_path + "tagall" + division +  ".txt"
    # output target tags to a file
    if(os.path.exists(valid_targets_fileout) and read_flag ):
        print("read target_all.txt from",valid_targets_fileout)
        for line in open(valid_targets_fileout):
            valid_targets.append(line.strip())
    else:
        if read_flag:
            print("target file does not exit, then write target file")
        else:
            print("write tag_vocabulary.txt")
        valid_targets = list(set(tt))
        with open(valid_targets_fileout,'w') as fw:
            for tt in valid_targets:
                fw.write(tt)
                fw.write('\n')
                fw.close

    return valid_inputs, valid_targets

#単にvocabを作る
def make_vocabulary(train_X, train_T, test_X, test_T):

    xx, tt = make_vocabulary_from_stored_data(train_X, train_T, test_X, test_T)
    valid_inputs = list(set(xx))
    valid_targets = list(set(tt))
    return valid_inputs, valid_targets


'''
一回だけ呼ばれる。この時にいろいろ指定しても良い。
vocabularyをどこに保存するかなど
'''
def make_alldata():
    input_data_dir = '../data/gru/feature'
    input_data_idiom_num = '0016'
    input_file_path = input_data_dir + '/' + input_data_idiom_num + '/'
    division = '0'

    train_X, train_T, test_X, test_T = load_data(input_file_path,division)
    #valid_inputs, valid_targets = read_vocabulary_from_file_or_data(input_file_path,train_X, train_T, test_X, test_T, division)
    valid_inputs, valid_targets = make_vocabulary(train_X, train_T, test_X, test_T)

    train_num_X = []
    train_num_T = []
    test_num_X = []
    test_num_T = []

    #train_txt_X = ['','彼','に','','','','']といったデータ
    for train_txt_X, train_txt_T in zip(train_X, train_T):
        train_num_X.append(text2num(train_txt_X,valid_inputs))
        train_num_T.append(text2num(train_txt_T,valid_targets))

    for test_txt_X, test_txt_T in zip(test_X, test_T):
        test_num_X.append(text2num(test_txt_X,valid_inputs))
        test_num_T.append(text2num(test_txt_T,valid_targets))

    #check
    #for train_txt_X, train_num_l_X in zip(train_X, train_num_X):
    #    print(train_txt_X)
    #    print(train_num_l_X)

    max_input_len = max(map(len, train_num_X+test_num_X))
    max_target_len = max(map(len, train_num_T+test_num_T))
    #print ("max_length=",max_input_len)
    #print ("max_target=",max_target_len)

    return train_num_X, \
           train_num_T, \
           test_num_X, \
           test_num_T, \
           train_txt_X, \
           train_txt_T, \
           test_txt_X, \
           test_txt_T, \
           valid_inputs, \
           valid_targets , \
           max_input_len, \
           max_target_len \

''' =============================================
    sequential file + BOW  file 2019/6 
    ==============================================='''
def make_alldata_file_withBOW(train_bw,test_bw,id2wd_file):
    # make_alldata_fileとともに使う．BOWファイルを読んで渡す
    #時系列データは vocabはBOWと違うが別のベクトル体系なので関係無し
    train_Xbws, _  =  read_feature_file(train_bw)
    test_Xbws, _ = read_feature_file(test_bw)
    train_Xbw = make_feature_space(train_Xbws,id2wd_file)
    test_Xbw = make_feature_space(test_Xbws,id2wd_file)

    # ベクトルの表示
    ''' 
    print(train_Xbws[0])
    out = ''
    for x in train_Xbw[0]:
        out += str(x)
    print(out)
    '''
    # batch x vocab (bag-of-words)
    return train_Xbw, test_Xbw

# read vectors from vectored file
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

    return listx, listy

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


'''
一回だけ呼ばれる。この時にいろいろ指定しても良い。
fileを指定して、データを取り出す
'''
def make_alldata_file(train_file,test_file):
    train_X, train_T, test_X, test_T = load_data_file(train_file,test_file)
    valid_inputs, valid_targets = make_vocabulary(train_X, train_T, test_X, test_T)

    train_num_X = []
    train_num_T = []
    test_num_X = []
    test_num_T = []

    #train_txt_X = ['','彼','に','','','','']といったデータ
    for train_txt_X, train_txt_T in zip(train_X, train_T):
        train_num_X.append(text2num(train_txt_X,valid_inputs))
        train_num_T.append(text2num(train_txt_T,valid_targets))
        #出力: [12,35,68,6,289] といった数字化したデータ

    for test_txt_X, test_txt_T in zip(test_X, test_T):
        test_num_X.append(text2num(test_txt_X,valid_inputs))
        test_num_T.append(text2num(test_txt_T,valid_targets))

    max_input_len = max(map(len, train_num_X+test_num_X))
    max_target_len = max(map(len, train_num_T+test_num_T))
    print ("max_length=",max_input_len)
    print ("max_target=",max_target_len)

    return train_num_X, \
           train_num_T, \
           test_num_X, \
           test_num_T, \
           train_txt_X, \
           train_txt_T, \
           test_X, \
           test_T, \
           valid_inputs, \
           valid_targets , \
           max_input_len, \
           max_target_len \


#テキストを数字に変える（配列版）
def text2num(text,vocabulary):
    #unknown処理
    num_input_unk = []
    for c in text:
        if c in vocabulary:
            num_input_unk.append(vocabulary.index(c))
        else:
            num_input_unk.append(vocabulary.index('unk'))
    return num_input_unk
#テキストを数字に変える（1単語）
def test2num_one(text,vocabulary):
    if text in vocabulary:
        return vocabulary.index(text)
    else:
        return vocabulary.index('unk')


if __name__ == '__main__':
    train_pdfile = "../data/feature/train.pdffeature"
    test_pdfile = "../data/feature/test.pdfeature"

    train_bwfile = "../data/feature/train.feature"
    test_bwfile = "../data/feature/test.feature"

    id2wd_file =  "../data/feature/id2wd.pickle"
    make_alldata_file_withBOW(train_bwfile,test_bwfile,id2wd_file)
