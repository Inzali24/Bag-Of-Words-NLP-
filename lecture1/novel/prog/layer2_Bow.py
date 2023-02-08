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
3 layer neural network
'''

np.random.seed(0)
tf.random.set_seed(1234)

# 3layer neural network
def inference_3lay(xbw, n_batch, is_training, vocabulary, bow_vocabulary, maxlen=None, n_hidden=256, n_out=None):

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    #===== simple 2layer neural ======
    #V = weight_variable([bow_vocabulary, n_out]) # 2 layer vocab x 256
    #c = bias_variable([n_out])
    #yl = tf.matmul(xbw, V) + c

    #===== simple 3layer neural =======
    V1 = weight_variable([bow_vocabulary, n_hidden]) # 2 layer vocab x 256
    V2 = weight_variable([n_hidden, n_out]) # 128 x 3
    c1 = bias_variable([n_hidden])
    c2 = bias_variable([n_out])
    y1 = tf.tanh(tf.matmul(xbw, V1) + c1)
    yl = tf.matmul(y1, V2) + c2  # 線形活性

    y = tf.nn.softmax(yl)  # softmax

    return y #, assign_op, x_placeholder


def loss(y, t):  # 0が入っても大丈夫なようにしている
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       reduction_indices=[1]))
    return cross_entropy

def loss_2d(y, t):  # 2次元のt batch x [0, 1]  y batch x 2次元
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
            tf.cast(t,tf.float32) * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))
    return cross_entropy


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(t, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# [batch,出力カテゴリ数]
#def get_final_tag(y): #出力ユニットをもらって最大値をもらう
#    #[batch] 正解配列番号
#    predicted_args = tf.argmax(y, -1)
#    return valid_targets[predicted_args] #valid_targtesはもらないと無理


# ここが学習
# ２回、３回と呼び出して同じデータに対して計算する
# X_train [batch x time]
def many_learn_and_test(X_train, X_trbw, T_train, X_test_oh, X_testbw, T_test_oh, n_batches, epochs=100,batch_size=20):
#def many_learn_and_test(X_train, T_train,X_test_nm, T_test_oh, n_batches, embedding, epochs=100,batch_size=20):
    print("start initlization",file=sys.stderr)
    init = tf.global_variables_initializer()
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2) #0.2
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
    sess.run(init)

    # skip-gramのデータを読み込む (one-hotでは不要)
    #sess.run(assign_op,feed_dict={x_placeholder: embedding})
    #1epoch で全データ学習
    for epoch in range(epochs):

        X_bw, T_ = shuffle(X_trbw, T_train)  #シャッフルするので端数も学習される
        #X_, X_bw, T_ = shuffle(X_train, X_trbw, T_train)  #シャッフルするので端数も学習される
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            #print("start end=",start,end)
            #print("x=",len(X_[start:end]))
            sess.run([train_step], feed_dict={
                #x:   X_[start:end],
                xbw: X_bw[start:end],
                t:   T_[start:end],
                n_batch: batch_size,
                is_training: True
            })

    #test データの出力
    #まずone-hotに変換
    count=0

    # 精度
    test_acc,predicted = sess.run([acc,y], feed_dict={
        #x: X_test_oh,
        xbw: X_testbw,
        t: T_test_oh,
        n_batch: len(X_test_oh), #N_validation,
        is_training: False
    })
    print("acc for test=",test_acc,file=sys.stderr)

    #最後の正解タグ列(one_hot  => 数字) e.g. [0, 3.5 ,0.5 , 1] => [2] としている
    output_tag = np.argmax(predicted, -1) # [batch]
    t_text_num = np.argmax(T_test_oh, -1) # [batch]
    #t_text_num = T_test_nm

    #ここでは one-hotでなく数字  [batch] で比較
    for correct_nm, out_nm in zip (t_text_num, output_tag):
        if(correct_nm == out_nm):
            count += 1
    all = len(t_text_num)
    rate = float(count) / float(all)
    print("final rate = %f (%d/%d)" % (rate, count, all),file=sys.stderr)
    return rate, output_tag #テストの正解率, 出力タグ列(batch x 1-string)
#--- end of many_learn_


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
    hidden_state_num = 128
    #serialized_file = "./myidembedding.pickle" #embeddingの保存先


    '''
    データの生成
    '''
    X_tr, T_tr, X_test, T_test, X_tr_txt, T_tr_txt, X_test_txt, T_test_txt, valid_inputs, valid_targets, X_max_len, T_max_len =  make_alldata_file(train_pdfile,test_pdfile)
    #bag-of-words vector
    X_trbw, X_testbw =  make_alldata_file_withBOW(train_bwfile,test_bwfile,id2wd_file)
    X_trbw = np.array(X_trbw, dtype=np.float64)
    X_testbw = np.array(X_testbw, dtype=np.float64)
    bow_vocabulary = X_trbw.shape[1]

    # print_for_check
    print("bow_vecabr",bow_vocabulary)

    #print("num of X_tr",len(X_tr))
    #print("num of X_test_txt", len(X_test_txt))
    #print("num of T_test_txt", len(T_test_txt))
    #print("num of X_test", len(X_test))
    #print("num of T_test", len(T_test))

    X_vocabulary = len(valid_inputs)
    T_vocabulary = len(valid_targets)
    print("vacab X=", X_vocabulary)
    print("vacab T=", T_vocabulary)

    X  = np.zeros((len(X_tr), X_max_len, X_vocabulary), dtype=np.integer)
    Xn = np.array(X_tr, dtype=np.integer)
    T  = np.zeros((len(T_tr), T_vocabulary), dtype=np.integer)
    Tn = np.array(T_tr, dtype=np.integer)

    N = len(X_tr)  #学習用データの数（trainingとvalid）数字に変換
    for i in range(N):
        for t, num in enumerate(X_tr[i]):
            X[i, t, num] = 1
        for t, num in enumerate(T_tr[i]):
            T[i, num] = 1   #t =0 のみなので不要とする

    #testデータ用のone-hot vector
    X_test_oh = np.zeros((len(X_test), X_max_len, X_vocabulary), dtype=np.integer)
    T_test_oh = np.zeros((len(T_test), T_vocabulary), dtype=np.integer)
    X_test_nm = np.array(X_test,dtype=np.integer)
    T_test_nm = np.array(T_test,dtype=np.integer)
    for i in range(len(X_test)):
        for t, num in enumerate(X_test[i]):
            X_test_oh[i, t, num] = 1
        for t, num in enumerate(T_test[i]):
            T_test_oh[i, num] = 1
    #
    # 学習データ準備
    #
    N_train = len(X)  #学習データ
    n_batches = N_train // batch_size

    #print("parameter")
    print("epochs",epochs,file=sys.stderr)
    print("batch_size=",batch_size,file=sys.stderr)
    #print("N=",N)
    #print("n_batches=",n_batches)

    '''
    X = [question  time  encode]   encodeはone-hot
    Y = [question  encode]
    input_digits 入力長
    output_digits 出力長
    '''

    '''
    モデル設定
    '''
    n_in = X_vocabulary #xinput 語彙数
    n_hidden = hidden_state_num #128
    n_out = 3           #target 語彙数

    #x = tf.placeholder(tf.float32, shape=[None, X_max_len, n_in]) #no_emb
    xbw = tf.placeholder(tf.float32, shape=[None,  bow_vocabulary]) #bag-of-words
    t = tf.placeholder(tf.int32, shape=[None, n_out])  # tの次元は batch x n_out
    n_batch = tf.placeholder(tf.int32, shape=[])
    is_training = tf.placeholder(tf.bool)

    #最後の答えはy
    #y  = inference_lstm(x, xbw, n_batch, is_training, valid_inputs, bow_vocabulary,
    #                    maxlen=X_max_len, n_hidden=n_hidden, n_out=n_out)

    y  = inference_3lay(xbw, n_batch, is_training, valid_inputs, bow_vocabulary,
                        maxlen=X_max_len, n_hidden=n_hidden, n_out=n_out)
    print("y====",y)  #[?,2]
    print("t====",t)  #[?,2]
    mloss = loss_2d(y, t) #tの次元は2
    train_step = training(mloss)

    acc = accuracy(y, t)

    history = {
        'val_loss': [],
        'val_acc': [],
        'test_acc': []
    }

    # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
    # ================ end graph =============================================
    '''
    モデル学習
    '''
    #embedding = get_skip_gram_vocab(len(valid_inputs),serialized_file)
    rates = np.array([])
    out_results = []
    for k in range(3):
        rate, out_num = many_learn_and_test(X, X_trbw, T, X_test_oh, X_testbw, T_test_oh, n_batches, epochs ,batch_size)
        # TとT_test_ohは one-hot。XnとX_test_nmは 数字。[0,0,1,0,0..]か [2]かの違い
        #rate, out_num = many_learn_and_test(Xn, T, X_test_nm, T_test_oh, n_batches, embedding, epochs ,batch_size)
        rates = np.append(rates,rate)
        out_result = [valid_targets[i] for i in out_num] #one hotを文字に変える
        out_results.append(out_result)
    #best の結果
    best_index = np.argmax(rates,axis=0)
    out_best = out_results[best_index]
    #print("X=",X_test)
    #print("T=",T_test)
    #print("best_index=",best_index)
    #print("out",out_best)


    for crrct, out, inp in zip(T_test_txt, out_best, X_test_txt):
        #ここで crrct は [[I]] など出力が配列扱いなので1つ減らす
        output_string = " ".join(inp)
        vector = ",".join([crrct[0],out,output_string])
        print(vector)
    print("rates",rates)
