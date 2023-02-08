# coding: UTF-8

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold as SKF

def main():
    # irisでやる
    iris = load_iris()

    # svmで分類してみる
    svm = SVC(C=3, gamma=0.1)

    # 普通の交差検証
    trues = []
    preds = []
    for train_index, test_index in SKF().split(iris.data, iris.target):
        svm.fit(iris.data[train_index], iris.target[train_index])
        trues.append(iris.target[test_index])
        preds.append(svm.predict(iris.data[test_index]))
        
    # 今回の記事の話題はここ
    print("iris")
    print(classification_report(np.hstack(trues), np.hstack(preds), 
                                target_names=iris.target_names))

    print("iris=",iris.target_names)
    print("nps=",np.hstack(trues))
if __name__ == "__main__":
    main()
