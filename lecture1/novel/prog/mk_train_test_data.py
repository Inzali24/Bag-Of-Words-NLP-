import sys
import random

#各小説を読み込んで
# 学習データとテストデータを作成する

def readfile(filepath):
   fp = open(filepath, "r")
   lines = fp.readlines()  #全行を読み込む list型
   fp.close()
   out_lines = [x.rstrip() for x in lines]
   return out_lines

def writefile(data_lines,wpath):
   with open(wpath, mode='w') as f:
     for line in data_lines:
       f.write(line)
       f.write("\n")


if __name__ == '__main__':
   #argvs = sys.argv
   dir = "../data"
   flist = ['m.txt','a.txt','e.txt']
   data_lines = []
   for file in flist:
     file_path = dir + '/' + file
     lines = readfile(file_path)
     data_lines.extend(lines)
   print(data_lines)
   random.shuffle(data_lines)
   tarining = 0.8
   tr_num = int(len(data_lines) * 0.8)
   train_data = data_lines[:tr_num]
   test_data = data_lines[tr_num:]
   print(train_data)
   print(test_data)

   wpath = dir + '/train.txt'
   writefile(train_data,wpath)
   wpath = dir + '/test.txt'
   writefile(test_data,wpath)
