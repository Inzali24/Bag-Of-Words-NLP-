import sys

#各小説を読み込んで
# 「クラス,文」の形のテキストに出力する

def readfile(key, filepath):
   fp = open(filepath, "r")
   lines = fp.readlines()  #全行を読み込む list型
   outputs = []
   fp.close()
   #list型内で改行をとる
   # return ここでの変数 # scalaと違って returnは省略できない(returnがなくてもエラーは出ないので注意!!)
   out_lines = [x.rstrip() for x in lines]  # 単にstripだと前の半角が消える(mecabのとき失敗する)
   for sentence in out_lines:
    key_sentence = [key,sentence]
    outputs.append(key_sentence)
   return outputs

if __name__ == '__main__':
   #argvs = sys.argv
   dir = "data"
   flist = {'m':'mori.txt','a':'akutagawa.txt','e':'edogawa.txt'}
   for k in flist.keys():
     file = dir + '/' + flist[k]
     lines = readfile(k,file)
     wfile = dir + '/' + k + '.txt'
     with open(wfile, mode='w') as f:
       for line in lines:
         output_string = line[0]+ ',' + line[1]
         f.write(output_string)
         f.write("\n")

     print(lines)


   #for line in lines:
   #     print (line)
