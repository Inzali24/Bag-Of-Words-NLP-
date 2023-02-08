import sys
import re

p = re.compile(r"\((.+)\)")

def readfile(filepath):
   fp = open(filepath, "r")
   lines = fp.readlines()  #全行を読み込む list型
   fp.close()
   #list型内で改行をとる
   # return ここでの変数 # scalaと違って returnは省略できない(returnがなくてもエラーは出ないので注意!!)
   out_lines = [x.rstrip() for x in lines]  # 単にstripだと前の半角が消える(mecabのとき失敗する)
   return out_lines

def getwords(lines):
  out_lines = [x for x in lines if x != '']
  output = []
  for st in out_lines:
    m = p.search(st)
    if m:
      #print(m.group(0))
      output.append(m.group(0))
    else:
      pass
  return output

def make_sentence(lines):
  outputs = []
  output = ""
  for st in lines:
    if("(ID" in st):
      outputs.append(output)
      output = "" #リセット
    elif("*" in st):
      #print("out",st)
      pass
    else:
      st.replace('(','')
      pos, surface = st.split(" ")
      surface = surface.replace(')','')
      output += surface
  return outputs

if __name__ == '__main__':
   #argvs = sys.argv
   dir = "data"
   flist = ['mori-1912.pos','akutagawa-1921.pos','edogawa-1929.pos']
   #file = argvs[1]
   file = dir + '/' + flist[2]
   lines = readfile(file)
   lines = getwords(lines)
   lines = make_sentence(lines)
   for line in lines:
        #ここでぷりんとなど
        print (line)
