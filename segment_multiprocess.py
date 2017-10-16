import jieba
from wordsegment import segment
import os
from multiprocessing import Pool
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--worker", type=int, help="The process number", default = 4)
parser.add_argument("--flag", type=str, help="model flag", default = 'mul')
args = parser.parse_args()

jieba.enable_parallel(args.worker)
flag = args.flag
flagdir = os.mkdir(os.path.join('middleresult',flag))
if not os.path.exists(flagdir):
    os.mkdir(flagdir)

en_file = 'ai_challenger_translation_train_20170904/translation_train_data_20170904/train.en'
zh_file = 'ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
file_middle = 'middleresult/{}/segmented_train_seg_by_word.txt'.format(flag)
pickle_middle = 'middleresult/{}/segmented_train_seg_by_char.pkl'.format(flag)
zh_vocab = 'middleresult/{}/zh_vocab.txt'.format(flag)
en_vocab = 'middleresult/{}/en_vocab.txt'.format(flag)

train_chinese = []
train_english = []

with open(en_file,encoding='utf-8') as fhdl:
    for line in fhdl:
        train_english.append(line.strip())
with open(zh_file,encoding='utf-8') as fhdl:
    for line in fhdl:
        train_chinese.append(line.strip())
      
print(len(train_chinese),len(train_english))
from utils import *
pb = ProgressBar(worksum=len(train_chinese),auto_display=False)

pb.startjob()
train_token_chinese = []
train_token_english = []
zh_word_dic = {}
en_word_dic = {}
num = 0

def write_words_to_file():
    with open(zh_vocab,'w',encoding='utf-8') as whdl:
        for i,j in sorted(zh_word_dic.items(),key=lambda x:x[1],reverse=True):
            whdl.write("{}\t{}\n".format(i,j))
    with open(en_vocab,'w',encoding='utf-8') as whdl:
        for i,j in sorted(en_word_dic.items(),key=lambda x:x[1],reverse=True):
            whdl.write("{}\t{}\n".format(i,j))


with open(file_middle,'w',encoding='utf-8') as whdl:
    for ch,en in zip(train_chinese,train_english):
        num += 1
        token_en = [i.lower() for i in jieba.cut(en) if i.strip()]
        token_ch = [i.lower() for i in jieba.cut(ch) if i.strip()]
        for word in token_en:
            en_word_dic.setdefault(word,0)
            en_word_dic[word] += 1
        for word in token_ch:
            zh_word_dic.setdefault(word,0)
            zh_word_dic[word] += 1
        train_token_chinese.append(token_ch)
        train_token_english.append(token_en)
        whdl.write("{}\n".format(' '.join(token_en)))
        whdl.write("{}\n".format(' '.join(token_ch)))
        pb.complete(1)
        if num % 32 == 0:
            pb.display_progress_bar()
        if num % 1024 == 0:
            whdl.flush()
            os.fsync(whdl.fileno())
        if num % 100000 == 0:
            write_words_to_file()
            
import pickle
with open(pickle_middle,'wb') as whdl:
    pickle.dump((train_token_chinese,train_token_english),whdl)