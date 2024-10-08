import os,string
import random,csv
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
import pandas as pd
import json
os.environ['CUDA_VISIBLE_DEVICES']='2'

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['R8_2','R8_1','R8_3','R8_4','R8_5', 'R52_2','R52_4','R52_3','R52_5','R52_1', 'ohsumed_2','ohsumed_3','ohsumed_5','ohsumed_1','ohsumed_4', 'mr_2','mr_5', 'mr_1', 'mr_3', 'mr_4', 'mr_6']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset +'/'+ dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)n

doc_content_list = []
# f = open('data/corpus/' + dataset + '.clean.txt', 'r')
f = open('data/' + dataset +'/'+ dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

# [[five gazes of each each word],[],[]]
doc_gaze_dict = {}
print('data/'+ dataset +'/new_' + dataset + '_gaze.csv')
data = pd.read_csv('data/'+ dataset +'/new_' + dataset + '_gaze.csv', encoding='utf-8')
sentence_id = data['sentence_id']
word_id = data['word_id']
word = data['word']
nFix = data['nFix']
GD = data['GD']
FFD = data['FFD']
GPT = data['GPT']
TRT = data['TRT']
id_list = []
num=-1
for i in range(len(sentence_id)):
    if sentence_id[i] not in id_list:
        id_list.append(sentence_id[i])
        num+=1
        doc_gaze_dict[num]=[]

    doc_gaze_dict[num].append([FFD[i],GD[i],GPT[i],TRT[i],nFix[i]])

# print(doc_gaze_dict)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
# print('train_ids:')
# print(train_ids)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset +'/'+ dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
# print('test_ids:')
# print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset +'/'+ dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
# print(ids)
# print(len(ids))

# -------------------open gaze jsonl--------------------
'''
train_ex=[]
test_ex=[]
all_ex = []
with open('data/corpus/new_' + dataset + '_gaze.jsonl', encoding="utf-8") as f:
    temp_id=0
    for line in f:
        line = line.strip()
        ex = json.loads(line)
        all_ex.append(ex)
        if temp_id in train_ids:
            train_ex.append(ex)
        elif temp_id in test_ids:
            test_ex.append(ex)
        temp_id+=1
    # token,head,deptag,postag,nFix,GD,FFD,GPT,TRT
num=0
with open('data/corpus/new_' + dataset + '_train_gaze.jsonl', 'w', encoding='utf-8') as fw:
    for ex in train_ex:
        fw.write(json.dumps(ex) + '\n')

with open('data/corpus/new_' + dataset + '_test_gaze.jsonl', 'w', encoding='utf-8') as fw:
    for ex in test_ex:
        fw.write(json.dumps(ex) + '\n')

with open('data/corpus/new_' + dataset + '_all_gaze.jsonl', 'w', encoding='utf-8') as fw:
    for ex in all_ex:
        fw.write(json.dumps(ex) + '\n')
        num += 1
print('num---------------', num)
'''
# -------------------open gaze jsonl--------------------

shuffle_doc_name_list = []
shuffle_doc_words_list = []
shuffle_doc_gaze_list = []
num=0
for id in ids:
    if num<10:
        print(id)
        print(doc_content_list[int(id)])
        print(doc_gaze_dict[int(id)])
        num+=1

    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_gaze_list.append(doc_gaze_dict[int(id)])
print('---------split-------------')
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
# shuffle_doc_gaze_str = '\n'.join(shuffle_doc_gaze_list)
for i in range(3):
    print(len(shuffle_doc_words_list[i].split()),shuffle_doc_words_list[i])
    print(len(shuffle_doc_gaze_list[i]),shuffle_doc_gaze_list[i])
print(len(shuffle_doc_words_list),len(shuffle_doc_gaze_list))
print('===========shuffle1==============')
f = open('data/' + dataset + '/'+ dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()
print('===========shuffle2==============')
# f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f = open('data/'+ dataset + '/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# f = open('data/corpus/' + dataset + '_gaze_shuffle.txt', 'w')
# f.write(shuffle_doc_gaze_str)
# f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

print('===========vocab==============')
# f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f = open('data/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

print('===========label==============')
label_list_str = '\n'.join(label_list)
# f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f = open('data/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)
# real_train_gaze_list = shuffle_doc_gaze_list[:real_train_size]

print('===========real_train.name==============')
f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    # print('real_train_size 1:', i / real_train_size)
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    # print('real_train_size 2:', i / real_train_size)
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)
# gaze = np.array(real_train_gaze_list)


# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    # print('1:', i / test_size)
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    # print('2:',i/test_size)
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)
# real_test_gaze_list = shuffle_doc_gaze_list[real_train_size:real_train_size+test_size]
# tgaze = np.array(real_test_gaze_list)
# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    # print('train 1:', i / train_size)
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    # print('train 2:',i/train_size)
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

# allgaze = np.array(shuffle_doc_gaze_list)
# print(x.shape, y.shape,gaze.shape, tx.shape, ty.shape, tgaze.shape,allx.shape, ally.shape,allgaze.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

num = 0
for doc_words in shuffle_doc_words_list:
    # print('doc:',num/(len(shuffle_doc_words_list)))
    num+=1
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
num = 0
for window in windows:
    # print(num/len(windows))
    num+=1
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
row = []
col = []
row_text = []
col_text = []
row_gaze = []
col_gaze = []
row_fuse = []
col_fuse = []
weight_text = []
weight_gaze = [] # 只替换掉PPMI
weight_fuse = [] # PPMI+weight_gaze

# pmi as weights

num_window = len(windows)
# Inword  if i, j are words and i 6 = j;i,j>train_size
# 在pair count里面的才会计算weight
print(len(shuffle_doc_gaze_list),len(word_pair_count))
num=0
for key in word_pair_count:
    # print(num/len(word_pair_count))
    num+=1
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    # vocab
    row.append(train_size + i)
    col.append(train_size + j)
    row_text.append(train_size + i)
    col_text.append(train_size + j)
    weight_text.append(pmi)
print('------------row,col------------')
# print(row,col)
print('------------row_text,col_text------------')
# print(row_text,col_text)
print('------------weight_text------------')
# print(weight_text)
# print(train_size,vocab_size,test_size,train_size+vocab_size+test_size)
# exit()
# word vector cosine similarity as weight_gaze

def Wasserstein(p,q):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(p,q)
# shuffle_doc_gaze_list
# row = []
# col = []
# weight_gaze=[]
min_was=999
max_was=-1

log_path = 'r8_was.csv'
file = open(log_path, 'a+', encoding='utf-8', newline='')
csv_writer = csv.writer(file)

for i in range(len(shuffle_doc_words_list)):
    # print('1:',i/len(shuffle_doc_words_list))
    doc_words = shuffle_doc_words_list[i]
    gaze_words = shuffle_doc_gaze_list[i]
    # print(len(gaze_words),gaze_words)
    words = doc_words.split()
    # print(len(words),words)
    # new_words=[]
    new_gaze_words=[]
    num=0
    for word in words:
        for c in string.punctuation:
            word = word.replace(c, "")
        # print(word, num)
        if word=='':
            new_gaze_words.append([0,0,0,0,0])
        else:
            new_gaze_words.append(gaze_words[num])
            num+=1

    # =============在gaze这里加上，取消注释===========
    # ttotal = 0
    # for j in range(len(new_gaze_words)):
    #     # print(new_gaze_words[j])
    #     ttotal += float(new_gaze_words[j][3])

    for j in range(len(words)):
        for k in range(len(words)):
            if len(new_gaze_words[j])!=5 or len(new_gaze_words[k])!=5:
                was=0
            else:
                was = Wasserstein(new_gaze_words[j],new_gaze_words[k])
            row.append(train_size + j)
            col.append(train_size + k)
            row_gaze.append(train_size + j)
            col_gaze.append(train_size + k)
            # print(was)
            csv_writer.writerow([str(words[j]),str(words[k]),str(was)])
            if was>max_was:
                max_was=was
                print('over max:',words[j],words[k],was)
            if was<min_was:
                min_was=was
                print('over min:', words[j], words[k], was)
            weight_gaze.append(was)
            
            # =============在gaze这里加上，取消注释===========
            
            # row.append(j)
            # col.append(k)
            # row_gaze.append(train_size+j)
            # col_gaze.append(train_size+k)
            # row_text.append(train_size + j)
            # col_text.append(train_size + k)
            # gdf_1 = float(new_gaze_words[j][3] / ttotal)
            # gdf_2 = float(new_gaze_words[k][3] / ttotal)
            # weight_gaze.append((gdf_1+gdf_2))
            # weight_text.append((gdf_1+gdf_2))
            # weight_fuse.append((gdf_1+gdf_2))


for i in range(len(weight_gaze)):
    print('weight_gaze:',i/len(weight_gaze))
    weight_gaze[i]=1-(weight_gaze[i]-min_was)/(max_was-min_was)
print(max_was,min_was)
# print(len(weight),len(weight_gaze))
weight_fuse = weight_text + weight_gaze
# 接下来的步骤是一样的
# print(len(weight))
# exit()
'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {}
# doc_word_gaze = {}
num = 0
for doc_id in range(len(shuffle_doc_words_list)):
    print('doc_id:',num/(len(shuffle_doc_words_list)))
    num+=1
    doc_words = shuffle_doc_words_list[doc_id]
    gaze_words = shuffle_doc_gaze_list[doc_id]

    words = doc_words.split()
    # num = 0
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
            # doc_word_gaze[doc_word_str] +=gaze_words[num]
        else:
            doc_word_freq[doc_word_str] = 1
            # doc_word_gaze[doc_word_str] = 0
        # num+=1

# Indoc 这里的i是document，j是word，所以i<train_size,j>train_size
# =============在gaze这里去掉，加上注释===========

for i in range(len(shuffle_doc_words_list)):
    print('2:',i/len(shuffle_doc_words_list))
    doc_words = shuffle_doc_words_list[i]
    gaze_words = shuffle_doc_gaze_list[i]

    words = doc_words.split()
    new_gaze_words = []
    num = 0
    for word in words:
        for c in string.punctuation:
            word = word.replace(c, "")
        print(word, num)
        if word == '':
            new_gaze_words.append([0, 0, 0, 0, 0])
        else:
            new_gaze_words.append(gaze_words[num])
            num += 1
    doc_word_set = set()
    ttotal = 0
    for j in range(len(new_gaze_words)):
        ttotal +=float(new_gaze_words[j][3])
    num = 0
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row_text.append(i)
            row.append(i)
            row_gaze.append(i)
        else:
            row.append(i + vocab_size)
            row_text.append(i + vocab_size)
            row_gaze.append(i + vocab_size)
        col.append(train_size + j)
        col_text.append(train_size + j)
        col_gaze.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])

        gdf = float(new_gaze_words[num][3]/ttotal)
        
        num+=1
        # weight_text.append(freq * idf)
        # weight_gaze.append(freq * idf)
        # weight_fuse.append(freq * idf)
        weight_text.append((freq + gdf) * idf)
        weight_gaze.append((freq + gdf) * idf)
        weight_fuse.append((freq + gdf) * idf)
        # weight_text.append(gdf)
        # weight_gaze.append(gdf)
        # weight_fuse.append(gdf)
        doc_word_set.add(word)

print('------------weight 2------------')
# print(len(weight),weight)
print(len(row),len(col))
print(len(row_text),len(col_text),len(weight_text))
print(len(row_gaze),len(col_gaze),len(weight_gaze))
print(len(row_fuse),len(col_fuse),len(weight_fuse))

node_size = train_size + vocab_size + test_size
adj_text = sp.csr_matrix(
    (weight_text, (row_text, col_text)), shape=(node_size, node_size))
adj_gaze = sp.csr_matrix(
    (weight_gaze, (row_gaze, col_gaze)), shape=(node_size, node_size))
print('===========x==============')
# dump objects
f = open("data/{}/ind.{}.x".format(dataset,dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/{}/ind.{}.y".format(dataset,dataset), 'wb')
pkl.dump(y, f)
f.close()

# f = open("data/ind.{}.gaze".format(dataset), 'wb')
# pkl.dump(gaze, f)
# f.close()

f = open("data/{}/ind.{}.tx".format(dataset,dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/{}/ind.{}.ty".format(dataset,dataset), 'wb')
pkl.dump(ty, f)
f.close()

# f = open("data/ind.{}.tgaze".format(dataset), 'wb')
# pkl.dump(tgaze, f)
# f.close()

f = open("data/{}/ind.{}.allx".format(dataset,dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/{}/ind.{}.ally".format(dataset,dataset), 'wb')
pkl.dump(ally, f)
f.close()

# f = open("data/ind.{}.allgaze".format(dataset), 'wb')
# pkl.dump(allgaze, f)
# f.close()

f = open("data/{}/ind.{}.adj_text".format(dataset,dataset), 'wb')
pkl.dump(adj_text, f)
f.close()

f = open("data/{}/ind.{}.adj_gaze".format(dataset,dataset), 'wb')
pkl.dump(adj_gaze, f)
f.close()


adj_fuse = sp.csr_matrix(
    (weight_fuse, (row, col)), shape=(node_size, node_size))
# print('=============adj==============')
# print(adj.shape,adj)

f = open("data/{}/ind.{}.adj_fuse".format(dataset,dataset), 'wb')
pkl.dump(adj_fuse, f)
f.close()
