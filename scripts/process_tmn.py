encoding = 'UTF-8'
import numpy as np
import gensim
from scipy import sparse
import pickle
import json
import pandas as pd
import csv
import nltk
import logging
import copy
import os
from nltk.stem.porter import *
from feature import *
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

# 先是进行数据的读取
data_file = '../data/tmn/labeled_data.csv'
data_dir = os.path.dirname(data_file)
df = pd.read_csv('../data/tmn/labeled_data.csv', encoding='utf8')
tweets = df['tweet'].values
tweets = [x for x in tweets if type(x) == str]
tweets_class = df['class'].values
tweets_class_list = tweets_class.tolist()
len_label = len(tweets_class_list)
label_dict = {}
# 读取数据集中的类标签
for i in range(len_label):
    if tweets_class_list[i] == 0:
        label_dict['0'] = 'Hate speech'
    elif tweets_class_list[i] == 1:
        label_dict['1'] = 'offensive_language'
    else:
        label_dict['2'] = 'neither'

stopwords = stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt", "iphone", "ipad", "android"]
stopwords.extend(other_exclusions)
space_pattern = r'\s+'  # 空格
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # url正则表达式
mention_regex = r'@[\w\-]+'  # @users

length = len(tweets)
# print(length)
# 2020/8/22 将URLHERE 和MENTIONHERE替换为空格
# 2020/9/17 使用
"""
    接受一个文本字符串并替换:
    1) urls with URLHERE  使用URLHERE 替换url
    2) lots of whitespace with one instance 用一个空格替代多个空格
    3) mentions with MENTIONHERE  使用MENTIONHERE替代mentions
    """
for a in range(0, length):
    parsed_text = tweets[a]
    parsed_text = re.sub(space_pattern, ' ', parsed_text)  # 将text_string中的space_pattern用‘ ’ 替代
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)  # 将parsed_text中的space_pattern用‘ URLHERE’ 替代
    # parsed_text = re.sub(giant_url_regex, ' ', parsed_text)  # 将parsed_text中的space_pattern用‘ URLHERE’ 替代
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)  # 将parsed_text中的mention_regex用MENTIONHERE替代
    parsed_text = re.sub(mention_regex, ' ', parsed_text)  # 将parsed_text中的mention_regex用MENTIONHERE替代
    tweet = " ".join(re.split("[^a-zA-Z]*", parsed_text.lower())).strip()
    stemmer = PorterStemmer()
    tweets_token = [stemmer.stem(t) for t in tweet.split()]
    # tweet = tweet.split(" ")
    tweets[a] = tweets_token

logging.info("数据预处理完成！")
# build dictionary  # 建立词典
dictionary = gensim.corpora.Dictionary(tweets)
# Dictionary(2796 unique tokens: ['program', 'truck', 'empti', 'us', 'zoe']...)
bow_dictionary = copy.deepcopy(dictionary)   # 深拷贝 改变dictionary不会对bow_dictionary产生影响
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, stopwords)))

len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
# 选定出现的次数为1次的词标记
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
# 将出现次数为1次的词过滤掉
bow_dictionary.filter_extremes(no_below=3, keep_n=None)
# .filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
# 1.去掉出现次数低于no_below的
# 3.在1和2的基础上，保留出现频率前keep_n的单词

bow_dictionary.compactify()


def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []
    tweets_feat_text=[]
    # build bow # 建立词袋
    row = []
    col = []
    value = [] # 存储的值row行col列存储的value值
    row_id = 0
    m_labels = []
    for d_i, doc in enumerate(text_doc):
        if len(bow_dictionary.doc2bow(doc)) < 3:  # filter too short  # 过滤器太短
            # .doc2bow()新的文章按照词典转换成corpus的向量表示形式，doc2bow是Gensim中封装的一个方法，主要用于实现Bow模型
            continue
        else:
            tweets_feat_text.append(doc)
        for i, j in bow_dictionary.doc2bow(doc):
            row.append(row_id)
            col.append(i)
            value.append(j)
            # print(value)

        row_id += 1

        wids = list(map(seq_dictionary.token2id.get, doc))  # #字典，{词，对应的单词id}
        # map()函数以迭代的方式将提供的功能应用于每个项目
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        # filter() 函数用于过滤序列
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
    lens = list(map(len, seq_doc))
    # 生成词典后，转换为向量形式
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))
    # coo_matrix，即n行，m列存了data[i],其余位置皆为0.
    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels,tweets_feat_text


# 处理成seq 、bow词向量    bow_title为稀疏矩阵   bow_title的形状为： (844, 562)   seq_title为：844
seq_title, bow_title, label_title, tweets_feat_text = get_wids(tweets, dictionary, bow_dictionary, tweets_class)

# print(len(tweets_feat_text))  # 22622  成功提取出对应的特征
# print(tweets_feat_text[0])
feat_text = []
for i in range(len(tweets_feat_text)):   # 转化为对应的句子
    list2 = [str(j) for j in tweets_feat_text[i]]
    list3 = ' '.join(list2)
    feat_text.append(list3)
for i in range(len(tweets_feat_text)-1):
    maxlen = 0
    if len(feat_text[i]) > maxlen:
        maxlen = len(feat_text[i])
# print(maxlen)
feat_array = get_oth_features(feat_text)  # 输入到feature中转换为矩阵
# print(feat_array.shape)

# split data  数据分割
indices = np.arange(len(seq_title))
np.random.shuffle(indices)
# 训练集和测试集的比例：8：2
nb_test_samples = int(0.2 * len(seq_title))
seq_title = np.array(seq_title)[indices]
seq_title_train = seq_title[:-nb_test_samples]
seq_title_test = seq_title[-nb_test_samples:]

bow_title = bow_title.tocsr()
bow_title = bow_title[indices]
bow_title_train = bow_title[:-nb_test_samples]
bow_title_test = bow_title[-nb_test_samples:]


label_title = np.array(label_title)[indices]
label_title_train = label_title[:-nb_test_samples]
label_title_test = label_title[-nb_test_samples:]

feat_title = feat_array
feat_title_train = feat_title[:-nb_test_samples]
feat_title_test = feat_title[-nb_test_samples:]

# save
logging.info("save data...")
pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))

pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))

pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))

pickle.dump(feat_title, open(os.path.join(data_dir, "dataMsgFeatTrain"), "wb"))
pickle.dump(feat_title_train, open(os.path.join(data_dir, "dataMsgFeatTrain"), "wb"))
pickle.dump(feat_title_test, open(os.path.join(data_dir, "dataMsgFeatTest"), "wb"))

dictionary.save(os.path.join(data_dir, "dataDictSeq"))
bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("done!")
