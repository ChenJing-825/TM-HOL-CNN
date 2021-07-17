# -- coding: UTF-8 --
"""
此文件包含要执行的代码
a)加载预先训练好的分类器和相关文件。
(b)将新的输入数据转换为正确的分类器格式。
(c)对转换后的数据运行分类器，返回结果。
"""
import io
import re
import pickle
# 以序列化对象并保存到磁盘中，并在需要的时候读取出来，任何对象都可以执行序列化操作
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
# 情感分类器
from textstat.textstat import *
# textstat:文本可读性计算包

stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()

stemmer = PorterStemmer()
# Porter_Stemmer英文进行分词处理
# 它不能保证还原到单词的原本,只是把很多基于这个单词的变种变为某一种形式


def preprocess(text_string):
    # 接受文本字符串并替换:
    # 1)带有URLHERE的url
    # 2）一个实例有很多空白
    # 3)@user
    # 这允许我们获得标准化的url和提及次数
    # 不关心提到的特定的人
    # 第一步，我们清理推文。这包括删除URL（以
    # 'http：//'或 'https：//'开头的URL）和标签（即'@user''）和不相关的表达式（以不受ANSI编码支持）。
    space_pattern = '\s+ '
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    return parsed_text


def tokenize(tweet):
    # 删除标点符号和多余的空白，设置为小写，
    # 返回一个被截断的列表。
    print(type(tweet))
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    # 英文分词
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def tfidf_score(tweets):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords, # We do better when we keep stopwords
        use_idf=True,
        smooth_idf=False,
        norm=None, # Applies l2 norm smoothing
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.501
        )

    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    idf_dict = {i: idf_vals[i] for i in vocab.values()} # keys are indices; values are IDF scores
    return tfidf, idf_dict


def get_pos_tags(tweets):
    """获取字符串列表(tweets)和
    返回一个(POS标记)字符串列表。
    """
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        # nltk.pos_tag()函数是一种用来进行词性标注的工具
        tag_list = [x[1] for x in tags]
        # for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags


def pos_array(tweet_tags):
    # We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,  # We do better when we keep stopwords
        use_idf=False,
        smooth_idf=False,
        norm=None,  # Applies l2 norm smoothing
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.501,
    )
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names())}
    return pos

# 返回URL，提及和主题标签的计数。
def count_twitter_objs(text_string):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE')


def other_features_(tweet):
    """此函数采用字符串并返回功能列表。
     其中包括情绪评分，文字和可读性评分，
     以及Twitter的特定功能。
     """
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)  # 音节统计
    # textstat:文本可读性计算包
    num_chars = sum(len(w) for w in words)  # num chars in words
    num_chars_total = len(tweet)            # 文本的长度
    num_terms = len(tweet.split())     # split() 通过指定分隔符对字符串进行切片  默认空格
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    # round() 方法返回浮点数x的四舍五入值。 4为保留位数
    num_unique_terms = len(set(words.split()))
    # set(text3)获得text3的词汇表
    # ## Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    # #Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)  # Count #, @, and http://返回URL，提及和主题标签的计数
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    # features = pandas.DataFrame(features)
    return features


def get_oth_features(tweets):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats = []
    for t in tweets:
        feats.append(other_features_(t))
    feats = np.array(feats)
    [rows, cols] = feats.shape
    for i in range(rows):
        for j in range(cols):
            if feats[i, j] < 0:
                feats[i, j] = 0

    return feats


# vectorizer = TfidfVectorizer(
#         # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
#         tokenizer=tokenize,
#         preprocessor=preprocess,
#         ngram_range=(1, 3),
#         stop_words=stopwords, #We do better when we keep stopwords
#         use_idf=True,
#         smooth_idf=False,
#         norm=None, # Applies l2 norm smoothing
#         decode_error='replace',
#         max_features=10000,
#         min_df=5,   # 词频低于五忽略
#         max_df=0.501  # 词频高于此值则忽略
#         )
#
# pos_vectorizer = TfidfVectorizer(
#         # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
#         tokenizer=None,
#         lowercase=False,
#         preprocessor=None,
#         ngram_range=(1, 3),
#         stop_words=None,  # We do better when we keep stopwords
#         use_idf=False,
#         smooth_idf=False,
#         norm=None,  # Applies l2 norm smoothing
#         decode_error='replace',
#         max_features=5000,
#         min_df=5,
#         max_df=0.501,
#     )
# # 读取文件
# df = pd.read_csv('../data/tmn/labeled_data.csv')
# tweets = df['tweet'].values
# tweets = [x for x in tweets if type(x) == str]
# tweets_class = df['class'].values
# # 获得特征
# tweet_tags = get_pos_tags(tweets)
# tfidf = vectorizer.fit_transform(tweets).toarray()  # type为array，(24783, 8068)
# pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()  # type 为array (24783, 4210)
#
# features = get_oth_features(tweets)
#
# all_feature = np.concatenate([tfidf, pos, features], axis=1)
# print(all_feature.shape)