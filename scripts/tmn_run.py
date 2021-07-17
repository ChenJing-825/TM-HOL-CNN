encoding = 'UTF-8'
from keras import backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input, Dense, Lambda, Activation, Dropout, Flatten, Bidirectional, Conv2D, MaxPool2D, Reshape, BatchNormalization, Layer, Embedding, dot
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model, Progbar, normalize
from keras.layers.recurrent import LSTM
# import prettytensor as pt
from keras.layers.merge import add, concatenate
from keras.models import Model
import utils
import tensorflow as tf
import keras
import numpy as np
from datetime import datetime
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import matplotlib as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
######################## configurations 配置 ########################
data_dir = '../data/tmn'  # "../data/tmn"    # data dir 数据文件夹
embedding_fn = "../data/enbedding/glove.twitter.27B.100d.txt"  # 训练好的词向量
output_dir = "../labeled_output"     # output save dir 输出保存的文件夹
TOPIC_NUM = int(5)  # topic number  主题数目
HIDDEN_NUM = [500, 500]  # hidden layer size   # 隐藏层的大小
TOPIC_EMB_DIM = 150  # topic memory size       # 主题内存大小
MAX_SEQ_LEN = 24    # clip length for a text  # 剪辑文本的长度
MAX_FEAT_LEN = 82
BATCH_SIZE = 32     # 批处理数量
MAX_EPOCH = 800
MIN_EPOCH = 50
PATIENT = 10
PATIENT_GLOBAL = 60
PRE_TRAIN_EPOCHS = 50
ALTER_TRAIN_EPOCHS = 50
TARGET_SPARSITY = 0.75  # 论文中为0.8  γ= 0.8用于权衡θ和P
KL_GROWING_EPOCH = 0
SHORTCUT = True
TRANSFORM = None    # 'relu'|'softmax'|'tanh'
######################## configurations ########################

# 读取文件
dataSeqTrain_fn = os.path.join(data_dir, "dataMsgTrain")
dataSeqTest_fn = os.path.join(data_dir, "dataMsgTest")
dataBowTrain_fn = os.path.join(data_dir, "dataMsgBowTrain")
dataBowTest_fn = os.path.join(data_dir, "dataMsgBowTest")
dataLabelTrain_fn = os.path.join(data_dir, "dataMsgLabelTrain")
dataLabelTest_fn = os.path.join(data_dir, "dataMsgLabelTest")

dataFeatTrain_fn = os.path.join(data_dir, "dataMsgFeatTrain")
dataFeatTest_fn = os.path.join(data_dir, "dataMsgFeatTest")

dataDictBow_fn = os.path.join(data_dir, "dataDictBow")
# 加载bow词典
dictionary_bow = gensim.corpora.Dictionary.load(dataDictBow_fn)

dataDictSeq_fn = os.path.join(data_dir, "dataDictSeq")
# 加载seq词典
dictionary_seq = gensim.corpora.Dictionary.load(dataDictSeq_fn)

# something need to save  需要保存的数据
if not os.path.exists(output_dir):   # 如果文件不存在
    os.makedirs(output_dir)   # 创建输出文件夹啊
sessLog_fn = os.path.join(output_dir, "Stem-labeled-CNN-topic2-top20.txt")
docTopicTrain_fn = os.path.join(output_dir, "doc_topic_train")
docTopicTest_fn = os.path.join(output_dir, "doc_topic_test")
topicWord_fn = os.path.join(output_dir, "topic_word")
topicWordSample_fn = os.path.join(output_dir, "topwords1.txt")

# 读取bow和seq以及label
bow_train = pickle.load(open(dataBowTrain_fn, 'rb'))
bow_test = pickle.load(open(dataBowTest_fn, 'rb'))

seq_train = pickle.load(open(dataSeqTrain_fn, 'rb'))
seq_test = pickle.load(open(dataSeqTest_fn, 'rb'))

label_train = pickle.load(open(dataLabelTrain_fn, 'rb'))
label_test = pickle.load(open(dataLabelTest_fn, 'rb'))

feat_train = pickle.load(open(dataFeatTrain_fn, 'rb'))   # (676,11)
feat_test = pickle.load(open(dataFeatTest_fn, 'rb'))     # (168,11)

# print(feat_test)
label_dict = json.load(open(os.path.join(data_dir, "labelDict.json")))
CATEGORY = len(label_dict)


def log(logfile, text, write_to_log=True):
    print(text)
    if write_to_log:
        with open(logfile, 'a') as f:
            f.write(text + '\n')


class CustomizedL1L2(regularizers.L1L2):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(K.cast_to_floatx(l1))
        self.l2 = K.variable(K.cast_to_floatx(l2))


def generate_arrays_from_source(sp_mat):
    arrays = np.array(list(map(lambda x: np.squeeze(np.asarray(x.todense())), sp_mat)))
    index_arrays = np.zeros_like(arrays, dtype="float32")
    index_arrays[arrays > 0] = 1
    return normalize(arrays), index_arrays


def sampling(args):
    mu, log_sigma = args
    epsilon = K.random_normal(shape=(TOPIC_NUM,), mean=0.0, stddev=1.0)
    return mu + K.exp(log_sigma / 2) * epsilon


def print_weight_shape(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight.shape)


# 输出打印主题词
def print_top_words(model, n=20):
    beta_exp = np.exp(model.get_weights()[-2])
    beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
    for k, beta_k in enumerate(beta):
        topic_words = [dictionary_bow[w_id] for w_id in np.argsort(beta_k)[:-n-1:-1]]
        print('Topic {}: {}'.format(k, ' '.join(topic_words)))


# 预定义参数θ  主题混合程度  θ = softmax(fθ(z))
# f ∗（·）是一个线性变换输入的神经感知器
def output_theta(model, bow_input, fn):
    theta, _ = model.predict(bow_input)
    print("theta shape", theta.shape)
    pickle.dump(theta, open(fn, 'wb'))


# 预定义参数β
def output_beta(model):
    beta_exp = np.exp(model.get_weights()[-2])
    beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
    pickle.dump(beta, open(topicWord_fn, 'wb'))
    with open(topicWordSample_fn, 'w') as fout:
        for k, beta_k in enumerate(beta):
            topic_words = [dictionary_bow[w_id] for w_id in np.argsort(beta_k)[:-11:-1]]
            fout.write("%s\n" % ' '.join(topic_words))


# 检测文本的稀疏性
def check_sparsity(model, sparsity_threshold=1e-3):
    kernel = model.get_weights()[-2]
    num_weights = kernel.shape[0] * kernel.shape[1]
    num_zero = np.array(np.abs(kernel) < sparsity_threshold, dtype=float).sum()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    current_l1 = K.get_value(cur_l1.l1)
    diff = sparsity_target - cur_sparsity
    new_l1 = current_l1 * 2.0 ** diff
    K.set_value(cur_l1.l1, K.cast_to_floatx(new_l1))

"""
main program
"""
# process input

# 为了将数据输入到模型中，所有的输入序列都必须具有相同的长度，该函数是将序列转化为经过填充以后的一个新序列
# maxlen为序列的最大长度，大于此长度的序列将被截短，小于此长度的序列将在后部填0，在命名实体识别的任务中，主要是指句子的最大长度。
seq_train_pad = pad_sequences(seq_train, maxlen=MAX_SEQ_LEN)  # (18098, 24) =434352
seq_test_pad = pad_sequences(seq_test, maxlen=MAX_SEQ_LEN)  # (4524, 24)= 108576

label_train = keras.utils.to_categorical(label_train)   # (18098, 3)=54294

label_test = keras.utils.to_categorical(label_test)  # (4524, 3)=13572

bow_train, bow_train_ind = generate_arrays_from_source(bow_train)
# bow_train =  bow_train_ind = (18098, 5290)
bow_test, bow_test_ind = generate_arrays_from_source(bow_test)
# bow_test = bow_test_ind (4524, 5290)
test_count_indices = np.sum(bow_test_ind, axis=1)  # (4524,)

# build model
# (feat_train.shape)  # (18098, 11)
# len(dic_bow) = 15575
bow_input = Input(shape=(len(dictionary_bow),), name="bow_input")     # the normalised input
# bow_input (?, 5290)
feat_input = Input(shape=(17,), name="feat_input")  # (?, 11)
seq_input = Input(shape=(MAX_SEQ_LEN, ), dtype='float32', name='seq_input')  # (?,24)

embedding_mat = utils.build_embedding(embedding_fn, dictionary_seq, data_dir)  # (15576, 50)

emb_dim = embedding_mat.shape[1]  # 50

seq_emb = Embedding(len(dictionary_seq) + 1,
                    emb_dim,
                    weights=[embedding_mat],
                    input_length=MAX_SEQ_LEN,
                    trainable=False)
# (,24,50)

topic_emb = Embedding(TOPIC_NUM, len(dictionary_bow), input_length=TOPIC_NUM, trainable=False)

# feat_emb = Embedding(11, 848, input_length=11, trainable=False)
feat_emb = Embedding(len(dictionary_seq) + 1,
                    emb_dim,
                    input_length=17,
                    trainable=False)

psudo_input = Input(shape=(TOPIC_NUM, ), dtype='float32', name="psudo_input")  # (?, 2)

######################## build ntm #########################
e1 = Dense(HIDDEN_NUM[0], activation='relu')
e2 = Dense(HIDDEN_NUM[1], activation='relu')
e3 = Dense(TOPIC_NUM)
e4 = Dense(TOPIC_NUM)
h = e1(bow_input)
h = e2(h)
if SHORTCUT:
    es = Dense(HIDDEN_NUM[1], use_bias=False)
    h = add([h, es(bow_input)])

z_mean = e3(h)
z_log_var = e4(h)
# sample
hidden = Lambda(sampling, output_shape=(TOPIC_NUM,))([z_mean, z_log_var])
# build generator
g1 = Dense(TOPIC_NUM, activation="tanh")
g2 = Dense(TOPIC_NUM, activation="tanh")
g3 = Dense(TOPIC_NUM, activation="tanh")
g4 = Dense(TOPIC_NUM)


def generate(h):
    tmp = g1(h)
    tmp = g2(tmp)
    tmp = g3(tmp)
    tmp = g4(tmp)
    if SHORTCUT:
        r = add([Activation("tanh")(tmp), h])
    else:
        r = tmp
    if TRANSFORM is not None:
        r = Activation(TRANSFORM)(r)
        return r
    else:
        return r


represent = generate(hidden)
represent_mu = generate(z_mean)

# build decoder
l1_strength = CustomizedL1L2(l1=0.001)
d1 = Dense(len(dictionary_bow), activation="softmax", kernel_regularizer=l1_strength, name="p_x_given_h")
p_x_given_h = d1(represent)

# # build discriminator
filter_sizes = [3, 4, 5]
num_filters = 512

# build classifier

c1 = Dense(TOPIC_EMB_DIM, activation='relu')
t1 = Dense(TOPIC_EMB_DIM, activation='relu')
f1 = Dense(TOPIC_EMB_DIM, activation="relu")
f2 = Dense(TOPIC_EMB_DIM, activation="relu")
f3 = Dense(TOPIC_EMB_DIM, activation="relu")
f4 = Dense(TOPIC_EMB_DIM, activation="relu")
f5 = Dense(TOPIC_EMB_DIM, activation="relu")
f6 = Dense(TOPIC_EMB_DIM, activation="relu")

# 以下为feat_input
d1 = Dense(TOPIC_EMB_DIM, activation='relu')
d2 = Dense(TOPIC_EMB_DIM, activation='relu')
d3 = Dense(TOPIC_EMB_DIM, activation="relu")
d4 = Dense(TOPIC_EMB_DIM, activation='relu')
d5 = Dense(TOPIC_EMB_DIM, activation='relu')
d6 = Dense(TOPIC_EMB_DIM, activation="relu")
d7 = Dense(TOPIC_EMB_DIM, activation="relu")

o1 = Dense(TOPIC_EMB_DIM, activation='relu')
o2 = Dense(TOPIC_EMB_DIM, activation='relu')
o3 = Dense(TOPIC_EMB_DIM, activation='relu')
o4 = Dense(TOPIC_EMB_DIM, activation='relu')
o5 = Dense(TOPIC_EMB_DIM, activation='relu')

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], TOPIC_EMB_DIM), padding="valid",
                kernel_initializer='normal', activation='relu')
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], TOPIC_EMB_DIM), padding="valid",
                kernel_initializer='normal', activation='relu')
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], TOPIC_EMB_DIM), padding="valid",
                kernel_initializer='normal', activation='relu')

s1 = Bidirectional(LSTM(80))
s2 = Dense(CATEGORY, activation='softmax')
cls_vars = [c1, t1, f1, o1, s1, s2]
x = seq_emb(seq_input)  # (?, 24, 50)
x = c1(x)   # 降维后的x (?, 24, 150)
x = Dropout(0.05)(x)  # 处理后的x (?, 24, 150)
wt_emb = topic_emb(psudo_input) # (?, 2, 5290)
wt_emb = t1(wt_emb)     # 降维(?, 2, 150)

feat_emb = feat_emb(feat_input) # (?, 11, 5290)
feat_emb = d1(feat_emb)  # 降维(?, 11, 150)

# 图2中的计算层，三个计算层
# first match layer
# x= (?, 24, 150)   wt_emb = (?, 2, 150)
match = dot([x, wt_emb], axes=(2, 2))  # (?, 24, 2) # represent_mu = (?,2)
joint_match = add([represent_mu, match])  # (?, 24, 2)
joint_match = f1(joint_match)  # (?, 24, 150)
topic_sum = add([joint_match, x])
topic_sum = o1(topic_sum)  # (?, 24, 150)

match = dot([topic_sum, feat_emb], axes=(2, 2))  # (?, 24, 11)
joint_match = d2(match)  # (?, 24, 150)
feat_topic_sum = add([joint_match, topic_sum])
feat_topic_sum = d3(feat_topic_sum)  #

# second match layer
# topic_sum = (?, 24, 150); wt_emb= (?, 2, 150)
match = dot([topic_sum, wt_emb], axes=(2, 2))
match_2 = dot([x, feat_emb], axes=(2, 2))
joint_match = add([represent_mu, match])
joint_match = f2(joint_match)
topic_sum = add([joint_match, x])
topic_sum = o2(topic_sum)

match = dot([topic_sum, feat_emb], axes=(2, 2))  # (?, 24, 11)
joint_match = d4(match)  # (?, 24, 150)
feat_topic_sum = add([joint_match, topic_sum])
feat_topic_sum = d5(feat_topic_sum)  #

# third match layer
match = dot([topic_sum, wt_emb], axes=(2, 2))
joint_match = add([represent_mu, match])
joint_match = f3(joint_match)
topic_sum = add([joint_match, x])
topic_sum = o3(topic_sum)  # (?, 24, 150)

match = dot([topic_sum, feat_emb], axes=(2, 2))  # (?, 24, 11)
joint_match = d6(match)  # (?, 24, 150)
feat_topic_sum = add([joint_match, topic_sum])
feat_topic_sum = d7(feat_topic_sum)  #
# # fourth match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f4(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o4(topic_sum)
# # fifth match layer
# match = dot([topic_sum, wt_emb], axes=(2, 2))
# joint_match = add([represent_mu, match])
# joint_match = f5(joint_match)
# topic_sum = add([joint_match, x])
# topic_sum = o5(topic_sum)


x = Reshape((MAX_SEQ_LEN, TOPIC_EMB_DIM, 1))(feat_topic_sum)
x0 = conv_0(x)
x1 = conv_1(x)
x2 = conv_2(x)
mp0 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(x0)
mp1 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(x1)
mp2 = MaxPool2D(pool_size=(MAX_SEQ_LEN - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(x2)
out = concatenate([mp0, mp1, mp2], axis=1)
out = Dropout(0.05)(Flatten()(out))
cls_out = s2(out)


def kl_loss(x_true, x_decoded):
    kl_term = - 0.5 * K.sum(
        1 - K.square(z_mean) + z_log_var - K.exp(z_log_var),
        axis=-1)
    return kl_term


def nnl_loss(x_true, x_decoder):
    nnl_term = - K.sum(x_true * K.log(x_decoder + 1e-32), axis=-1)
    return nnl_term


kl_strength = K.variable(1.0)

# 损失函数为主题模型的训练目标variational lower-bound以及文本分类器的训练目标cross-entropy的加权和。
# build combined model
ntm_model = Model(bow_input, [represent_mu, p_x_given_h])
ntm_model.compile(loss=[kl_loss, nnl_loss], loss_weights=[kl_strength, 1.0], optimizer="adagrad")

combine_model = Model([bow_input, seq_input, psudo_input, feat_input], cls_out)
combine_model.compile(optimizer="adadelta", loss=K.categorical_crossentropy, metrics=["accuracy"])

vis_model = Model([bow_input, seq_input, psudo_input, feat_input], [represent_mu, wt_emb, match, cls_out])
vis_model.summary()
# print_weight_shape(combine_model)

# init kl strength
num_batches = int(bow_train.shape[0] / BATCH_SIZE)
kl_base = float(KL_GROWING_EPOCH * num_batches)

optimize_ntm = True
first_optimize_ntm = True
min_bound_ntm = np.inf
min_bound_cls = - np.inf
epoch_since_improvement = 0
epoch_since_improvement_global = 0

# training
for epoch in range(1, MAX_EPOCH + 1):
    progress_bar = Progbar(target=num_batches)  # 进度条
    epoch_train = []
    epoch_test = []

    # shuffle data  打乱数据
    indices = np.arange(bow_train.shape[0])   # indices为0-675的数字
    np.random.shuffle(indices)

    seq_train_shuffle = seq_train_pad[indices]  # (676, 24)
    bow_train_shuffle = bow_train[indices]  # (676, 562)
    bow_train_ind_shuffle = bow_train_ind[indices]  # (676, 562)
    label_train_shuffle = label_train[indices]  # (676, 3)
    psudo_indices = np.expand_dims(np.arange(TOPIC_NUM), axis=0)
    psudo_train = np.repeat(psudo_indices, seq_train_pad.shape[0], axis=0)  # (676, 2)
    psudo_test = np.repeat(psudo_indices, seq_test_pad.shape[0], axis=0)  # (168, 2)
    feat_train_shuffle = feat_train[indices]  # (676, 11)
    if optimize_ntm:
        print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "ntm"))
        for index in range(num_batches):
            # update kl_strength
            if epoch < KL_GROWING_EPOCH:
                K.set_value(kl_strength, np.float32((epoch * num_batches + index) / kl_base))
            else:
                K.set_value(kl_strength, 1.)
            bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            bow_index_batch = bow_train_ind_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            epoch_train.append(ntm_model.train_on_batch(
                bow_batch, [np.zeros([len(bow_batch), TOPIC_NUM]), bow_index_batch]))
            progress_bar.update(index + 1)
        # compute training cost
        [train_loss, train_kld, train_nnl] = np.mean(epoch_train, axis=0)   # 计算每一列的均值   epoch_train可能是空
        print("ntm train loss: %.4f, kld: %.4f, nnl: %.4f" % (train_loss, train_kld, train_nnl))
        print_top_words(ntm_model)
        # check sparsity
        sparsity = check_sparsity(ntm_model)
        update_l1(l1_strength, sparsity, TARGET_SPARSITY)
        # estimate perplexity
        for j in range(5):
            epoch_test.append(ntm_model.evaluate(bow_test, [bow_test, bow_test_ind]))
        [val_loss, kld, nnl] = np.mean(epoch_test, axis=0)
        bound = np.exp(val_loss / np.mean(test_count_indices))
        print("ntm estimated perplexity upper bound on validation set: %.3f" % bound)
        # It is approximated perplexity
        # record the best perplexity
        if bound < min_bound_ntm and epoch > KL_GROWING_EPOCH:
            # log(sessLog_fn, "New best val bound: %.3f in %d epoch" % (bound, epoch))
            min_bound_ntm = bound
            if first_optimize_ntm:
                print("Saving model")
                # ntm_model.save(Model_fn)
                output_theta(ntm_model, bow_train, docTopicTrain_fn)
                output_theta(ntm_model, bow_test, docTopicTest_fn)
            output_beta(ntm_model)
            epoch_since_improvement = 0
            epoch_since_improvement_global = 0
        elif bound >= min_bound_ntm:
            epoch_since_improvement += 1
            epoch_since_improvement_global += 1
            print("No improvement in epoch %d" % epoch)
        if epoch < KL_GROWING_EPOCH:
            print("Growing kl strength %.3f" % K.get_value(kl_strength))
        if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
            optimize_ntm = False
            first_optimize_ntm = False
            epoch_since_improvement = 0
            beta_exp = np.exp(ntm_model.get_weights()[-2])
            beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
            topic_emb.set_weights([beta])   # update topic-word matrix
            # min_bound_ntm += 2    # relax ntm bound a bit
        if epoch_since_improvement_global > PATIENT_GLOBAL:
            break
    else:
        print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "cls"))
        for index in range(num_batches):
            bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # (32, 562)
            seq_batch = seq_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # (32, 24)
            psudo_batch = psudo_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # (32, 2)
            feat_batch = feat_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # (32, 11)
            label_batch = label_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # (32, 3)

            epoch_train.append(combine_model.train_on_batch(
                [bow_batch, seq_batch, psudo_batch, feat_batch], label_batch))
            progress_bar.update(index + 1)
        train_loss, train_acc = np.mean(epoch_train, axis=0)
        print("cls train loss: %.4f" % train_loss)
        y_pred = combine_model.predict([bow_test, seq_test_pad, psudo_test, feat_test])
        y_pred_label = np.argmax(y_pred, axis=1)
        y_true_label = np.argmax(label_test, axis=1)
        test_acc = accuracy_score(y_true_label, y_pred_label)
        test_f1 = f1_score(y_true_label, y_pred_label, average="weighted")
        test_precision_score = precision_score(y_true_label, y_pred_label, average='macro')
        test_recall = recall_score(y_true_label, y_pred_label, average='macro')
        if test_acc > min_bound_cls:
            min_bound_cls = test_acc
            log(sessLog_fn, "New best val acc: %.4f, f1: %.4f, precision_score: %.4f,recall: %.4f in %d epoch" % (min_bound_cls, test_f1, test_precision_score,test_recall, epoch))
            epoch_since_improvement = 0
            epoch_since_improvement_global = 0

        else:
            epoch_since_improvement += 1
            epoch_since_improvement_global += 1
            print("New val acc: %.4f, f1: %.4f, precision_score: %.4f,recall: %.4f in %d epoch " % (test_acc, test_f1,test_precision_score,test_recall,epoch))
            log(sessLog_fn, "New val acc: %.4f, f1: %.4f, precision_score: %.4f,recall: %.4f in %d epoch " % (test_acc, test_f1, test_precision_score,test_recall, epoch))

        if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
            optimize_ntm = True
            epoch_since_improvement = 0
        if epoch_since_improvement_global > PATIENT_GLOBAL:
            break
