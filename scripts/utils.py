from keras import backend as K
import numpy as np
import os
import os
import logging
import argparse
from tqdm import tqdm


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def build_embedding(embedding_fn, dictionary, data_dir):
    print("building embedding matrix for dict %d if need..." % len(dictionary))
    embedding_mat_fn = os.path.join(data_dir, "embedding_mat_%d.npy" % (len(dictionary)))
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
    embedding_indexval = list(embedding_index.values())
    embedding_dim = len(embedding_indexval[0])
    embedding_mat = np.zeros((len(dictionary) + 1, embedding_dim))   # 0 is for padding
    for i, word in dictionary.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i + 1] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat


def unfreeze(layers):
    for layer in layers:
        layer.trainable = True


def freeze(layers):
    for layer in layers:
        layer.trainable = False

def config():
    UNK_token = 0
    PAD_token = 1
    EOS_token = 2
    SOS_token = 3

    USE_CUDA = True
    MAX_LENGTH = 10

    parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
    parser.add_argument('-ds', '--dataset', help='dataset, babi or kvr', required=False)
    parser.add_argument('-t', '--task', help='Task Number', required=False)
    parser.add_argument('-dec', '--decoder', help='decoder model', required=False)
    parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False)
    parser.add_argument('-bsz', '--batch', help='Batch_size', required=False)
    parser.add_argument('-lr', '--learn', help='Learning Rate', required=False)
    parser.add_argument('-dr', '--drop', help='Drop Out', required=False)
    parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', required=False, default=1)
    parser.add_argument('-layer', '--layer', help='Layer Number', required=False)
    parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
    parser.add_argument('-path', '--path', help='path of the file to load', required=False)
    parser.add_argument('-test', '--test', help='Testing mode', required=False)
    parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
    parser.add_argument('-useKB', '--useKB', help='Put KB in the input or not', required=False, default=1)
    parser.add_argument('-ep', '--entPtr', help='Restrict Ptr only point to entity', required=False, default=0)
    parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=3)
    parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
    parser.add_argument('-model-name', '--model-name', help='An add name for running model', required=False,
                        default='rnn_mem')
    parser.add_argument('-serverip', '--serverip', help='server of visdom ip adress,for example "http://10.15.62.15"',
                        required=False)

    parser.add_argument('-rnnlayer', '--rnnlayers', help='the layer of encoder rnn when use model rnn encoder',
                        required=False, default=1)
    parser.add_argument('-addition-name', '--addition-name', help='An additional name for running model',
                        required=False, default='')
    parser.add_argument('-debirnn', '--debirnn', help='Use birnn in memory decoder', action='store_true',
                        required=False, default=False)
    parser.add_argument('-enbirnn', '--enbirnn', help='Use birnn in memory encoder', action='store_true',
                        required=False, default=False)
    parser.add_argument('-kb-layer', '--kb-layer', help='adjust the kb mem layer', required=False, default=None)
    parser.add_argument('-load-limits', '--load-limits', help='Limits the history inputs', required=False, default=1000)
    parser.add_argument('-epoch', '--epoch', help='The total training epochs', required=False, default=100)
    parser.add_argument('-embeddingsize', '--embeddingsize', help='The word embedding size, used in seq2seq models',
                        required=False, default=None)

    args = vars(parser.parse_args())
    print(args)

    name = str(args['task']) + str(args['decoder']) + str(args['hidden']) + str(args['batch']) + str(
        args['learn']) + str(args['drop']) + str(args['layer']) + str(args['limit'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')  # ,filename='save/logs/{}.log'.format(str(name)))

    LIMIT = int(args["limit"])
    USEKB = int(args["useKB"])
    ENTPTR = int(args["entPtr"])
    ADDNAME = args["addName"]
    LOAD_LIMITS = int(args['load_limits'])