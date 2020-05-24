import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import sys

if not '../' in sys.path: sys.path.append('../')

import pandas as pd

from utils import data_utils
from model_config import config
from ved_detAttn import VarSeq2SeqDetAttnModel


def train_model(config):
    print('[INFO] Preparing data for experiment: {}'.format(config['experiment']))
    SRC, TRG, train_data, image_train_data, valid_data, test_data, \
    encoder_embeddings_matrix, decoder_embeddings_matrix = data_utils.read_data()
    x_train, y_train = train_data.src, train_data.trg
    x_val, y_val = valid_data.src, valid_data.trg
    x_test, y_test = test_data.src, test_data.trg

    # Re-calculate the vocab size based on the word_idx dictionary
    config['encoder_vocab'] = len(SRC.vocab)
    config['decoder_vocab'] = len(TRG.vocab)

    config['image_size'] = 32

    model = VarSeq2SeqDetAttnModel(config,
                                   encoder_embeddings_matrix,
                                   decoder_embeddings_matrix,
                                   input_word_index=SRC.vocab,
                                   output_word_index=TRG.vocab)

    model.train(x_train, image_train_data, y_train, x_val, y_val, y_val)


if __name__ == '__main__':
    train_model(config)
