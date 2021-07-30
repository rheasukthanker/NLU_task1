import json
from data_handler import DataHandler
from vocab_handler import VocabHandler
from lstm import LSTMModel
from util import perplexity
import numpy as np
import os
import re
import pickle
from tags import Tag, TAGS
import tensorflow as tf
from sys import exit
from load_embedding import load_embedding


INVALID_STR = ''
TRAIN_EXPERIMENT = INVALID_STR
TEST_EXPERIMENT = INVALID_STR


def set_global_vars(cfg):
    global TRAIN_EXPERIMENT
    global TEST_EXPERIMENT
    TRAIN_EXPERIMENT = cfg['train_experiment']
    TEST_EXPERIMENT = cfg['test_experiment']


def tf_multithreaded_cfg():
    tf_cfg = tf.ConfigProto()
    tf_cfg.intra_op_parallelism_threads = 8
    tf_cfg.inter_op_parallelism_threads = 8
    return tf_cfg


def get_test_data_handler(cfg):
    data_handler = DataHandler(
        path=cfg['data_test'],
        sentence_len_with_tags=cfg['sentence_len_with_tags'],
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    return data_handler


def get_condgen_data_handler(cfg):
    data_handler = DataHandler(
        path=cfg['data_continuation'],
        sentence_len_with_tags=cfg['condgen_sentence_len_with_tags'],
        batch_size=cfg['batch_size'],
        shuffle=False,
        add_eos=False
    )
    return data_handler


def get_vocab_handler(vocab_handler_store_path):
    with open(vocab_handler_store_path, 'rb') as f:
        vocab_handler = pickle.load(f)

    return vocab_handler


def get_lstm(cfg):
    state_size = 1024 if TRAIN_EXPERIMENT == 'c' else 512
    embedding_trainable = TRAIN_EXPERIMENT == 'a'

    lstm = LSTMModel(
        sentence_len=cfg['sentence_len_with_tags'],
        cg_sentence_len=cfg['condgen_sentence_len_with_tags'],
        vocab_size=cfg['vocab_size'],
        learning_rate=cfg['learning_rate'],
        embedding_size=cfg['embedding_size'],
        grad_clip=cfg['grad_clip'],
        state_size=state_size,
        softmax_size=cfg['softmax_size'],
        embedding_trainable=embedding_trainable
    )
    return lstm


def evaluate(data_handler, vocab_handler, lstm, sess):
    for batch_cnt, (sent_x, sent_y, sent_len) in enumerate(data_handler):
        x = vocab_handler.vocab_indices(sent_x)
        y = vocab_handler.vocab_indices(sent_y)
        y_pred = lstm.predict(x, sent_len, sess)
        perp = perplexity(y_pred, y, sent_len)
        print('\n'.join((str(pval) for pval in perp)))


def cond_gen(data_handler, vocab_handler, lstm, sess):
    for batch_cnt, (sent_x, sent_y, sent_len) in enumerate(data_handler):
        x = vocab_handler.vocab_indices(sent_x)
        continued_sent_indices = lstm.cond_gen(x, sent_len, sess)

        lens = [continued_sent_indices.shape[1] - 1]*continued_sent_indices.shape[0]
        continued_sent = vocab_handler.vocab_words(continued_sent_indices, lens)

        # exclude bos
        for i, sent in enumerate(continued_sent):
            continued_sent[i] = sent[1:]

        for i, sent in enumerate(continued_sent):
            try:
                eos_idx = sent.index(TAGS[Tag.EOS])
                continued_sent[i] = sent[:eos_idx + 1]
            except ValueError:
                pass

        for sent in continued_sent:
            print(sent[0], end='')
            for word in sent[1:]:
                print(' {}'.format(word), end='')
            print()


def main():
    with open('config_ref.json') as json_file:
        config_dict = json.load(json_file)

    set_global_vars(config_dict)
    assert(INVALID_STR not in [TRAIN_EXPERIMENT, TEST_EXPERIMENT])
    assert(TRAIN_EXPERIMENT in ['a', 'b', 'c'])
    assert(TEST_EXPERIMENT in ['evaluate', 'cond_gen'])

    vocab_handler = get_vocab_handler(config_dict['vocab_handler_store_path'])
    lstm = get_lstm(config_dict)

    saver = tf.train.Saver()
    tf_cfg = tf_multithreaded_cfg()

    with tf.Session(config=tf_cfg) as sess:
        sess.run(tf.global_variables_initializer())
        #TODO: Train new graph, and restore it here.
        try:
            saver.restore(sess, config_dict['testing_checkpoint_file'])
        except ValueError as e:
            print('Could not restore trained model for testing.')
            print('Error was:')
            print(e)
            print('Aborting...')
            exit(-1)

        if TEST_EXPERIMENT == 'evaluate':
            data_handler = get_test_data_handler(config_dict)
            evaluate(data_handler, vocab_handler, lstm, sess)
        elif TEST_EXPERIMENT == 'cond_gen':
            data_handler = get_condgen_data_handler(config_dict)
            cond_gen(data_handler, vocab_handler, lstm, sess)
        else:
            print('How?')
            exit(-567)


if __name__ == "__main__":
    main()
