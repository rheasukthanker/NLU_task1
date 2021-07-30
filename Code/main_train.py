import json
from data_handler import DataHandler
from vocab_handler import VocabHandler
from lstm import LSTMModel
from util import perplexity
from shutil import copyfile
import numpy as np
import os
import re
import pickle
from time import time
from datetime import datetime
import tensorflow as tf
from load_embedding import load_embedding

INVALID_STR = ''
EXPERIMENT = INVALID_STR
LOG_DIR = INVALID_STR
LOG_TRAIN_DIR = INVALID_STR
LOG_EVAL_DIR = INVALID_STR
CP_DIR = INVALID_STR
TS_FORMAT = '%Y_%m_%d__%H_%M_%S'


def set_global_vars(cfg):
    global EXPERIMENT
    global LOG_DIR
    global LOG_TRAIN_DIR
    global LOG_EVAL_DIR
    global CP_DIR
    base_log_dir = cfg['log_dir']
    base_cp_dir = cfg['cp_dir']
    EXPERIMENT = cfg['train_experiment']
    timestamp = datetime.fromtimestamp(time()).strftime(TS_FORMAT)
    LOG_DIR = f"{base_log_dir}_{EXPERIMENT}_{timestamp}"
    LOG_TRAIN_DIR = os.path.join(LOG_DIR, 'train')
    LOG_EVAL_DIR = os.path.join(LOG_DIR, 'test')
    CP_DIR = f"{base_cp_dir}_{EXPERIMENT}_{timestamp}"


def save(filename, data):
    savepath = os.path.join(LOG_DIR, filename)
    np.save(savepath, data)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def tf_multithreaded_cfg():
    tf_cfg = tf.ConfigProto()
    tf_cfg.intra_op_parallelism_threads = 16
    tf_cfg.inter_op_parallelism_threads = 16
    return tf_cfg


def get_train_data_handler(cfg):
    train_data_handler = DataHandler(
        path=cfg['data_train'],
        sentence_len_with_tags=cfg['sentence_len_with_tags'],
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    return train_data_handler


def get_eval_data_handler(cfg):
    eval_data_handler = DataHandler(
        path=cfg['data_eval'],
        sentence_len_with_tags=cfg['sentence_len_with_tags'],
        batch_size=cfg["batch_size"],
        shuffle=False
    )
    return eval_data_handler


def get_vocab_handler(train_data_handler, cfg):
    vocab_handler = VocabHandler(
        data_handler=train_data_handler,
        vocab_size=cfg['vocab_size'],
    )
    return vocab_handler


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_lstm(cfg):
    state_size = 1024 if EXPERIMENT == 'c' else 512
    embedding_trainable = True

    lstm = LSTMModel(
        sentence_len=cfg['sentence_len_with_tags'],
        cg_sentence_len=cfg['condgen_sentence_len_with_tags'],
        vocab_size=cfg['vocab_size'],
        learning_rate=cfg['learning_rate'],
        embedding_size=cfg['embedding_size'],
        grad_clip=cfg['grad_clip'],
        state_size=state_size,
        softmax_size=cfg['softmax_size'],
        embedding_trainable=embedding_trainable,
    )
    return lstm


def train(data_handler, vocab_handler, lstm, sess, epoch):
    total_train_loss = []
    writer_train= tf.summary.FileWriter(LOG_TRAIN_DIR, sess.graph)
    for batch_cnt, (sent_x, sent_y, sent_len) in enumerate(data_handler):
        x = vocab_handler.vocab_indices(sent_x)
        y = vocab_handler.vocab_indices(sent_y)
        current_loss = lstm.perform_train_step(x, y, sent_len, sess)
        total_train_loss.append(current_loss)
        print(f'TRAIN Epoch: {epoch}  ---  Batch: {batch_cnt}  ---  Loss: {current_loss:.4f}', end='\r')

    total_train_loss = np.asarray(total_train_loss)
    avg_loss = np.mean(total_train_loss)
    std_loss = np.std(total_train_loss)

    summary={"loss":avg_loss,"loss_sd":std_loss}
    writer_train.add_summary(make_summary(summary),epoch)
    writer_train.flush()

    print(f'\nTRAIN Epoch: {epoch}  ---  Loss mean: {avg_loss:.4f}  ---  Loss std: {std_loss:.4f}')
    save(f'train_loss_{epoch}', total_train_loss)


def evaluate(data_handler, vocab_handler, lstm, sess, epoch):
    # Evaluate model by predicting next word of evaluation set and computing perplexity
    total_eval_loss = []
    perp = []
    writer_eval= tf.summary.FileWriter(LOG_EVAL_DIR, sess.graph)
    for batch_cnt, (sent_x, sent_y, sent_len) in enumerate(data_handler):
        x = vocab_handler.vocab_indices(sent_x)
        y = vocab_handler.vocab_indices(sent_y)
        current_loss, y_pred = lstm.eval(x, y, sent_len, sess)
        total_eval_loss.append(current_loss)
        current_perp = np.mean(perplexity(y_pred, y, sent_len))
        perp.append(current_perp)
        print(f'EVAL  Epoch: {epoch}  ---  Batch: {batch_cnt}  ---  Loss: {current_loss:.4f}  --- Perplexity: {current_perp:.4f}', end='\r')

    total_eval_loss = np.asarray(total_eval_loss)
    perp = np.asarray(perp)
    avg_loss = np.mean(total_eval_loss)
    std_loss = np.std(total_eval_loss)
    avg_perp = np.mean(perp)
    std_perp = np.std(perp)
    summary={"loss":avg_loss,"loss_sd":std_loss,"perp":avg_perp,"perp_sd":std_perp}
    writer_eval.add_summary(make_summary(summary),epoch)
    writer_eval.flush()

    print(f'\nEVAL Epoch: {epoch}  ---  Loss mean: {avg_loss:.4f}  Loss std: {std_loss:.4f}  ---  Perplexity mean: {avg_perp:.4f}  ---  Perplexity std: {std_perp:.4f}')
    save(f'eval_loss_{epoch}', total_eval_loss)
    save(f'eval_perp_{epoch}', perp)


def main():
    # Load configurations from personal config file
    with open('config_ref.json') as json_file:
        config_dict = json.load(json_file)

    set_global_vars(config_dict)
    assert(INVALID_STR not in [EXPERIMENT, LOG_DIR, CP_DIR])
    assert(EXPERIMENT in ['a', 'b', 'c'])
    create_dir(LOG_DIR)
    create_dir(LOG_TRAIN_DIR)
    create_dir(LOG_EVAL_DIR)
    create_dir(CP_DIR)
    
    # Copy configurations to log folder for future reference
    copyfile('config_ref.json', LOG_DIR + '/config_ref.json')

    train_data_handler = get_train_data_handler(config_dict)
    eval_data_handler = get_eval_data_handler(config_dict)
    vocab_handler = get_vocab_handler(train_data_handler, config_dict)
    lstm = get_lstm(config_dict)

    save_obj(vocab_handler, config_dict['vocab_handler_store_path'])

    cp_period = int(config_dict['cp_period'])
    saver = tf.train.Saver()
    tf_cfg = tf_multithreaded_cfg()

    # Start tensorflow session
    with tf.Session(config=tf_cfg) as sess:
        # Write out graph to summary (For Tensorboard)
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        if EXPERIMENT != 'a':
            load_embedding(
                session=sess,
                vocab=vocab_handler.vocab,
                emb=lstm.embedding_matrix,
                path=config_dict['data_embedding'],
                dim_embedding=config_dict['embedding_size'],
                vocab_size=config_dict['vocab_size']
            )

        # Check if a trained model already exists to continue training
        ckpt = tf.train.get_checkpoint_state(CP_DIR)
        step_init = 0
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            step_init = int(re.findall(r"t(.*)\.", ckpt.model_checkpoint_path)[0])+1

        # Start training and after every epoch evaluate perplexity
        for epoch in range(step_init, config_dict['num_epochs']):
            train(train_data_handler, vocab_handler, lstm, sess, epoch)
            evaluate(eval_data_handler, vocab_handler, lstm, sess, epoch)
            if epoch % cp_period == 0:
                save_path = saver.save(sess, "{}/checkpoint{}.ckpt".format(CP_DIR, epoch))
                print(f"Model saved in path: {save_path}")

        writer.close()


if __name__ == "__main__":
    main()
