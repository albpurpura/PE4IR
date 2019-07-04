"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import os

from tqdm import tqdm

from model_MP_original import Model
import tensorflow as tf
import data_utils as du
import numpy as np
import util

PROG_NAME = 'MP_Prob_cos_sim'

"""
This is the main file to REPLICATE the results in the original Matchpyramid paper on the Robust04 collection.
"""


def convert_we_dict_to_emb_matrix(wed, we_size=50):
    np.random.seed(0)
    max_k = -1
    for k, v in wed.items():
        if int(k) > max_k:
            max_k = k
    W_init_embed = np.float32(np.random.uniform(-0.02, 0.02, [max_k + 1, we_size]))

    for k, v in wed.items():
        W_init_embed[k] = v

    pad = np.zeros(we_size)
    W_init_embed[W_init_embed.shape[0] - 1] = pad
    return W_init_embed


def eval_stochastic(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value,
                    fold_index, test_qrels_file, epoch, n_stoch_iter=50):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in enumerate(
            du.compute_test_fd_w2v_v2(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value)):
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False}
        stoch_preds = np.zeros(len(d_names))
        for _ in range(n_stoch_iter):
            pred = model.test_step(sess, feed_dict)
            pred = np.array(pred)
            pred = np.reshape(pred, (-1))
            stoch_preds += pred

        stoch_preds /= n_stoch_iter
        pred = stoch_preds
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, PROG_NAME)
    return map_v


def eval_stochastic2(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index,
                     test_qrels_file, epoch,
                     n_stoch_iter=50):
    print('evaluating...')
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    preds_bn = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in tqdm(enumerate(
            du.compute_test_fd_w2v_v2(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value))):
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False}
        stoch_preds = []
        for k in range(len(d_names)):
            stoch_preds.append([])

        for niter in range(n_stoch_iter):
            pred = model.test_step(sess, feed_dict)
            pred = np.array(pred)
            pred = np.reshape(pred, (-1))
            for j in range(len(pred)):
                stoch_preds[j].append(pred[j])

        stoch_preds = np.array(stoch_preds)
        means = np.mean(stoch_preds, axis=1)
        variances = np.var(stoch_preds, axis=1)

        pred = means - variances
        preds_bn[q_name] = (means, variances, d_names)
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    best_map = 0
    best_alpha = 0
    for alpha in np.arange(0, 3, 0.1):
        for q_name in preds_bn.keys():
            means, variances, d_names = preds_bn[q_name]
            pred = means - (alpha * variances)
            rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
            sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

        map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                             sim_scores_by_qry, test_qrels_file, PROG_NAME)
        if map_v > best_map:
            best_map = map_v
            best_alpha = alpha

    print('alpha=%2.2f, map=%2.5f' % (best_alpha, best_map))
    alpha = best_alpha
    for q_name in preds_bn.keys():
        means, variances, d_names = preds_bn[q_name]
        pred = means - (alpha * variances)
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, PROG_NAME)

    return map_v


def evaluate(sess, model, validation_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, epoch,
             validation_qrels_file, fold_index):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in enumerate(
            du.compute_test_fd_w2v_v2(validation_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value)):
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, validation_qrels_file, PROG_NAME,
                                         output_folder='../results')
    return map_v


def run():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    np.random.seed(0)
    run_to_rerank = '../data/robust.terrier.krovetz.qld.2k.run'
    gt_path = '../data/qrels.robust2004.txt'
    batch_size = 64
    n_epochs = 70
    n_folds = 5
    max_patience = 10
    embeddings = util.load_model('../data/word_embeddings_matrix')

    qbn = util.load_model('../data/q_by_name')
    encoded_d_by_name = util.load_model('../data/d_by_name')
    dbn = {dn: dc[0: min(len(dc), 500)] for dn, dc in encoded_d_by_name.items()}
    mql = max([len(q) for q in qbn.values()])
    mdl = max([len(d) for d in dbn.values()])
    padding_value = embeddings.shape[0] - 1

    config = {'embed_size': 50, 'data1_psize': 3, 'data2_psize': 10, 'embedding': embeddings, 'feat_size': 0,
              'fill_word': padding_value, 'data1_maxlen': mql, 'data2_maxlen': mdl}

    folds = du.compute_kfolds_train_test(n_folds, list(qbn.keys()))

    for fold_index in range(len(folds)):
        print('Fold: %d' % fold_index)
        tf.reset_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            tf.set_random_seed(0)
            model = Model(config)
            model.init_step(sess)

            training_q_names, test_q_names = folds[fold_index]
            training_q_b_name = {training_q_names[i]: qbn[training_q_names[i]] for i in range(len(training_q_names))}
            test_q_b_name = {test_q_names[i]: qbn[test_q_names[i]] for i in range(len(test_q_names))}
            test_qrels_file = 'test_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), test_q_names, test_qrels_file)

            validation_q_names = np.random.choice(training_q_names, int(len(training_q_names) * 0.20), replace=False)
            validation_q_b_name = {validation_q_names[i]: qbn[validation_q_names[i]] for i in
                                   range(len(validation_q_names))}

            validation_qrels_file = 'validation_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), validation_q_names, validation_qrels_file)

            max_fold_map = 0.0
            best_epoch = -1
            patience = 0
            best_test_fold_map = 0.0
            for epoch in range(n_epochs):
                print('epoch=%d' % epoch)
                losses = []
                for i, (data1_max_len, data2_max_len, data1, data2, y, data1_len, data2_len) in \
                        enumerate(du.alt_compute_training_batches_w2v(gt_path, dbn, training_q_b_name, batch_size,
                                                                      padding_value=len(embeddings) - 1,
                                                                      fold_idx=fold_index, n_iter_per_query=1000,
                                                                      max_q_len=mql, max_d_len=mdl, we=embeddings)):
                    feed_dict = {model.X1: data1, model.X1_len: data1_len, model.X2: data2,
                                 model.X2_len: data2_len, model.Y: y, model.training: True}
                    loss = model.train_step(sess, feed_dict)
                    if i % 100 == 0:
                        print('fold=%d, epoch=%d, iter=%d, loss=%2.4f' % (fold_index, epoch, i, loss))
                    losses.append(loss)

                avg_loss = np.mean(np.array(losses))

                map_v = evaluate(sess, model, validation_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, epoch,
                                 validation_qrels_file, fold_index)
                print('Fold %d, validation map=%2.5f, avg_loss=%2.4f' % (fold_index, map_v, avg_loss))


                if map_v > max_fold_map:
                    max_fold_map = map_v
                    best_epoch = epoch
                    map_v_test = evaluate(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value,
                                          epoch,
                                          test_qrels_file, str(fold_index) + '_TEST')
                    print('Fold %d, test map=%2.5f, avg_loss=%2.4f' % (fold_index, map_v_test, avg_loss))
                    best_test_fold_map = map_v_test
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        print('Early stopping at epoch=%d (patience=%d)!' % (epoch, max_patience))
                        break
            print('Max validation map in fold=%2.4f, at epoch=%d, best test map=%2.4f' %
                  (max_fold_map, best_epoch, best_test_fold_map))


if __name__ == '__main__':
    run()
