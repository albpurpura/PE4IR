"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import os

import torch

from model_3 import Model
import tensorflow as tf
import data_utils as du
import numpy as np
import util


PROG_NAME = 'MP_Prob_p_we'


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


def evaluate_test(sess, model, fold_index, output_folder, test_qrels_file, epoch):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_len_test, data2_len_test, d_names, q_name, sim_m_test) in enumerate(
            # du.compute_test_fd_pwe(test_q_by_name, dbn, w, run_to_rerank, mql, mdl)):
            du.load_test_fd_pwe(fold_index, output_folder)):
        feed_dict = {model.X1_len: data1_len_test, model.cross: sim_m_test,
                     model.X2_len: data2_len_test, model.training: False}
        if len(sim_m_test) == 0:
            continue
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, PROG_NAME)
    return map_v


def evaluate(sess, model, fold_index, test_q_by_name, dbn, w, run_to_rerank, mql, mdl, test_qrels_file, epoch):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_len_test, data2_len_test, d_names, q_name, sim_m_test) in enumerate(
            du.compute_test_fd_pwe(test_q_by_name, dbn, w, run_to_rerank, mql, mdl)):
        # du.load_test_fd_pwe(fold_index, output_foldetest_q_by_namer_test_data)):
        feed_dict = {model.X1_len: data1_len_test, model.cross: sim_m_test,
                     model.X2_len: data2_len_test, model.training: False}
        if len(sim_m_test) == 0:
            continue
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, PROG_NAME)
    return map_v


def run():
    np.random.seed(0)
    # run_to_rerank = 'data/run.robust04.bm25.topics.robust04.301-450.601-700_hits=2000.txt'
    run_to_rerank = '/home/ims/albe/test/data/robust.terrier.krovetz.qld.2k.run'
    # run_to_rerank = 'data/rob.qlm.testq.stemming.stoplist.2k.txt'
    gt_path = 'data/qrels.robust2004.txt'
    output_folder_training_batches = 'data/training_batches_pwe'
    # output_folder_test_data = 'data/test_batches_pwe_alt_qlm'
    output_folder_test_data = 'data/test_batches_pwe'
    output_folder_valid_data = 'data/valid_batches_pwe'

    batch_size = 64 #512
    n_epochs = 100
    n_folds = 5
    max_patience = 8

    embs_path = 'data/embeddings_dim_50_margin_2.0'
    dict_pt = torch.load(embs_path, map_location='cpu')
    w = dict_pt["embeddings"]

    qbn = du.load_encoded_collection('encoded_queries_pe')
    encoded_d_by_name = du.load_encoded_collection('encoded_docs_pe')
    dbn = {dn: dc[0: min(len(dc), 500)] for dn, dc in encoded_d_by_name.items()}

    mql = max([len(q) for q in qbn.values()])
    mdl = max([len(d) for d in dbn.values()])
    # du.pre_compute_pwe_training_batches_overall(output_folder_training_batches, w, gt_path, dbn, qbn, 15,
    #                                            n_iter_per_query=500, max_q_len=4, max_d_len=500)
    folds = du.compute_kfolds_train_test(n_folds, list(qbn.keys()))
    # for fold_index in range(len(folds)):
    #     training_q_names, test_q_names = folds[fold_index]
    #     validation_q_names = np.random.choice(training_q_names, int(len(training_q_names) * 0.20), replace=False)
    #     validation_q_b_name = {validation_q_names[i]: qbn[validation_q_names[i]] for i in
    #                            range(len(validation_q_names))}
    #     # test_q_b_name = {test_q_names[i]: qbn[test_q_names[i]] for i in range(len(test_q_names))}
    #     du.pre_compute_test_fd_pwe(validation_q_b_name, dbn, w, run_to_rerank, mql, mdl, fold_index, output_folder_valid_data)

    config = {'embed_size': 50, 'data1_psize': 3, 'data2_psize': 10, 'data1_maxlen': mql, 'data2_maxlen': mdl}
    tf.reset_default_graph()
    tf.set_random_seed(0)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        model = Model(config)
        model.init_step(sess)

        for fold_index in range(len(folds)):
            # for fold_index in range(4, 5):
            saver = tf.train.Saver(tf.global_variables())
            checkpoint_name = 'ckpt_' + PROG_NAME + '-model_fold=%d' % fold_index
            checkpoint_dir = 'checkpoints_prob'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoint:
                print('Reading model parameters from %s' % checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print('Created model with fresh parameters')
                model.init_step(sess)

            training_q_names, test_q_names = folds[fold_index]
            validation_q_names = np.random.choice(training_q_names, int(len(training_q_names) * 0.20), replace=False)
            validation_q_b_name = {validation_q_names[i]: qbn[validation_q_names[i]] for i in
                                   range(len(validation_q_names))}

            validation_qrels_file = 'validation_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), validation_q_names, validation_qrels_file)

            training_q_names = [n for n in training_q_names if n not in validation_q_names]

            # training_q_b_name = {training_q_names[i]: qbn[training_q_names[i]] for i in range(len(training_q_names))}
            test_q_b_name = {test_q_names[i]: qbn[test_q_names[i]] for i in range(len(test_q_names))}
            test_qrels_file = 'test_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), test_q_names, test_qrels_file)
            max_fold_map = 0.0
            best_epoch = -1
            min_loss = 3
            patience = 0
            for epoch in range(n_epochs):
                print('epoch=%d' % epoch)
                losses = []
                # for i, (data1_max_len, data2_max_len, y, data1_len, data2_len, sim_m) in \
                #         enumerate(du.load_training_batches(output_folder_training_batches, fold_index)):
                for i, (data1_max_len, data2_max_len, y, data1_len, data2_len, sim_m) in \
                        enumerate(
                            du.alt_load_training_batches(output_folder_training_batches, training_q_names, batch_size,
                                                         mql, mdl)):
                    feed_dict = {model.X1_len: data1_len, model.X2_len: data2_len, model.Y: y, model.cross: sim_m,
                                 model.training: True}
                    loss = model.train_step(sess, feed_dict)
                    # print('loss=%2.4f' % loss)
                    losses.append(loss)

                map_v = evaluate_test(sess, model, fold_index, output_folder_valid_data, validation_qrels_file,
                                               str(epoch))

                print('Fold %d, validation map=%2.5f' % (fold_index, map_v))
                if map_v > max_fold_map:
                    max_fold_map = map_v
                    best_epoch = epoch
                    map_v_test = evaluate_test(sess, model, fold_index, output_folder_test_data, test_qrels_file,
                                               str(epoch) + '_TEST')
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        print('Early stopping at epoch=%d, iter=%d (patience=%d)!' % (epoch, i, max_patience))
                        break

                avg_loss = np.mean(np.array(losses))

            print('Max test map in fold=%2.4f, at epoch=%d' % (map_v_test, best_epoch))


if __name__ == '__main__':
    # docs_proc_folder = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/corpus'
    # queries_proc_folder = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/topics'
    # wi_p = 'data/word_index_json'
    # du.encode_collection_pwe(queries_proc_folder, wi_p, 'encoded_queries_pe')
    # du.encode_collection_pwe(docs_proc_folder, wi_p, 'encoded_docs_pe')

    run()
    exit(0)
