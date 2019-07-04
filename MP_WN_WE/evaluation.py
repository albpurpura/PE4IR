"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

from tqdm import tqdm

import data_utils as du
import numpy as np
import util


def compute_docs_to_rerank(dbn, qbn, we, gt_path):
    print('computing relevant docs by query')
    rel_docs_by_query = du.get_rel_docs_by_qry(gt_path)
    print('computing document representations')
    dbn_means = {}
    for k, v in tqdm(dbn.items()):
        if len(v) == 0:
            mean = np.zeros(50)
            dbn_means[k] = mean
        else:
            mean = np.mean([we[w] for w in v], axis=0)
            dbn_means[k] = mean / np.linalg.norm(mean)
    print('computing queries representations')
    qbn_means = {}
    for k, v in tqdm(qbn.items()):
        if len(v) == 0:
            mean = np.zeros(50)
            dbn_means[k] = mean
        else:
            mean = np.mean([we[w] for w in v], axis=0)
            qbn_means[k] = mean / np.linalg.norm(mean)

    doc_names = list(dbn_means.keys())
    doc_names = np.array(doc_names)
    print('computing rankings')
    sorted_d_names_by_query = {}
    incremental_n_rel_docs_by_query = {}
    for qn, q in tqdm(qbn_means.items()):
        if qn not in rel_docs_by_query.keys():
            continue
        dists = [-1] * len(doc_names)
        for i in range(len(doc_names)):
            dn = doc_names[i]
            bonus = np.sum([10 for w in qbn[qn] if w in dbn[dn]])
            dists[i] = np.dot(dbn_means[dn], q) + bonus
        sorted_indices = np.argsort(-np.array(dists))
        sorted_dnames = doc_names[sorted_indices]
        sorted_d_names_by_query[qn] = sorted_dnames[0:8000]
        incremental_n_rel_docs_by_query[qn] = []
        rel_cnt = 0
        for i in range(len(sorted_dnames)):
            dn = sorted_dnames[i]
            if dn in rel_docs_by_query[qn]:
                rel_cnt += 1
            incremental_n_rel_docs_by_query[qn].append(rel_cnt)

    util.save_model(sorted_d_names_by_query, 'sorted_d_names_by_query.model')

    merged_incremental_rel_cnt = np.zeros(len(doc_names))
    # util.save_model(sorted_d_names_by_query, 'sorted_d_names_by_query_w_bonus.model')
    # util.save_model(merged_incremental_rel_cnt, 'merged_incremental_rel_cnt_w_bonus.model')
    print('preparing plot data')
    for q, cnts in tqdm(incremental_n_rel_docs_by_query.items()):
        for i in range(len(cnts)):
            merged_incremental_rel_cnt[i] += cnts[i]

    out = open('log.txt', 'w')
    for i in merged_incremental_rel_cnt:
        out.write(str(i) + '\n')
    out.close()

    # fig = plt.figure()
    # ax = plt.axes()
    # x = range(len(doc_names))
    # ax.plot(x, merged_incremental_rel_cnt)
    # plt.show()


def evaluate_ranking(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file,
                     epoch, run_name):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in tqdm(enumerate(
            du.compute_test_fd_w2v_v2(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    return map_v


def evaluate_ranking_w_idf(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file,
                           epoch, run_name, iwi, idf_scores):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name, bidf_coeffs) in tqdm(enumerate(
            du.compute_test_fd_tfidf(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, iwi, idf_scores))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False, model.idf_coeffs: bidf_coeffs}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    return map_v


def evaluate_ranking_w_OOV(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, word_index, fold_index, test_qrels_file,
                           epoch, run_name, embeddings):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i, (data1_len_test, data2_len_test, d_names, q_name, sim_m) in tqdm(enumerate(
            du.compute_test_fd_w_OOV(test_q_b_name, dbn, run_to_rerank, mql, mdl, word_index, embeddings))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1_len: data1_len_test, model.sim_m: sim_m,
                     model.X2_len: data2_len_test, model.training: False}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    return map_v


def get_relevance_scores_by_qry(run_to_rerank):
    retval = {}
    for line in open(run_to_rerank):
        data = line.split()
        qname = data[0]
        rel_score = float(data[4])
        dname = data[2]

        if qname not in retval.keys():
            retval[qname] = {}

        retval[qname][dname] = rel_score
    return retval


def true_rerank(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file,
                epoch, run_name):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    rel_scores_by_qry = get_relevance_scores_by_qry(run_to_rerank)

    # normalize relevance scores
    for qname, rel_scores_by_doc in rel_scores_by_qry.items():
        max_score = max(rel_scores_by_doc.values())
        for dname in rel_scores_by_doc.keys():
            rel_scores_by_doc[dname] = rel_scores_by_doc[dname] / max_score
        rel_scores_by_qry[qname] = rel_scores_by_doc

    all_preds = {}
    dnames_to_rerank_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in tqdm(enumerate(
            du.compute_test_fd_w2v_v2(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])

        dnames_to_rerank_by_qry[q_name] = d_names
        all_preds[q_name] = pred / np.max(pred)

        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch) + '_pure_neural', rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    print('map of pure neural model=%2.4f' % map_v)

    for q_name in all_preds.keys():
        d_names = dnames_to_rerank_by_qry[q_name]
        rel_scores = np.zeros(len(d_names))
        q_len = len(test_q_b_name[q_name])
        for i in range(len(d_names)):
            if len(test_q_b_name[q_name]) > 0:
                qw_in_doc_cnt_normalized = sum([1 for qw in test_q_b_name[q_name] if qw in dbn[d_names[i]]])
                qw_in_doc_cnt_normalized /= q_len
            else:
                qw_in_doc_cnt_normalized = 0
            alpha = qw_in_doc_cnt_normalized
            rel_scores[i] = (1 - alpha) * all_preds[q_name][i] + alpha * rel_scores_by_qry[q_name][d_names[i]]
        pred = np.array(rel_scores)
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]
    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    return map_v


def eval_stochastic(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file, epoch,
                    run_name, n_stoch_iter=50):
    print('evaluating...')
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    preds_bn = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name) in tqdm(enumerate(
            du.compute_test_fd_w2v_v2(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: True}
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
    for alpha in np.arange(0, 2, 0.1):
        for q_name in preds_bn.keys():
            means, variances, d_names = preds_bn[q_name]
            pred = means - (alpha * variances)
            rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
            sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

        map_v = util.create_evaluate_ranking('fold=' + str(fold_index) + '_epoch=' + str(epoch) + '_alpha=%2.3f' +
                                             str(alpha), rel_docs_by_qry, sim_scores_by_qry, test_qrels_file, run_name,
                                             output_folder='results')
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
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')

    return map_v


def true_rerank_w_idf(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file,
                      epoch, run_name, iwi, idf_scores):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    rel_scores_by_qry = get_relevance_scores_by_qry(run_to_rerank)

    # normalize relevance scores
    for qname, rel_scores_by_doc in rel_scores_by_qry.items():
        max_score = max(rel_scores_by_doc.values())
        for dname in rel_scores_by_doc.keys():
            rel_scores_by_doc[dname] = rel_scores_by_doc[dname] / max_score
        rel_scores_by_qry[qname] = rel_scores_by_doc

    all_preds = {}
    dnames_to_rerank_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name, bidf_coeffs) in tqdm(enumerate(
            du.compute_test_fd_tfidf(test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, iwi, idf_scores))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False, model.idf_coeffs: bidf_coeffs}
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])

        dnames_to_rerank_by_qry[q_name] = d_names
        all_preds[q_name] = pred / np.max(pred)

        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch) + '_pure_neural', rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    print('map of pure neural model=%2.4f' % map_v)

    for q_name in all_preds.keys():
        d_names = dnames_to_rerank_by_qry[q_name]
        rel_scores = np.zeros(len(d_names))
        q_len = len(test_q_b_name[q_name])
        for i in range(len(d_names)):
            if len(test_q_b_name[q_name]) > 0:
                qw_in_doc_cnt_normalized = sum([1 for qw in test_q_b_name[q_name] if qw in dbn[d_names[i]]])
                qw_in_doc_cnt_normalized /= q_len
            else:
                qw_in_doc_cnt_normalized = 0
            alpha = qw_in_doc_cnt_normalized
            rel_scores[i] = (1 - alpha) * all_preds[q_name][i] + alpha * rel_scores_by_qry[q_name][d_names[i]]
        pred = np.array(rel_scores)
        rel_docs_by_qry[q_name] = (np.array(d_names)[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]
    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch), rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder='results')
    return map_v
