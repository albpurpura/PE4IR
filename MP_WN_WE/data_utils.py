"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import multiprocessing
import os
import time

import numpy as np
# import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import util
from fastText import load_model


# cythonize -i data_utils.py

def compute_kfolds_train_test(n_folds, q_names, coll):
    if not os.path.isfile('folds_' + coll):
        folds = []
        q_names = np.array(q_names)
        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(q_names):
            q_train, q_test = q_names[train_index], q_names[test_index]
            folds.append((q_train, q_test))
        util.save_model(folds, 'folds_' + coll)
    else:
        folds = util.load_model('folds_' + coll)
    return folds


def load_indri_stopwords():
    fpath = './data/indri_stoplist_eng.txt'
    sws = []
    for line in open(fpath, 'r'):
        sws.append(line.strip())
    return sws


def load_encoded_collection(encoded_coll_folder):
    encoded_docs_by_name = {}
    for filename in tqdm(os.listdir(encoded_coll_folder)):
        fp = os.path.join(encoded_coll_folder, filename)
        if os.path.isfile(fp):
            encoded_docs_by_name[filename] = util.load_model(fp)
    return encoded_docs_by_name


def get_retrieved_doc_names_for_qry(gt):
    retrieved_docs_by_qry = {}
    for line in tqdm(open(gt)):
        data = line.split()
        qn = data[0].strip()
        dn = data[2].strip()
        rj = int(data[3].strip())
        if rj == 0:
            continue
        if qn in retrieved_docs_by_qry.keys():
            retrieved_docs_by_qry[qn].append(dn)
        else:
            retrieved_docs_by_qry[qn] = [dn]
    return retrieved_docs_by_qry


def compute_test_gt_file(true_gt_file, filtered_docs_names, test_query_names, test_gt_file):
    if not os.path.isfile(test_gt_file):
        gt_docs_by_qry = get_retrieved_doc_names_for_qry(true_gt_file)
        out = open(test_gt_file, 'w')
        to_print = []
        for k in test_query_names:
            line = k + ' ' + ' Q0 '
            if k not in gt_docs_by_qry.keys():
                continue
            rdocs = gt_docs_by_qry[k]
            for d in rdocs:
                if d in filtered_docs_names:
                    to_print.append(line + d + ' 1\n')
        out.writelines(to_print)
        out.close()


def compute_training_batches(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, n_iter_per_query=500,
                             max_q_len=4, max_d_len=500, coll='ny'):
    if not os.path.isfile('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query)):
        rd_b_qry = {}
        nrd_by_qry = {}

        for line in open(gt_file):
            data = line.split()
            qname = data[0].strip()
            dname = data[2].strip()
            if dname not in dbn.keys():
                continue
            rj = int(data[3].strip())
            if qname not in rd_b_qry.keys():
                rd_b_qry[qname] = []
                nrd_by_qry[qname] = []

            if rj > 0:
                rd_b_qry[qname].append(dname)
            else:
                nrd_by_qry[qname].append(dname)
        test_q_names = list(qbn.keys())
        np.random.shuffle(test_q_names)

        qn_rd_nrd_pairs = []
        for qn in test_q_names:
            if qn not in rd_b_qry.keys():
                continue
            tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
            tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
            for i in range(n_iter_per_query):
                qn_rd_nrd_pairs.append((qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll))
    else:
        qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll))

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []

    print('# iterations per epoch: ' + str(len(qn_rd_nrd_pairs)))

    for encoded_q, encoded_rd, encoded_nrd in qn_rd_nrd_pairs:
        batch_pd_lengths.append(len(encoded_rd))
        batch_nd_lengths.append(len(encoded_nrd))
        batch_q_lengths.append(len(encoded_q))
        queries_batch.append(encoded_q)
        rel_docs_batch.append(encoded_rd)
        nrel_docs_batch.append(encoded_nrd)
        if len(queries_batch) == batch_size:
            data1 = []
            data2 = []
            y = []
            data1_len = []
            data2_len = []
            for i in range(len(queries_batch)):
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                y.append(1)
                y.append(0)
                data2.append(pad(rel_docs_batch[i], padding_value, max_d_len))
                data2.append(pad(nrel_docs_batch[i], padding_value, max_d_len))
                data1_len.append(batch_q_lengths[i])
                data1_len.append(batch_q_lengths[i])
                data2_len.append(batch_pd_lengths[i])
                data2_len.append(batch_nd_lengths[i])
            yield max_q_len, max_d_len, data1, data2, y, data1_len, data2_len

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []


def compute_training_batches_w_OOV(gt_file, dbn, qbn, batch_size, fold_idx, word_index,
                                   embeddings, n_iter_per_query=500, max_q_len=4, max_d_len=500, coll='ny'):
    if not os.path.isfile('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query)):
        rd_b_qry = {}
        nrd_by_qry = {}

        for line in open(gt_file):
            data = line.split()
            qname = data[0].strip()
            dname = data[2].strip()
            if dname not in dbn.keys():
                continue
            rj = int(data[3].strip())
            if qname not in rd_b_qry.keys():
                rd_b_qry[qname] = []
                nrd_by_qry[qname] = []

            if rj > 0:
                rd_b_qry[qname].append(dname)
            else:
                nrd_by_qry[qname].append(dname)
        test_q_names = list(qbn.keys())
        np.random.shuffle(test_q_names)

        qn_rd_nrd_pairs = []
        for qn in test_q_names:
            if qn not in rd_b_qry.keys():
                continue
            tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
            tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
            for i in range(n_iter_per_query):
                qn_rd_nrd_pairs.append((qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs,
                        'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query))
    else:
        qn_rd_nrd_pairs = util.load_model(
            'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query))

    queries_merged_batches = []
    docs_merged_batches = []
    y = []
    data1_len = []
    data2_len = []

    pool = multiprocessing.Pool(10)
    for encoded_q, encoded_rd, encoded_nrd in tqdm(qn_rd_nrd_pairs):

        queries_merged_batches.append(encoded_q)
        queries_merged_batches.append(encoded_q)

        data1_len.append(len(encoded_q))
        data1_len.append(len(encoded_q))

        docs_merged_batches.append(encoded_rd)
        data2_len.append(len(encoded_rd))

        docs_merged_batches.append(encoded_nrd)
        data2_len.append(len(encoded_nrd))

        y.append(1)
        y.append(0)

        if len(docs_merged_batches) == batch_size * 2:
            start = time.time()
            sim_batches = pool.starmap(compute_sim_m_w_OOV_alt,
                                       [(queries_merged_batches[i], docs_merged_batches[i], max_q_len, max_d_len,
                                         word_index,
                                         embeddings)
                                        for i in range(len(docs_merged_batches))])
            print('time to compute the sim matrices = %2.3fs' % (time.time() - start))

            for i in range(len(queries_merged_batches)):
                data1_len.append(len(queries_merged_batches[i]))
                data2_len.append(len(docs_merged_batches[i]))

            yield max_q_len, max_d_len, y, data1_len, data2_len, sim_batches

            queries_merged_batches = []
            docs_merged_batches = []
            y = []
            data1_len = []
            data2_len = []


def compute_training_batches_w_idf_weights(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, iwi, idf_scores,
                                           n_iter_per_query=500, max_q_len=4, max_d_len=500, coll='ny'):
    if not os.path.isfile('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query)):
        rd_b_qry = {}
        nrd_by_qry = {}

        for line in open(gt_file):
            data = line.split()
            qname = data[0].strip()
            dname = data[2].strip()
            if dname not in dbn.keys():
                continue
            rj = int(data[3].strip())
            if qname not in rd_b_qry.keys():
                rd_b_qry[qname] = []
                nrd_by_qry[qname] = []

            if rj > 0:
                rd_b_qry[qname].append(dname)
            else:
                nrd_by_qry[qname].append(dname)
        test_q_names = list(qbn.keys())
        np.random.shuffle(test_q_names)

        qn_rd_nrd_pairs = []
        for qn in test_q_names:
            if qn not in rd_b_qry.keys():
                continue
            tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
            tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
            for i in range(n_iter_per_query):
                qn_rd_nrd_pairs.append((qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs,
                        'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query))
    else:
        qn_rd_nrd_pairs = util.load_model(
            'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query))

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []
    batch_idf_coeffs = []
    for encoded_q, encoded_rd, encoded_nrd in qn_rd_nrd_pairs:
        batch_pd_lengths.append(len(encoded_rd))
        batch_nd_lengths.append(len(encoded_nrd))
        batch_q_lengths.append(len(encoded_q))
        queries_batch.append(encoded_q)
        rel_docs_batch.append(encoded_rd)
        nrel_docs_batch.append(encoded_nrd)
        if len(queries_batch) == batch_size:
            data1 = []
            data2 = []
            y = []
            data1_len = []
            data2_len = []
            for i in range(len(queries_batch)):
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                y.append(1)
                y.append(0)
                data2.append(pad(rel_docs_batch[i], padding_value, max_d_len))
                data2.append(pad(nrel_docs_batch[i], padding_value, max_d_len))
                data1_len.append(batch_q_lengths[i])
                data1_len.append(batch_q_lengths[i])
                data2_len.append(batch_pd_lengths[i])
                data2_len.append(batch_nd_lengths[i])

                batch_idf_coeffs.append(
                    compute_idf_scaling_coeffs(queries_batch[i], iwi, idf_scores, max_q_len, max_d_len))
                batch_idf_coeffs.append(
                    compute_idf_scaling_coeffs(queries_batch[i], iwi, idf_scores, max_q_len, max_d_len))
                # batch_idf_coeffs.append(compute_idf_scaling_coeffs_doc_query(queries_batch[i], rel_docs_batch[i], iwi,
                #                                                              idf_scores, max_q_len, max_d_len))
                # batch_idf_coeffs.append(compute_idf_scaling_coeffs_doc_query(queries_batch[i], nrel_docs_batch[i], iwi,
                #                                                              idf_scores, max_q_len, max_d_len))

            yield max_q_len, max_d_len, data1, data2, y, data1_len, data2_len, batch_idf_coeffs

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []
            batch_idf_coeffs = []


def compute_idf_scaling_coeffs(encoded_q, iwi, idf_scores, mql, mdl):
    coeffs = np.ones((mql, mdl))
    for i in range(len(encoded_q)):
        if int(encoded_q[i]) in iwi.keys():
            w = iwi[int(encoded_q[i])]
            coeffs[i] = np.ones(mdl) * idf_scores[w]
    return coeffs


def compute_idf_scaling_coeffs_doc_query(encoded_q, encoded_d, iwi, idf_scores, mql, mdl):
    coeffs = np.ones((mql, mdl))
    # text_query = ' '.join([iwi[int(encoded_q[i])] for i in range(len(encoded_q))])
    # print('Query: ' + text_query)
    for i in range(len(encoded_q)):
        if int(encoded_q[i]) in iwi.keys():
            qw = iwi[int(encoded_q[i])]
            for j in range(len(encoded_d)):
                if int(encoded_d[j]) in iwi.keys():
                    dw = iwi[int(encoded_d[j])]
                    coeffs[i, j] = 1.0 * idf_scores[dw]
                    # coeffs[i, j] = 1.0 * idf_scores[qw] * idf_scores[dw]
                else:
                    coeffs[i, j] = 1.0 * idf_scores[qw]
    return coeffs


def pad(to_pad, padding_value, max_len):
    retval = np.ones(max_len) * padding_value
    for i in range(len(to_pad)):
        retval[i] = to_pad[i]
    return retval


def compute_docs_to_rerank_by_query(run_to_rerank, test_query_names):
    docs_to_rerank_by_query = {}
    for line in open(run_to_rerank, 'r'):
        data = line.split()
        qname = data[0]
        dname = data[2]
        if qname in test_query_names:
            if qname in docs_to_rerank_by_query.keys():
                docs_to_rerank_by_query[qname].append(dname)
            else:
                docs_to_rerank_by_query[qname] = [dname]
    return docs_to_rerank_by_query


def compute_test_fd_w2v_v2(qbn, dbn, run_to_rerank, max_q_len, max_d_len, padding_value):
    d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
    for q_name in qbn.keys():
        if q_name not in d_t_rerank_by_query.keys():
            continue
        d_names = d_t_rerank_by_query[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        padded_d = [pad(d, padding_value, max_d_len) for d in docs]
        q = pad(qbn[q_name], padding_value, max_q_len)
        yield [q] * len(padded_d), padded_d, [len(qbn[q_name])] * len(padded_d), d_lengths, d_names, q_name


def compute_test_fd_w_OOV(qbn, dbn, run_to_rerank, max_q_len, max_d_len, word_index, embeddings):
    pool = multiprocessing.Pool(10)
    d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
    for q_name in qbn.keys():
        if q_name not in d_t_rerank_by_query.keys():
            continue
        d_names = d_t_rerank_by_query[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        sim_matrices = []
        for i in range(len(docs)):
            # sim_m, qlen, pdlen = compute_sim_m_w_OOV(qbn[q_name], docs[i], max_q_len, max_d_len, word_index, embeddings)
            sim_m = pool.starmap(compute_sim_m_w_OOV_alt,
                                 [(qbn[q_name], docs[i], max_q_len, max_d_len, word_index, embeddings)
                                  for i in range(len(docs))])
            sim_matrices.append(sim_m)
        yield [len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, sim_matrices


def compute_test_fd_tfidf(qbn, dbn, run_to_rerank, max_q_len, max_d_len, padding_value, iwi, idf_scores):
    d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
    for q_name in qbn.keys():
        if q_name not in d_t_rerank_by_query.keys():
            continue
        d_names = d_t_rerank_by_query[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        padded_d = [pad(d, padding_value, max_d_len) for d in docs]
        q = pad(qbn[q_name], padding_value, max_q_len)
        idf_coeffs = [compute_idf_scaling_coeffs(q, iwi, idf_scores, max_q_len, max_d_len)] * len(padded_d)
        # idf_coeffs = []
        # for d in docs:
        #     idf_coeffs.append(compute_idf_scaling_coeffs_doc_query(q, d, iwi, idf_scores, max_q_len, max_d_len))
        yield [q] * len(padded_d), padded_d, [len(qbn[q_name])] * len(padded_d), d_lengths, d_names, q_name, idf_coeffs


def encode_collection_with_stemming(text_by_name_p, word_dict_path, w2v_model_path, encoded_out_folder, wi=None,
                                    word_embeddings_matrix=None):
    text_by_name = {}
    print('reading files in folder')
    for filename in tqdm(os.listdir(text_by_name_p)):
        fp = os.path.join(text_by_name_p, filename)
        if os.path.isfile(fp):
            text_by_name[filename.split(r'.')[0]] = ' '.join(open(fp, 'r').readlines())

    # initialize embeddings matrix
    if word_embeddings_matrix is None:
        # read and adapt word index
        if wi is None:
            wi = {}
            wids_to_merge = {}
            for line in tqdm(open(word_dict_path)):
                data = line.split()
                word_stemmed = util.stem(data[0].strip())
                wid = int(data[1].strip())
                if word_stemmed not in wi.keys():
                    wi[word_stemmed] = len(wi)
                    wids_to_merge[word_stemmed] = [wid]
                else:
                    wids_to_merge[word_stemmed].append(wid)
        we_size = 50
        word_embeddings_matrix = np.float32(np.random.uniform(-0.02, 0.02, [len(wi) + 1, we_size]))
        padding_value = np.zeros(we_size)
        word_embeddings_matrix[word_embeddings_matrix.shape[0] - 1] = padding_value
        w2v_model = load_w2v_we(w2v_model_path)
        for k, v in wi.items():
            we = np.zeros(we_size)
            summed_something = False
            for wid in wids_to_merge[k]:
                if wid in w2v_model.keys():
                    we = np.sum((we, w2v_model[wid]), axis=0)
                    summed_something = True
            if summed_something:
                we = we / np.linalg.norm(we)  # normalize new word embedding
                word_embeddings_matrix[v] = we

    encoded_docs_by_name = {}
    sw = load_indri_stopwords()
    print('encoding data')
    for dn, dc in tqdm(text_by_name.items()):
        td = util.tokenize(dc, stemming=True, stoplist=sw)
        encoded_doc = [wi[w] for w in td if w in wi.keys()]
        util.save_model(encoded_doc, os.path.join(encoded_out_folder, dn))
        encoded_docs_by_name[dn] = encoded_doc
    return encoded_docs_by_name, wi, word_embeddings_matrix


def load_w2v_we(w2v_model_path):
    w = {}
    for line in open(w2v_model_path):
        data = line.split()
        wid = int(data[0])
        we = np.array([float(i) for i in data[1:]])
        w[wid] = we
    return w


def get_we_matrix_wi_encode_docs_w_fasttext(ftext_model_path, docs_text_main_folder, encoded_out_folder_docs):
    f = load_model(ftext_model_path)
    text_by_name = {}
    print('reading files in folder')
    for filename in tqdm(os.listdir(docs_text_main_folder)):
        fp = os.path.join(docs_text_main_folder, filename)
        if os.path.isfile(fp):
            text_by_name[filename.split(r'.')[0]] = ' '.join(open(fp, 'r').readlines())
    stoplist = load_indri_stopwords()

    print('encoding collection')
    encoded_docs_by_name = {}
    wi = {}
    we_matrix = []
    for dn, dt in tqdm(text_by_name.items()):
        tok_doc = util.tokenize(dt, stemming=False, stoplist=stoplist)
        encoded_doc = []
        for tok in tok_doc:
            if tok not in wi.keys():
                wv = f.get_word_vector(tok)
                wi[tok] = len(wi)
                we_matrix.append(wv)
            encoded_doc.append(wi[tok])
        util.save_model(encoded_doc, os.path.join(encoded_out_folder_docs, dn))
        encoded_docs_by_name[dn] = encoded_doc
    return encoded_docs_by_name, wi, we_matrix


def encode_coll(docs_text_path, wi, output_encoded_coll_path):
    text_by_name = {}
    print('reading files in folder')
    for filename in tqdm(os.listdir(docs_text_path)):
        fp = os.path.join(docs_text_path, filename)
        if os.path.isfile(fp):
            text_by_name[filename.split(r'.')[0]] = ' '.join(open(fp, 'r').readlines())
    stoplist = load_indri_stopwords()
    encoded_coll_by_name = {}
    print('encoding collection')
    for tn, tt in tqdm(text_by_name.items()):
        tokenized = util.tokenize(tt, stemming=False, stoplist=stoplist)
        encoded_text = [wi[t] for t in tokenized if t in wi.keys()]
        encoded_coll_by_name[tn] = encoded_text
        util.save_model(encoded_text, os.path.join(output_encoded_coll_path, tn))
    return encoded_coll_by_name


def precompute_data(docs_proc_folder, queries_proc_folder, word_dict_path, w2v_model_path, encoded_out_folder_docs,
                    encoded_out_folder_queries, output_path_wi_model, output_path_we_matrix_model,
                    output_path_encoded_q,
                    output_path_encoded_d, run_to_rerank, gt_file):
    # docs_proc_folder = '/media/alberto/DATA/ExperimentalCollections/ny/ny/nyt_proc_albe_2'
    # queries_proc_folder = '/media/alberto/DATA/ExperimentalCollections/ny/queries/queries_proc'
    # word_dict_path = 'data/word_dict.txt'
    # w2v_model_path = 'data/embed_wiki-pdc_d50_norm'
    # encoded_out_folder_docs = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/encoded_docs'
    # encoded_out_folder_queries = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/encoded_queries'
    # output_path_wi_model = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/word_index_stemmed'
    # output_path_we_matrix_model = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/word_embeddings_matrix'
    # output_path_encoded_q = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/q_by_name'
    # output_path_encoded_d = '/media/alberto/DATA/ExperimentalCollections/ny/encoded_corpus/d_by_name_filtered'

    dbn, wi, word_embeddings_matrix = encode_collection_with_stemming(docs_proc_folder, word_dict_path,
                                                                      w2v_model_path, encoded_out_folder_docs)
    util.save_model(wi, output_path_wi_model)
    util.save_model(word_embeddings_matrix, output_path_we_matrix_model)

    qbn, wi, word_embeddings_matrix = encode_collection_with_stemming(queries_proc_folder, word_dict_path,
                                                                      w2v_model_path, encoded_out_folder_queries, wi,
                                                                      word_embeddings_matrix)

    dbn_filtered = keep_only_used_docs(gt_file, run_to_rerank, encoded_out_folder_docs)
    util.save_model(qbn, output_path_encoded_q)
    util.save_model(dbn_filtered, output_path_encoded_d)


def precompute_data_w_fasttext(docs_main_folder, queries_main_folder, fastext_model_path, output_path_wi_model,
                               output_path_we_matrix_model, gt_file, run_to_rerank, encoded_out_folder_docs,
                               output_path_encoded_q, output_path_encoded_d):
    encoded_docs_by_name, wi, we_matrix = get_we_matrix_wi_encode_docs_w_fasttext(fastext_model_path, docs_main_folder,
                                                                                  output_path_encoded_d)
    encoded_queries_by_name = encode_coll(queries_main_folder, wi, output_path_encoded_q)
    util.save_model(wi, output_path_wi_model)
    util.save_model(we_matrix, output_path_we_matrix_model)

    dbn_filtered = keep_only_used_docs(gt_file, run_to_rerank, encoded_out_folder_docs)
    util.save_model(encoded_queries_by_name, output_path_encoded_q)
    util.save_model(dbn_filtered, output_path_encoded_d)


def keep_only_used_docs(gt_file, run_to_rerank, encoded_docs_folder):
    dbn_filtered = {}
    nskipped_rel_docs = 0
    for line in open(gt_file):
        data = line.split()
        dname = data[2].strip()
        rel_or_not = int(data[-1])

        if dname not in dbn_filtered.keys():
            if not os.path.isfile(os.path.join(encoded_docs_folder, dname)) and rel_or_not > 0:
                nskipped_rel_docs += 1
            elif os.path.isfile(os.path.join(encoded_docs_folder, dname)):
                dbn_filtered[dname] = util.load_model(os.path.join(encoded_docs_folder, dname))

    print('n skipped rel docs: ' + str(nskipped_rel_docs))

    for line in open(run_to_rerank, encoding='latin-1'):
        data = line.split()
        dname = data[2].strip()
        if dname not in dbn_filtered.keys():
            dbn_filtered[dname] = util.load_model(os.path.join(encoded_docs_folder, dname))
    return dbn_filtered


def compute_idf(coll_folder, stemming, output_file_path_ii, output_file_path_idf):
    if not os.path.isfile(output_file_path_idf):
        print('computing inverted index')
        inverted_idx = {}
        sw = load_indri_stopwords()
        doc_n = 0
        for filename in tqdm(os.listdir(coll_folder)):
            fp = os.path.join(coll_folder, filename)
            doc_id = filename.split(r'.')[0]
            if os.path.isfile(fp):
                doc_n += 1
                d = util.tokenize(' '.join(open(fp, 'r').readlines()), stemming, stoplist=sw)
                set_w_in_doc = set(d)
                for w in set_w_in_doc:
                    if w in inverted_idx.keys():
                        inverted_idx[w].append((doc_id, d.count(w)))
                    else:
                        inverted_idx[w] = [(doc_id, d.count(w))]

        util.save_model(inverted_idx, output_file_path_ii)
        print('computing inverse document frequencies')
        words = list(inverted_idx.keys())
        d_freqs = {}
        for k in tqdm(words):
            v = inverted_idx[k]
            df = len(v)
            d_freqs[k] = df

        idf_scores = {}
        for w in words:
            idf_scores[w] = np.log((1 + doc_n) / (d_freqs[w] + 1))

        util.save_model(idf_scores, output_file_path_idf)
    else:
        idf_scores = util.load_model(output_file_path_idf)

    return idf_scores


def compute_sim_m_w_OOV_alt(tok_query_text, tok_doc_text, mql, mdl, word_index, embeddings):
    sim_m = np.zeros((mql, mdl))
    query_len = len(tok_query_text)
    doc_len = len(tok_doc_text)
    for i in range(query_len):
        qw_code = word_index.get(tok_query_text[i])
        for j in range(doc_len):
            # start = time.time()
            dw_code = word_index.get(tok_doc_text[j])
            # print('time to search for a word in word index: ' + str(time.time() - start))
            if qw_code is not None and dw_code is not None:
                qw_emb = embeddings[qw_code]
                dw_emb = embeddings[dw_code]
                # start = time.time()
                cos_sim = np.dot(qw_emb, dw_emb) / (np.linalg.norm(qw_emb) * np.linalg.norm(dw_emb))
                # print('time to compute the dot product: ' + str(time.time() - start))
                sim_m[i, j] = cos_sim
            else:
                if tok_query_text[i] == tok_doc_text[j]:
                    sim_m[i, j] = 1.0
    return sim_m


def get_tokenized_text_by_name(text_by_name_p, stemming, output_model_path):
    if not os.path.isfile(output_model_path):
        sw = load_indri_stopwords()
        text_by_name = {}
        print('reading files in folder')
        for filename in tqdm(os.listdir(text_by_name_p)):
            fp = os.path.join(text_by_name_p, filename)
            if os.path.isfile(fp):
                text_by_name[filename.split(r'.')[0]] = util.tokenize(' '.join(open(fp, 'r').readlines()),
                                                                      stemming=stemming,
                                                                      stoplist=sw)
        util.save_model(text_by_name, output_model_path)
    else:
        text_by_name = util.load_model(output_model_path)

    max_text_len = 0
    for k, v in text_by_name.items():
        if max_text_len < len(v):
            max_text_len = len(v)

    return text_by_name, max_text_len



