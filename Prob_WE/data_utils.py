"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import os
import numpy as np
# import torch
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import util
from wasserstein.operators import BuresProductNormalizedModule
from wasserstein.operators import BuresProductNormalized


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


def compute_sim_matrices_w2v(queries, documents, max_q_len, max_d_len, w):
    sim_matrices = []
    for i in range(len(queries)):
        q = queries[i]
        d = documents[i]
        if len(d) > 0 and len(q) > 0:
            tmp_sim_matrix = np.zeros(shape=(max_q_len, max_d_len))
            for j in range(len(q)):
                for k in range(len(d)):
                    if q[j] in w and d[k] in w:
                        tmp_sim_matrix[j, k] = np.dot(w[q[j]], w[d[k]])
        else:
            tmp_sim_matrix = np.zeros(shape=(max_q_len, max_d_len))
        sim_matrices.append(tmp_sim_matrix)
    return sim_matrices


def sample_neg_doc(all_d_names, rel_doc_names_to_avoid):
    while True:
        neg_d_name = np.random.choice(all_d_names, 1)[0]
        if neg_d_name not in rel_doc_names_to_avoid:
            return neg_d_name


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


def pad(to_pad, padding_value, max_len):
    retval = np.ones(max_len) * padding_value
    for i in range(len(to_pad)):
        retval[i] = to_pad[i]
    return retval


def alt_compute_training_batches_w2v(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, n_iter_per_query=500,
                                     max_q_len=4, max_d_len=500, we=None):
    if not os.path.isfile('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx)):
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
        util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_w2v_gk' + str(fold_idx))
    else:
        qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_w2v_gk' + str(fold_idx))

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []

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
                data2_len.append(batch_pd_lengths[i])
                data2_len.append(batch_nd_lengths[i])

                # test_compute_sim_m_tf(data1, data1_len, data2, data2_len, queries_batch, rel_docs_batch, max_q_len, max_d_len,
                #                       we)
            yield max_q_len, max_d_len, data1, data2, y, data1_len, data2_len

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []


def pre_compute_test_fd_pwe(qbn, dbn, w, run_to_rerank, max_q_len, max_d_len, fold, output_folder):
    # test_fd = []
    test_fd_fp = os.path.join(output_folder, 'test_fd_' + str(fold))
    if not os.path.isfile(test_fd_fp):
        d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
        for q_name in tqdm(qbn.keys()):
            if os.path.isfile(test_fd_fp + '_' + q_name):
                continue
            if q_name not in d_t_rerank_by_query.keys():
                continue
            d_names = d_t_rerank_by_query[q_name]
            docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
            d_lengths = [len(d) for d in docs]
            # padded_d = [pad(d, padding_value, max_d_len) for d in docs]
            q = qbn[q_name]
            all_sim_m = []
            d_batch = []
            for d in docs:
                d_batch.append(d)
                if len(d_batch) == 16:
                    all_sim_m.extend(
                        parallel_compute_sim_matrices([q] * len(d_batch), d_batch, max_q_len, max_d_len, w))
                    d_batch = []
            # test_fd.append(([len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, all_sim_m))
            util.save_model(([len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, all_sim_m),
                            test_fd_fp + '_' + q_name)
        # util.save_model(test_fd, test_fd_fp)


def load_test_fd_pwe(fold, data_folder):
    # test_fd = []
    # test_fd_fp = 'test_fd_' + str(fold)
    for filename in os.listdir(data_folder):
        fp = os.path.join(data_folder, filename)
        if '_fd_' + str(fold) in filename:
            if os.path.isfile(fp):
                yield util.load_model(fp)

    # if not os.path.isfile(test_fd_fp):
    #     d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
    #     print('Evaluating model')
    #     for q_name in tqdm(qbn.keys()):
    #         if q_name not in d_t_rerank_by_query.keys():
    #             continue
    #         d_names = d_t_rerank_by_query[q_name]
    #         docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
    #         d_lengths = [len(d) for d in docs]
    #         # padded_d = [pad(d, padding_value, max_d_len) for d in docs]
    #         q = qbn[q_name]
    #         all_sim_m = []
    #         d_batch = []
    #         for d in tqdm(docs):
    #             d_batch.append(d)
    #             if len(d_batch) == 42:
    #                 all_sim_m.extend(
    #                     parallel_compute_sim_matrices([q] * len(d_batch), d_batch, max_q_len, max_d_len, w))
    #                 d_batch = []
    #         test_fd.append(([len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, all_sim_m))
    #     util.save_model(test_fd, test_fd_fp)
    # else:
    #     test_fd = util.load_model(test_fd_fp)
    #
    # for item in test_fd:
    #     yield item



def compute_test_fd_pwe(qbn, dbn, w, run_to_rerank, max_q_len, max_d_len):
    d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
    print('Evaluating model')
    for q_name in tqdm(qbn.keys()):
        if q_name not in d_t_rerank_by_query.keys():
            continue
        d_names = d_t_rerank_by_query[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        # padded_d = [pad(d, padding_value, max_d_len) for d in docs]
        q = qbn[q_name]
        sim_m = compute_sim_matrices([q] * len(docs), docs, max_q_len, max_d_len, w)
        # sim_m = parallel_compute_sim_matrices([q] * len(docs), docs, max_q_len, max_d_len, w)

        yield [len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, sim_m


def get_indices_q(d_len, q_len):
    l = []
    for i in range(q_len):
        l.append(torch.ones(d_len) * i)
    return torch.cat(l).long()


def get_indices_d(d_len, q_len):
    l = []
    for i in range(q_len):
        l.append(torch.arange(d_len))
    return torch.cat(l).long()



def compute_sim_matrices(queries, documents, max_q_len, max_d_len, w):
    metric = BuresProductNormalized()
    sim_matrices = []
    dim = 50

    for i in range(len(queries)):
        q = queries[i]
        qm, qc = (w[q, 0:dim].view(-1, dim), w[q, dim:].view((-1, dim, dim)))
        d = documents[i]
        if len(d) > 0 and len(q) > 0:
            q_ind = get_indices_q(len(d), len(q))
            d_ind = get_indices_d(len(d), len(q))

            embedded_dm, embedded_dc = (w[d, 0:dim].view(-1, dim), w[d, dim:].view((-1, dim, dim)))
            qm_replicated = qm[q_ind]
            qc_replicated = qc[q_ind]

            dm_replicated = embedded_dm[d_ind]
            dc_replicated = embedded_dc[d_ind]

            similarities = metric(qm_replicated, dm_replicated, qc_replicated, dc_replicated)
            similarities = np.reshape(np.array(similarities.cpu().detach().numpy()), newshape=-1)
            tmp_sim_matrix = np.zeros(shape=(max_q_len, max_d_len))
            for j in range(len(q)):
                for k in range(len(d)):
                    tmp_sim_matrix[j, k] = similarities[j * (len(d)) + k]
        else:
            tmp_sim_matrix = np.zeros(shape=(max_q_len, max_d_len))
        sim_matrices.append(tmp_sim_matrix)
    return sim_matrices


def parallel_compute_sim_matrices(queries, documents, max_q_len, max_d_len, w):
    # metric = BuresProductNormalized()
    metric = torch.nn.DataParallel(BuresProductNormalizedModule())
    dim = 50
    non_zero_q_docs_pairs = []
    nzqdp_indices = []
    sim_matrices = []
    q_lengths = []
    d_lengths = []
    for i in range(len(queries)):
        tmp_sim_matrix = np.zeros(shape=(max_q_len, max_d_len))
        sim_matrices.append(tmp_sim_matrix)
        q = queries[i]
        d = documents[i]
        if len(d) > 0 and len(q) > 0:
            non_zero_q_docs_pairs.append((q, d))
            nzqdp_indices.append(i)
            q_lengths.append(len(q))
            d_lengths.append(len(d))

    qm_replicated_batch = []
    qc_replicated_batch = []
    dm_replicated_batch = []
    dc_replicated_batch = []

    for i in range(len(non_zero_q_docs_pairs)):
        q, d = non_zero_q_docs_pairs[i]
        qm, qc = (w[q, 0:dim].view(-1, dim), w[q, dim:].view((-1, dim, dim)))
        embedded_dm, embedded_dc = (w[d, 0:dim].view(-1, dim), w[d, dim:].view((-1, dim, dim)))
        q_ind = get_indices_q(len(d), len(q))
        d_ind = get_indices_d(len(d), len(q))
        qm_replicated = qm[q_ind]
        qc_replicated = qc[q_ind]
        dm_replicated = embedded_dm[d_ind]
        dc_replicated = embedded_dc[d_ind]

        qm_replicated_batch.extend(qm_replicated)
        qc_replicated_batch.extend(qc_replicated)
        dm_replicated_batch.extend(dm_replicated)
        dc_replicated_batch.extend(dc_replicated)
    if len(qm_replicated_batch) > 0:
        qm_replicated_batch = torch.cat(qm_replicated_batch).reshape(-1, 50)
        qc_replicated_batch = torch.cat(qc_replicated_batch).reshape(-1, 50, 50)
        dm_replicated_batch = torch.cat(dm_replicated_batch).reshape(-1, 50)
        dc_replicated_batch = torch.cat(dc_replicated_batch).reshape(-1, 50, 50)

        similarities = metric(qm_replicated_batch, dm_replicated_batch, qc_replicated_batch, dc_replicated_batch)
        similarities = np.reshape(np.array(similarities.cpu().detach().numpy()), newshape=-1)

        assert len(similarities) == sum([len(p[0]) * len(p[1]) for p in non_zero_q_docs_pairs])

        pos = 0
        offset = 0
        for i in range(len(non_zero_q_docs_pairs)):
            curr_sim_m = sim_matrices[nzqdp_indices[i]]
            q, d = non_zero_q_docs_pairs[i]
            len_q = len(q)
            len_d = len(d)
            for k in range(len_q):
                for j in range(len_d):
                    curr_sim_m[k, j] = similarities[offset + k * len_d + j]
                    pos = offset + k * len_d + j
            offset += len_q * len_d
            sim_matrices[nzqdp_indices[i]] = curr_sim_m

        assert pos == len(similarities) - 1

    return sim_matrices


def load_training_batches(batches_folder, fold_idx):
    for filename in os.listdir(batches_folder):
        fp = os.path.join(batches_folder, filename)
        if '_fold=' + str(fold_idx) in fp:
            if os.path.isfile(fp):
                yield util.load_model(fp)


def alt_load_training_batches(batches_folder, training_qnames, batch_size, max_q_len, max_d_len):
    pairs = []
    print('loading training batches')
    for filename in tqdm(os.listdir(batches_folder)):
        fp = os.path.join(batches_folder, filename)
        tqn = filename.split('_')[-1].replace('qn=', '')
        if tqn in training_qnames:
            if os.path.isfile(fp):
                pair = util.load_model(fp)
                pairs.append(pair)
                # for p in pair:
                #     max_q_len_b.append(max_q_len)
                #     max_d_len_b.append(max_d_len)
                #     y_b.append(p[1])
                #     b_q_len.append(p[3])
                #     b_d_len.append(p[4])
                #     b_sm.append(p[2])
                # if len(max_q_len_b) == 2 * batch_size:
                #     yield (max_q_len_b, max_d_len_b, y_b, b_q_len, b_d_len, b_sm)
                #     max_q_len_b = []
                #     max_d_len_b = []
                #     y_b = []
                #     b_q_len = []
                #     b_d_len = []
                #     b_sm = []
    max_q_len_b = []
    max_d_len_b = []
    y_b = []
    b_q_len = []
    b_d_len = []
    b_sm = []
    for pair in pairs:
        for p in pair:
            max_q_len_b.append(max_q_len)
            max_d_len_b.append(max_d_len)
            y_b.append(p[1])
            b_q_len.append(p[3])
            b_d_len.append(p[4])
            b_sm.append(p[2])
        if len(max_q_len_b) == 2 * batch_size:
            yield (max_q_len_b, max_d_len_b, y_b, b_q_len, b_d_len, b_sm)
            max_q_len_b = []
            max_d_len_b = []
            y_b = []
            b_q_len = []
            b_d_len = []
            b_sm = []


#
# def pre_compute_pwe_training_batches(output_folder, w, gt_file, dbn, qbn, comp_batch_size, fold_idx,
#                                      n_iter_per_query=500,
#                                      max_q_len=4, max_d_len=500):
#     if not os.path.isfile('qn_rd_nrd_pairs_pwe_test_' + str(fold_idx)):
#         rd_b_qry = {}
#         nrd_by_qry = {}
#
#         for line in open(gt_file):
#             data = line.split()
#             qname = data[0].strip()
#             dname = data[2].strip()
#             if dname not in dbn.keys():
#                 continue
#             rj = int(data[3].strip())
#             if qname not in rd_b_qry.keys():
#                 rd_b_qry[qname] = []
#                 nrd_by_qry[qname] = []
#
#             if rj > 0:
#                 rd_b_qry[qname].append(dname)
#             else:
#                 nrd_by_qry[qname].append(dname)
#         test_q_names = list(qbn.keys())
#         np.random.shuffle(test_q_names)
#
#         qn_rd_nrd_pairs = []
#         for qn in test_q_names:
#             if qn not in rd_b_qry.keys():
#                 continue
#             tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
#             tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
#             for i in range(n_iter_per_query):
#                 qn_rd_nrd_pairs.append((qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))
#
#         np.random.shuffle(qn_rd_nrd_pairs)
#         util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_pwe_test_' + str(fold_idx))
#     else:
#         qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_pwe_test_' + str(fold_idx))
#
#     queries_batch = []
#     rel_docs_batch = []
#     nrel_docs_batch = []
#     batch_pd_lengths = []
#     batch_nd_lengths = []
#     batch_q_lengths = []
#     cnt = 0
#     for encoded_q, encoded_rd, encoded_nrd in tqdm(qn_rd_nrd_pairs):
#         batch_pd_lengths.append(len(encoded_rd))
#         batch_nd_lengths.append(len(encoded_nrd))
#         batch_q_lengths.append(len(encoded_q))
#         queries_batch.append(encoded_q)
#         rel_docs_batch.append(encoded_rd)
#         nrel_docs_batch.append(encoded_nrd)
#         if len(queries_batch) == comp_batch_size:
#             # print('computing similarity matrices')
#             sim_matrices_p = compute_sim_matrices(queries_batch, rel_docs_batch, max_q_len, max_d_len, w)
#             sim_matrices_n = compute_sim_matrices(queries_batch, nrel_docs_batch, max_q_len, max_d_len, w)
#
#             # y = []
#             # data1_len = []
#             # data2_len = []
#             # sim_m = []
#             for i in range(len(queries_batch)):
#                 # y.append(1)
#                 # y.append(0)
#                 # sim_m.append(sim_matrices_p[i])
#                 # sim_m.append(sim_matrices_n[i])
#                 # data1_len.append(batch_q_lengths[i])
#                 # data2_len.append(batch_pd_lengths[i])
#                 # data2_len.append(batch_nd_lengths[i])
#
#                 data_to_save = [(1, sim_matrices_p[i], batch_q_lengths[i], batch_pd_lengths[i]),
#                                 (0, sim_matrices_n[i], batch_q_lengths[i], batch_nd_lengths[i])]
#                 util.save_model(data_to_save,
#                                 os.path.join(output_folder, 'batch_data_item_fold=%d_#%d' % (fold_idx, cnt)))
#                 cnt += 1
#
#             # util.save_model((max_q_len, max_d_len, y, data1_len, data2_len, sim_m),
#             #                 os.path.join(output_folder, 'batch_data_fold=%d_%d' % (fold_idx, cnt)))
#             # cnt += 1
#
#             queries_batch = []
#             rel_docs_batch = []
#             nrel_docs_batch = []
#             batch_pd_lengths = []
#             batch_nd_lengths = []
#             batch_q_lengths = []


def pre_compute_pwe_training_batches_overall(output_folder, w, gt_file, dbn, qbn, comp_batch_size,
                                             n_iter_per_query=500, max_q_len=4, max_d_len=500):
    if not os.path.isfile('qn_rd_nrd_pairs_pwe_training_all'):
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
                qn_rd_nrd_pairs.append((qn, qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_pwe_training_all')
    else:
        qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_pwe_training_all')

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []
    q_names = []
    cnt = 0
    for qn, encoded_q, encoded_rd, encoded_nrd in tqdm(qn_rd_nrd_pairs):
        if os.path.isfile(os.path.join(output_folder, 'batch_data_item_#%d_qn=%s' % (cnt, qn))):
            continue
        batch_pd_lengths.append(len(encoded_rd))
        batch_nd_lengths.append(len(encoded_nrd))
        batch_q_lengths.append(len(encoded_q))
        queries_batch.append(encoded_q)
        rel_docs_batch.append(encoded_rd)
        nrel_docs_batch.append(encoded_nrd)
        q_names.append(qn)

        if len(queries_batch) == comp_batch_size:
            sim_matrices_p = parallel_compute_sim_matrices(queries_batch, rel_docs_batch, max_q_len, max_d_len, w)
            sim_matrices_n = parallel_compute_sim_matrices(queries_batch, nrel_docs_batch, max_q_len, max_d_len, w)

            for i in range(len(queries_batch)):
                data_to_save = [(q_names[i], 1, sim_matrices_p[i], batch_q_lengths[i], batch_pd_lengths[i]),
                                (q_names[i], 0, sim_matrices_n[i], batch_q_lengths[i], batch_nd_lengths[i])]
                util.save_model(data_to_save,
                                os.path.join(output_folder, 'batch_data_item_#%d_qn=%s' % (cnt, q_names[i])))
                cnt += 1

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []
            q_names = []

#
# def alt_compute_training_batches_pwe(w, gt_file, dbn, qbn, batch_size, fold_idx, n_iter_per_query=500,
#                                      max_q_len=4, max_d_len=500):
#     if not os.path.isfile('qn_rd_nrd_pairs_pwe' + str(fold_idx)):
#         rd_b_qry = {}
#         nrd_by_qry = {}
#
#         for line in open(gt_file):
#             data = line.split()
#             qname = data[0].strip()
#             dname = data[2].strip()
#             if dname not in dbn.keys():
#                 continue
#             rj = int(data[3].strip())
#             if qname not in rd_b_qry.keys():
#                 rd_b_qry[qname] = []
#                 nrd_by_qry[qname] = []
#
#             if rj > 0:
#                 rd_b_qry[qname].append(dname)
#             else:
#                 nrd_by_qry[qname].append(dname)
#         test_q_names = list(qbn.keys())
#         np.random.shuffle(test_q_names)
#
#         qn_rd_nrd_pairs = []
#         for qn in test_q_names:
#             if qn not in rd_b_qry.keys():
#                 continue
#             tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
#             tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
#             for i in range(n_iter_per_query):
#                 qn_rd_nrd_pairs.append((qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))
#
#         np.random.shuffle(qn_rd_nrd_pairs)
#         util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_pwe' + str(fold_idx))
#     else:
#         qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_pwe' + str(fold_idx))
#
#     queries_batch = []
#     rel_docs_batch = []
#     nrel_docs_batch = []
#     batch_pd_lengths = []
#     batch_nd_lengths = []
#     batch_q_lengths = []
#
#     for encoded_q, encoded_rd, encoded_nrd in qn_rd_nrd_pairs:
#         batch_pd_lengths.append(len(encoded_rd))
#         batch_nd_lengths.append(len(encoded_nrd))
#         batch_q_lengths.append(len(encoded_q))
#         queries_batch.append(encoded_q)
#         rel_docs_batch.append(encoded_rd)
#         nrel_docs_batch.append(encoded_nrd)
#         if len(queries_batch) == batch_size:
#             # print('computing similarity matrices')
#             sim_matrices_p = compute_sim_matrices(queries_batch, rel_docs_batch, max_q_len, max_d_len, w)
#             sim_matrices_n = compute_sim_matrices(queries_batch, nrel_docs_batch, max_q_len, max_d_len, w)
#
#             y = []
#             data1_len = []
#             data2_len = []
#             sim_m = []
#             for i in range(len(queries_batch)):
#                 y.append(1)
#                 y.append(0)
#                 sim_m.append(sim_matrices_p[i])
#                 sim_m.append(sim_matrices_n[i])
#                 data1_len.append(batch_q_lengths[i])
#                 data2_len.append(batch_pd_lengths[i])
#                 data2_len.append(batch_nd_lengths[i])
#             yield max_q_len, max_d_len, y, data1_len, data2_len, sim_m
#
#             queries_batch = []
#             rel_docs_batch = []
#             nrel_docs_batch = []
#             batch_pd_lengths = []
#             batch_nd_lengths = []
#             batch_q_lengths = []
#

def encode_collection(text_by_name_p, word_dict_path, encoded_out_folder):
    # word_dict_path = '/media/alberto/DATA/BaiduNetdiskDownload/data/word_dict.txt'
    text_by_name = {}
    print('reading files in folder')
    for filename in tqdm(os.listdir(text_by_name_p)):
        fp = os.path.join(text_by_name_p, filename)
        if os.path.isfile(fp):
            text_by_name[filename.split(r'.')[0]] = ' '.join(open(fp, 'r').readlines())
    print('reading word2vec model')
    encoded_docs_by_name = {}
    wi = {}
    for line in tqdm(open(word_dict_path)):
        data = line.split()
        word = data[0].strip()
        wid = int(data[1].strip())
        if word not in wi.keys():
            wi[word] = wid
    sw = load_indri_stopwords()
    print('encoding data')
    for dn, dc in tqdm(text_by_name.items()):
        td = util.tokenize(dc, stemming=False, stoplist=sw)
        encoded_doc = [wi[w] for w in td if w in wi.keys()]
        util.save_model(encoded_doc, os.path.join(encoded_out_folder, dn))
        encoded_docs_by_name[dn] = encoded_doc
    return encoded_docs_by_name


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


def get_rel_docs_by_qry(gt_path):
    rel_docs_for_training_queries = {}
    for line in open(gt_path, 'r'):
        data = line.split()
        qname = data[0].strip()
        dname = data[2].strip()
        rel_j = int(data[3])
        if rel_j > 0:
            if qname in rel_docs_for_training_queries.keys():
                rel_docs_for_training_queries[qname].append(dname)
            else:
                rel_docs_for_training_queries[qname] = [dname]
    return rel_docs_for_training_queries


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


def compute_train_test_q_names(q_names):
    np.random.seed(0)
    if not os.path.isfile('test_q_names'):
        training_q_names = np.random.choice(q_names, 200, replace=False)
        test_q_names = [qn for qn in q_names if qn not in training_q_names]
        util.save_model(test_q_names, 'test_q_names')
        util.save_model(training_q_names, 'train_q_names')
    else:
        training_q_names = util.load_model('train_q_names')
        test_q_names = util.load_model('test_q_names')
    return training_q_names, test_q_names


def compute_kfolds_train_test(n_folds, q_names):
    if not os.path.isfile('folds'):
        folds = []
        q_names = np.array(q_names)
        kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(q_names):
            q_train, q_test = q_names[train_index], q_names[test_index]
            folds.append((q_train, q_test))
        util.save_model(folds, 'folds')
    else:
        folds = util.load_model('folds')
    return folds


def encode_collection_pwe(text_by_name_p, word_dict_path, encoded_out_folder):
    # word_dict_path = '/media/alberto/DATA/BaiduNetdiskDownload/data/word_dict.txt'
    text_by_name = {}
    print('reading files in folder')
    for filename in tqdm(os.listdir(text_by_name_p)):
        fp = os.path.join(text_by_name_p, filename)
        if os.path.isfile(fp):
            text_by_name[filename.split(r'.')[0]] = ' '.join(open(fp, 'r').readlines())
    print('reading word2vec model')
    encoded_docs_by_name = {}
    wi = util.load_json(word_dict_path)
    sw = load_indri_stopwords()
    print('encoding data')
    for dn, dc in tqdm(text_by_name.items()):
        td = util.tokenize(dc, stemming=True, stoplist=sw)
        encoded_doc = [wi[w] for w in td if w in wi.keys()]
        util.save_model(encoded_doc, os.path.join(encoded_out_folder, dn))
        encoded_docs_by_name[dn] = encoded_doc
    return encoded_docs_by_name


def compute_sim_m_gc(queries, docs, mql, mdl, we):
    sim_matrices = []
    for i in range(len(queries)):
        q = queries[i]
        d = docs[i]
        sim_m = np.zeros((mql, mdl))
        a = np.array([we[w] for w in q])
        b = np.array([we[w] for w in d])
        for r in range(len(q)):
            for c in range(len(d)):
                sim_m[r, c] = np.math.exp(-np.square(np.linalg.norm(a[r] - b[c])))
        sim_matrices.append(sim_m)
    return sim_matrices


def compute_training_batches_w2v_gc(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, n_iter_per_query, max_q_len,
                                    max_d_len, we):
    if not os.path.isfile('qn_rd_nrd_pairs_' + str(fold_idx)):
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
        util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_' + str(fold_idx))
    else:
        qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_' + str(fold_idx))

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []

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
                sim_m_p = compute_sim_m_gc(queries_batch, rel_docs_batch, max_q_len, max_d_len, we)
                sim_m_n = compute_sim_m_gc(queries_batch, nrel_docs_batch, max_q_len, max_d_len, we)
                sim_m = []
                for k in range(len(sim_m_p)):
                    sim_m.append(sim_m_p[k])
                    sim_m.append(sim_m_n[k])
                y.append(1)
                y.append(0)
                data2.append(pad(rel_docs_batch[i], padding_value, max_d_len))
                data2.append(pad(nrel_docs_batch[i], padding_value, max_d_len))
                data1_len.append(batch_q_lengths[i])
                data2_len.append(batch_pd_lengths[i])
                data2_len.append(batch_nd_lengths[i])
            yield max_q_len, max_d_len, data1, data2, y, data1_len, data2_len, sim_m

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []


def load_training_batches_w2v_gc(data_folder, training_query_names, batch_size):
    data1_len_batch = []
    data2_len_batch = []
    y_batch = []
    cross_batch = []
    batches = []
    for filename in tqdm(os.listdir(data_folder)):
        fp = os.path.join(data_folder, filename)
        for qn in training_query_names:
            if 'qn=' + qn == filename.split('_')[0]:
                if os.path.isfile(fp):
                    data = util.load_model(fp)
                    for p in data:
                        y_batch.append(p[0])
                        data1_len_batch.append(p[1])
                        data2_len_batch.append(p[2])
                        cross_batch.append(p[3])
                    if len(cross_batch) == 2 * batch_size:
                        batches.append((data1_len_batch, data2_len_batch, y_batch, cross_batch))
                        data1_len_batch = []
                        data2_len_batch = []
                        y_batch = []
                        cross_batch = []
    np.random.seed(0)
    np.random.shuffle(batches)
    for b in batches:
        yield b


def precompute_training_batches_w2v_gk(gt_file, dbn, qbn, batch_size, n_iter_per_query,
                                       max_q_len, max_d_len, we, output_folder):
    if not os.path.isfile('qn_rd_nrd_pairs_w2v_gk'):
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
                qn_rd_nrd_pairs.append((qn, qbn[qn], dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs, 'qn_rd_nrd_pairs_w2v_gk')
    else:
        qn_rd_nrd_pairs = util.load_model('qn_rd_nrd_pairs_w2v_gk')

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []
    batch_qn = []
    cnt = 0
    for qn, encoded_q, encoded_rd, encoded_nrd in tqdm(qn_rd_nrd_pairs):
        batch_pd_lengths.append(len(encoded_rd))
        batch_nd_lengths.append(len(encoded_nrd))
        batch_q_lengths.append(len(encoded_q))
        queries_batch.append(encoded_q)
        batch_qn.append(qn)
        rel_docs_batch.append(encoded_rd)
        nrel_docs_batch.append(encoded_nrd)
        if len(queries_batch) == batch_size:
            sim_m_p = compute_sim_m_gc(queries_batch, rel_docs_batch, max_q_len, max_d_len, we)
            sim_m_n = compute_sim_m_gc(queries_batch, nrel_docs_batch, max_q_len, max_d_len, we)
            for i in range(len(queries_batch)):
                data_to_save = [(1, batch_q_lengths[i], batch_pd_lengths[i], sim_m_p[i]),
                                (0, batch_q_lengths[i], batch_nd_lengths[i], sim_m_n[i])]
                util.save_model(data_to_save, os.path.join(output_folder,
                                                           'qn=%s_w2v_gk_cnt%d' % (batch_qn[i], cnt)))
                cnt += 1

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []
            batch_qn = []


def pre_compute_test_fd_w2v_gc(qbn, dbn, w, run_to_rerank, max_q_len, max_d_len, fold, output_folder):
    test_fd_fp = os.path.join(output_folder, 'test_fd_w2v_gc' + str(fold))
    if not os.path.isfile(test_fd_fp):
        d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
        for q_name in tqdm(qbn.keys()):
            if os.path.isfile(test_fd_fp + '_' + q_name):
                continue
            if q_name not in d_t_rerank_by_query.keys():
                continue
            # print(q_name)
            d_names = d_t_rerank_by_query[q_name]
            d_names = [dn for dn in d_names if dn in dbn.keys()]
            docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
            d_lengths = [len(d) for d in docs]
            q = qbn[q_name]
            sim_m = compute_sim_m_gc([q] * len(d_names), docs, max_q_len, max_d_len, w)
            util.save_model(([len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, sim_m), test_fd_fp + '_qn=' + q_name)


def pre_compute_test_fd_batches_w2v_gk(qbn, dbn, w, max_q_len, max_d_len, fold, output_folder):
    test_fd_fp = os.path.join(output_folder, 'test_fd_w2v_gc' + str(fold))
    d_t_rerank_by_query = util.load_model('sorted_d_names_by_query.model')
    if not os.path.isfile(test_fd_fp):
        # d_t_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, qbn.keys())
        for q_name in tqdm(qbn.keys()):
            if os.path.isfile(test_fd_fp + '_qn=' + q_name):
                continue
            if q_name not in d_t_rerank_by_query.keys():
                continue
            # print(q_name)
            d_names = d_t_rerank_by_query[q_name]
            d_names = [dn for dn in d_names if dn in dbn.keys()]
            docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
            d_lengths = [len(d) for d in docs]
            q = qbn[q_name]
            sim_m = compute_sim_m_gc([q] * len(d_names), docs, max_q_len, max_d_len, w)
            util.save_model(([len(qbn[q_name])] * len(docs), d_lengths, d_names, q_name, sim_m), test_fd_fp + '_qn=' + q_name)


def load_test_batches(data_folder, qnames):
    for qn in qnames:
        for filename in os.listdir(data_folder):
            fp = os.path.join(data_folder, filename)
            if 'qn=' + qn == filename.split('_')[-1]:
                if os.path.isfile(fp):
                    len_q, len_d, d_names, q_name, sim_m = util.load_model(fp)
                    yield len_q, len_d, d_names, q_name, sim_m
