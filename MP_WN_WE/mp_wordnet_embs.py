"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import multiprocessing
import os
from fastText import load_model
from gensim.models import KeyedVectors
from tqdm import tqdm
import data_utils as du
import util
import numpy as np
from model import Model
import tensorflow as tf
import evaluation as ev
import gensim
import lexical_models as lm
from nltk.corpus import wordnet
from itertools import product


#####################################################################################################################
# INPUT DATA PARSING
#####################################################################################################################

def read_collection(coll_main_folder, output_model_path, stemming, stoplist=None):
    if not os.path.isfile(output_model_path):
        if stoplist is None:
            stoplist = util.load_indri_stopwords()
        text_by_name = {}
        print('reading files in folder')
        pool = multiprocessing.Pool(8)
        fnames_list = os.listdir(coll_main_folder)
        doc_paths_list = [os.path.join(coll_main_folder, filename) for filename in fnames_list]
        print('processing collection')
        tokenized_docs = pool.starmap(util.tokenize,
                                      [(' '.join(open(fp, 'r').readlines()), stemming, stoplist) for fp in
                                       doc_paths_list])

        for i in range(len(fnames_list)):
            text_by_name[fnames_list[i].split(r'.')[0]] = tokenized_docs[i]

        print('saving model')
        util.save_model(text_by_name, output_model_path)
    else:
        print('loading model: %s' % output_model_path)
        text_by_name = util.load_model(output_model_path)
    return text_by_name


def compute_input_data(text_by_name, ftext_model_path, encoded_out_folder_docs):
    print('loading fasttext model')
    f = load_model(ftext_model_path)
    encoded_docs_by_name = {}
    wi = {}
    print('encoding collection')
    we_matrix = []
    for dn, dt in tqdm(text_by_name.items()):
        encoded_d = []
        for tok in dt:
            if tok not in wi.keys():
                wv = f.get_word_vector(tok)
                wi[tok] = len(wi)
                we_matrix.append(wv)
            encoded_d.append(wi[tok])
        encoded_docs_by_name[dn] = encoded_d
        util.save_model(encoded_d, os.path.join(encoded_out_folder_docs, dn))
    return encoded_docs_by_name, wi, np.array(we_matrix)


def encode_queries(queries_main_folder, wi, stemming):
    sw = util.load_indri_stopwords()
    encoded_qbn = {}
    for filename in tqdm(os.listdir(queries_main_folder)):
        fp = os.path.join(queries_main_folder, filename)
        if os.path.isfile(fp):
            tokenized_query = util.tokenize(' '.join(open(fp, 'r').readlines()), stemming=stemming, stoplist=sw)
            qn = filename.split(r'.')[0]
            encoded_qbn[qn] = [wi[w] for w in tokenized_query if w in wi.keys()]
    return encoded_qbn


def compute_inverted_index(coll_folder, stemming, output_file_path_ii):
    if not os.path.isfile(output_file_path_ii):
        print('computing inverted index')
        inverted_idx = {}
        sw = util.load_indri_stopwords()
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
    else:
        inverted_idx = util.load_model(output_file_path_ii)
    return inverted_idx


def compute_data():
    ftext_model_path = '../data/fasttext_models/wiki.en.bin'
    output_path_wi_model = '../data/fasttext_models/wi_robust'
    output_path_ii_model = '../data/fasttext_models/ii_robust'
    output_path_idf_model = '../data/fasttext_models/idf_robust'
    output_path_encoded_d_model = '../data/fasttext_models/encoded_dbn'
    output_path_encoded_q_model = '../data/fasttext_models/encoded_qbn'
    output_path_we_matrix_model = '../data/fasttext_models/word_embeddings_matrix_robust'
    coll_path = '/Users/albertopurpura/ExperimentalCollections/Robust04/processed/corpus'
    queries_main_folder = '/Users/albertopurpura/ExperimentalCollections/Robust04/processed/topics'
    output_model_path = 'data/robust/stemmed_coll_model'
    encoded_out_folder_docs = 'data/robust/stemmed_encoded_docs_ft'

    stemming = True

    if not os.path.isfile(output_path_ii_model):
        print('computing inverted index')
        ii = compute_inverted_index(coll_path, stemming, output_path_ii_model)
        util.save_model(ii, output_path_ii_model)
    else:
        print('loading inverted index')
        ii = util.load_model(output_path_ii_model)

    if not os.path.isfile(output_path_encoded_d_model):
        text_dbn = read_collection(coll_path, output_model_path, stemming=stemming,
                                   stoplist=util.load_indri_stopwords())

        encoded_dbn, wi, we_matrix = compute_input_data(text_dbn, ftext_model_path, encoded_out_folder_docs)

        util.save_model(encoded_dbn, output_path_encoded_d_model)
        util.save_model(wi, output_path_wi_model)
        util.save_model(we_matrix, output_path_we_matrix_model)
    else:
        encoded_dbn = util.load_model(output_path_encoded_d_model)
        wi = util.load_model(output_path_wi_model)
        we_matrix = util.load_model(output_path_we_matrix_model)

    if not os.path.isfile(output_path_encoded_q_model):
        encoded_qbn = encode_queries(queries_main_folder, wi, stemming)
        util.save_model(encoded_qbn, output_path_encoded_q_model)
    else:
        encoded_qbn = util.load_model(output_path_encoded_q_model)

    idf_scores = du.compute_idf(coll_path, stemming, output_path_ii_model, output_path_idf_model)

    return encoded_dbn, encoded_qbn, we_matrix, wi, ii, idf_scores


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
# TRAINING
#####################################################################################################################
def pad(to_pad, padding_value, max_len):
    retval = np.ones(max_len) * padding_value
    for i in range(len(to_pad)):
        retval[i] = to_pad[i]
    return retval


def compute_training_pairs_w_queries_variations(fold_idx, coll, n_iter_per_query, gt_file, dbn, qbn, ftt_model, iwi,
                                                wi):
    model_name = 'qn_rd_nrd_pairs_w2v_gk_' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query)
    if not os.path.isfile(model_name):
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

            # add training examples with original query:
            encoded_q = qbn[qn]
            tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
            tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
            for i in range(n_iter_per_query):
                qn_rd_nrd_pairs.append((encoded_q, dbn[tmp_rdocs[i]], dbn[tmp_nrdocs[i]]))
            print('original query: ' + ' '.join([iwi[w] for w in encoded_q]))
            # add extra training examples
            for i in range(len(encoded_q)):
                encoded_q_variation = encoded_q
                curr_q_word = iwi[encoded_q[i]]
                similar_words = get_synonyms(curr_q_word, ftt_model)
                for sw in similar_words:
                    sw = util.stem(sw)
                    if sw in wi.keys() and curr_q_word != sw:
                        print('word = ' + curr_q_word + ', substitute = ' + sw)
                        encoded_q_variation[i] = wi[sw]
                        print('alternative query: ' + ' '.join([iwi[w] for w in encoded_q_variation]))
                        tmp_rdocs = np.random.choice(rd_b_qry[qn], n_iter_per_query, replace=True)
                        tmp_nrdocs = np.random.choice(nrd_by_qry[qn], n_iter_per_query, replace=True)
                        for j in range(n_iter_per_query):
                            qn_rd_nrd_pairs.append((encoded_q_variation, dbn[tmp_rdocs[j]], dbn[tmp_nrdocs[j]]))

        np.random.shuffle(qn_rd_nrd_pairs)
        util.save_model(qn_rd_nrd_pairs, model_name)
    else:
        qn_rd_nrd_pairs = util.load_model(model_name)
    return qn_rd_nrd_pairs


def get_synonyms(source, ft_model):
    # a good replacement for a query term should have a similar semantic meaning and a similar usage context
    # the first aspect is satisfied by wordnet, the second aspect is checked with the fastText model.
    synonyms = []
    for syn in wordnet.synsets(source):
        for lm in syn.lemmas():
            if lm.name() in ft_model.wv.vocab:
                rcsim = ft_model.relative_cosine_similarity(source, lm.name(), topn=10)
                if rcsim > 0.10:
                    synonyms.append(lm.name())
    # print(set(synonyms))
    return set(synonyms)


def compute_training_pairs(fold_idx, coll, n_iter_per_query, gt_file, dbn, qbn):
    if not os.path.isfile('qn_rd_nrd_pairs_wn2v' + str(fold_idx) + '_' + str(coll) + '_' + str(n_iter_per_query)):
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
    return qn_rd_nrd_pairs


def compute_wn_sim(qw, dw, i, j, k):
    allsyns1 = wordnet.synsets(qw)
    allsyns2 = wordnet.synsets(dw)
    if len(allsyns2) == 0 or len(allsyns1) == 0:
        return 0, i, j, k
    best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))[0]
    return best, i, j, k


"""
def compute_addit_info_sm_parallel(encoded_q, encoded_d, mql, mdl, iwi):

    pool = multiprocessing.Pool(100)
    args_to_pass = []
    for i in range(len(encoded_q)):
        qw = iwi[encoded_q[i]]
        for j in range(len(encoded_d)):
            dw = iwi[encoded_d[j]]
            args_to_pass.append((qw, dw, i, j))
    print('computing data in parallel')
    results = pool.starmap(compute_wn_wup_sim, args_to_pass)
    pool.close()
    coeff = np.ones((mql, mdl))
    for item in results:
        val = item[0]
        i = item[1]
        j = item[2]
        coeff[i, j] = val
    return coeff


"""


def compute_addit_info_sm_parallel(encoded_queries, encoded_docs, mql, mdl, iwi, wn2v_model):
    # print('computing input data for parallel computation')
    coeffs = []
    for k in range(len(encoded_queries)):
        coeffs.append(np.ones((mql, mdl)))  # to use for multipl
        # coeffs.append(np.zeros((mql, mdl)))
        encoded_q = encoded_queries[k]
        encoded_d = encoded_docs[k]
        for i in range(len(encoded_q)):
            qw = iwi[encoded_q[i]]
            for j in range(len(encoded_d)):
                dw = iwi[encoded_d[j]]
                if qw in wn2v_model.wv.vocab and dw in wn2v_model.wv.vocab:
                    qwe = wn2v_model[qw] / np.linalg.norm(wn2v_model[qw])
                    dwe = wn2v_model[dw] / np.linalg.norm(wn2v_model[dw])
                    cos_sim = np.dot(qwe, dwe)
                    coeffs[k][i, j] = cos_sim
    return coeffs


def compute_addit_info_sm(encoded_q, encoded_d, mql, mdl, iwi):
    # http://www.nltk.org/howto/wordnet.html
    # https://stackoverflow.com/questions/30829382/check-the-similarity-between-two-words-with-nltk-with-python

    coeff = np.ones((mql, mdl))
    for i in range(len(encoded_q)):
        qw = iwi[encoded_q[i]]
        allsyns1 = wordnet.synsets(qw)  # set(ss for word in list1 for ss in wordnet.synsets(word))
        for j in range(len(encoded_d)):
            dw = iwi[encoded_d[j]]
            allsyns2 = wordnet.synsets(dw)  # set(ss for word in list2 for ss in wordnet.synsets(word))
            if len(allsyns2) == 0 or len(allsyns1) == 0:
                coeff[i, j] = 0
            else:
                best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
                coeff[i, j] = best[0]
    return coeff


def compute_training_batches(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, ftt_model, iwi, wi,
                             n_iter_per_query=200, max_q_len=4, max_d_len=500, coll='robust'):
    qn_rd_nrd_pairs = compute_training_pairs(fold_idx, coll, n_iter_per_query, gt_file, dbn, qbn)

    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []

    print('# iterations per epoch: ' + str(int(len(qn_rd_nrd_pairs) / batch_size)))
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


def compute_idf_scaling_coeffs(encoded_q, iwi, idf_scores, mql, mdl):
    coeffs = np.ones((mql, mdl))
    for i in range(len(encoded_q)):
        if int(encoded_q[i]) in iwi.keys():
            w = iwi[int(encoded_q[i])]
            coeffs[i] = np.ones(mdl) * idf_scores[w]
    return coeffs


def compute_training_batches_w_coeffs(gt_file, dbn, qbn, batch_size, padding_value, fold_idx, ftt_model, iwi, wi,
                                      wn2v_model, n_iter_per_query=200, max_q_len=4, max_d_len=500, coll='robust'):
    qn_rd_nrd_pairs = compute_training_pairs(fold_idx, coll, n_iter_per_query, gt_file, dbn, qbn)
    queries_batch = []
    rel_docs_batch = []
    nrel_docs_batch = []
    batch_pd_lengths = []
    batch_nd_lengths = []
    batch_q_lengths = []
    coeffs = []
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

            coeffs_p = compute_addit_info_sm_parallel(queries_batch, rel_docs_batch, max_q_len, max_d_len, iwi,
                                                      wn2v_model)
            coeffs_n = compute_addit_info_sm_parallel(queries_batch, nrel_docs_batch, max_q_len, max_d_len, iwi,
                                                      wn2v_model)

            coeffs = []
            for i in range(len(queries_batch)):
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                data1.append(pad(queries_batch[i], padding_value, max_q_len))
                y.append(1)
                y.append(0)
                coeffs.append(coeffs_p[i])
                coeffs.append(coeffs_n[i])
                data2.append(pad(rel_docs_batch[i], padding_value, max_d_len))
                data2.append(pad(nrel_docs_batch[i], padding_value, max_d_len))
                data1_len.append(batch_q_lengths[i])
                data1_len.append(batch_q_lengths[i])
                data2_len.append(batch_pd_lengths[i])
                data2_len.append(batch_nd_lengths[i])

            yield max_q_len, max_d_len, data1, data2, y, data1_len, data2_len, coeffs

            queries_batch = []
            rel_docs_batch = []
            nrel_docs_batch = []
            batch_pd_lengths = []
            batch_nd_lengths = []
            batch_q_lengths = []
            coeffs = []


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
# RANKING
#####################################################################################################################

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


def compute_docs_to_rerank_by_query(queries_names, qbn, dbn, iwi, ii, fasttext_vec_model):
    docs_to_rerank_by_qry = {}
    for qn in tqdm(queries_names):
        q = qbn[qn]
        for qw in q:
            query_word = iwi[qw]
            if query_word not in fasttext_vec_model.wv.vocab:
                continue
            # here I try to find the most similar terms to the stemmed word
            similar_words = [w[0] for w in fasttext_vec_model.most_similar(positive=[query_word], topn=10)]
            for w in similar_words:
                # stem the most similar words found in the model
                w = util.stem(w)
                if w in ii.keys():
                    if qn not in docs_to_rerank_by_qry.keys():
                        docs_to_rerank_by_qry[qn] = []
                    docs_to_rerank_by_qry[qn].extend([pl[0] for pl in ii[w]])

    return docs_to_rerank_by_qry


def compute_docs_to_rerank_by_query_lexical(run_to_rerank, test_query_names):
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


def compute_docs_to_rerank_by_query_lexical_bm25(query, dbn, ii, iwi, n_docs_to_keep=2000):
    docs_with_scores = lm.rank_docs_w_bm25(query, dbn, ii, iwi)
    dnames = docs_with_scores.keys()
    scores = np.array([docs_with_scores[dn] for dn in dnames])
    return (np.array(list(dnames))[np.argsort(-scores)])[0:n_docs_to_keep]


def compute_test_fd_w2v(qbn, dbn, fasttext_vec_model, iwi, ii_dict, max_q_len, max_d_len, padding_value, run_to_rerank):
    d_names_bqn = compute_docs_to_rerank_by_query_lexical(run_to_rerank, qbn.keys())
    for q_name in qbn.keys():
        # d_names = compute_docs_to_rerank_by_query_lexical_bm25(qbn[q_name], dbn, ii_dict, iwi)
        d_names = d_names_bqn[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        padded_d = [pad(d, padding_value, max_d_len) for d in docs]
        q = pad(qbn[q_name], padding_value, max_q_len)

        doc_batch = []
        d_lengths_batch = []
        d_names_batch = []
        for i in range(len(padded_d)):
            doc_batch.append(padded_d[i])
            d_lengths_batch.append(d_lengths[i])
            d_names_batch.append(d_names[i])
            if len(doc_batch) == 5000:
                yield [q] * len(doc_batch), doc_batch, [len(qbn[q_name])] * len(doc_batch), d_lengths_batch, \
                      d_names_batch, q_name
                doc_batch = []
                d_lengths_batch = []
                d_names_batch = []
        if len(doc_batch) > 0:
            yield [q] * len(doc_batch), doc_batch, [len(qbn[q_name])] * len(doc_batch), d_lengths_batch, \
                  d_names_batch, q_name

        # yield [q] * len(padded_d), padded_d, [len(qbn[q_name])] * len(padded_d), d_lengths, d_names, q_name


def compute_test_fd_w_coeffs(qbn, dbn, fasttext_vec_model, iwi, ii_dict, max_q_len, max_d_len, padding_value,
                             run_to_rerank, wn2v_model, idf_scores):
    d_names_bqn = compute_docs_to_rerank_by_query_lexical(run_to_rerank, qbn.keys())
    for q_name in qbn.keys():
        # d_names = compute_docs_to_rerank_by_query_lexical_bm25(qbn[q_name], dbn, ii_dict, iwi)
        d_names = d_names_bqn[q_name]
        docs = [dbn[dn] for dn in d_names if dn in dbn.keys()]
        d_lengths = [len(d) for d in docs]
        padded_d = [pad(d, padding_value, max_d_len) for d in docs]
        q = pad(qbn[q_name], padding_value, max_q_len)
        # print('computing coefficients')
        coeffs = compute_addit_info_sm_parallel([q] * len(docs), docs, max_q_len, max_d_len, iwi, wn2v_model)
        # idf_coeffs = [compute_idf_scaling_coeffs(q, iwi, idf_scores, max_q_len, max_d_len)] * len(padded_d)
        yield [q] * len(padded_d), padded_d, [len(qbn[q_name])] * len(padded_d), d_lengths, d_names, q_name, coeffs


def true_rerank(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl, padding_value, fold_index, test_qrels_file,
                epoch, run_name, fasttext_vec_model, iwi, ii_dict, wn2v_model, idf_scores, output_f='../results'):
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}

    all_preds = {}
    dnames_to_rerank_by_qry = {}
    for i, (data1_test, data2_test, data1_len_test, data2_len_test, d_names, q_name, coeffs) in tqdm(enumerate(
            compute_test_fd_w_coeffs(test_q_b_name, dbn, fasttext_vec_model, iwi, ii_dict, mql, mdl, padding_value,
                                     run_to_rerank, wn2v_model, idf_scores))):
        if len(test_q_b_name[q_name]) == 0:
            continue
        feed_dict = {model.X1: data1_test, model.X1_len: data1_len_test, model.X2: data2_test,
                     model.X2_len: data2_len_test, model.training: False, model.coeffs: coeffs}
        # print('computing prediction %d of %d' % (i, len(test_q_b_name)))
        pred = model.test_step(sess, feed_dict)
        pred = np.array([p[0] for p in pred])
        if q_name not in dnames_to_rerank_by_qry.keys():
            dnames_to_rerank_by_qry[q_name] = []
            all_preds[q_name] = []
        dnames_to_rerank_by_qry[q_name].extend(d_names)
        all_preds[q_name].extend(pred / np.max(pred))

    for q_name in test_q_b_name.keys():
        if q_name not in all_preds.keys():
            continue
        pred = np.array(all_preds[q_name])
        rel_docs_by_qry[q_name] = (np.array(dnames_to_rerank_by_qry[q_name])[np.argsort(-pred)])[0:1000]
        sim_scores_by_qry[q_name] = pred[np.argsort(-pred)][0:1000]

    map_v = util.create_evaluate_ranking(str(fold_index) + '_' + str(epoch) + '_pure_neural', rel_docs_by_qry,
                                         sim_scores_by_qry, test_qrels_file, run_name, output_folder=output_f)
    print('map of pure neural model=%2.5f' % map_v)
    return map_v


#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
# MAIN:
#####################################################################################################################

def run():
    # map intorno al 17-18%
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gt_path = '../data/robust/qrels.robust2004.txt'
    run_to_rerank = '../data/robust/robust.terrier.krovetz.qld.2k.run'
    run_name = 'wordnet_embs_MP_evaluated_at_each_epoch'
    batch_size = 64
    n_epochs = 20
    n_folds = 5
    max_patience = 10

    encoded_dbn, encoded_qbn, we_matrix, wi, ii, idf_scores = compute_data()
    print('loading fasttext model with gensim')
    fasttext_vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        '../data/fasttext_models/wiki-news-300d-1M.vec')  # non viene usato piu perche non uso wordnet direttamente ma uso degli embeddings
    wn2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../data/wn_embeddings/wn2vec.txt')
    print('inverting word index')
    iwi = util.invert_wi(wi)
    print('turning inverted index to a dict of dicts')
    ii_dict = {w: {p[0]: p[1] for p in pl} for w, pl in ii.items()}

    print('reducing document lengths')
    dbn = {dn: dc[0: min(len(dc), 500)] for dn, dc in encoded_dbn.items()}
    mql = max([len(q) for q in encoded_qbn.values()])
    mdl = max([len(d) for d in dbn.values()])

    print('max query length; %d' % mql)
    print('max doc length; %d' % mdl)
    padding_value = we_matrix.shape[0] - 1

    config = {'embed_size': 50, 'data1_psize': 3, 'data2_psize': 10, 'embedding': we_matrix, 'feat_size': 0,
              'fill_word': padding_value, 'data1_maxlen': mql, 'data2_maxlen': mdl}

    folds = du.compute_kfolds_train_test(n_folds, list(encoded_qbn.keys()), coll='robust')
    print('starting training')
    for fold_index in range(len(folds)):
        print('Fold: %d' % fold_index)
        tf.reset_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        tf.set_random_seed(0)
        with tf.Session(config=sess_config) as sess:
            tf.set_random_seed(0)
            model = Model(config)
            model.init_step(sess)

            training_q_names, test_q_names = folds[fold_index]

            test_q_b_name = {test_q_names[i]: encoded_qbn[test_q_names[i]] for i in range(len(test_q_names))}
            test_qrels_file = 'test_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), test_q_names, test_qrels_file)

            validation_q_names = np.random.choice(training_q_names, int(len(training_q_names) * 0.20), replace=False)
            validation_q_b_name = {validation_q_names[i]: encoded_qbn[validation_q_names[i]] for i in
                                   range(len(validation_q_names))}

            validation_qrels_file = 'validation_qrels_file_' + str(fold_index)
            du.compute_test_gt_file(gt_path, list(dbn.keys()), validation_q_names, validation_qrels_file)

            training_q_names = [n for n in training_q_names if n not in validation_q_names]
            training_q_b_name = {training_q_names[i]: encoded_qbn[training_q_names[i]] for i in
                                 range(len(training_q_names))}

            max_map = 0.0
            best_epoch = -1
            best_iter = -1
            patience = 0
            done = False

            for epoch in range(n_epochs):
                if done:
                    print('early stopping')
                    break
                print('epoch=%d' % epoch)
                losses = []
                for i, (data1_max_len, data2_max_len, data1, data2, y, data1_len, data2_len, coeffs) in \
                        tqdm(enumerate(compute_training_batches_w_coeffs(gt_path, dbn, training_q_b_name, batch_size,
                                                                         padding_value=len(we_matrix) - 1,
                                                                         fold_idx=fold_index, n_iter_per_query=1000,
                                                                         max_q_len=mql, max_d_len=mdl,
                                                                         ftt_model=fasttext_vec_model,
                                                                         wn2v_model=wn2vec_model,
                                                                         iwi=iwi, wi=wi))):
                    feed_dict = {model.X1: data1, model.X1_len: data1_len, model.X2: data2,
                                 model.X2_len: data2_len, model.Y: y, model.training: True, model.coeffs: coeffs}
                    loss = model.train_step(sess, feed_dict)
                    losses.append(loss)

                    #if i % 1000 == 0 and i != 0:
                print('evaluating at epch %d' % epoch)
                map_v = true_rerank(sess, model, validation_q_b_name, dbn, run_to_rerank, mql, mdl,
                                    padding_value, fold_index, validation_qrels_file, epoch, run_name,
                                    fasttext_vec_model, iwi, ii_dict, wn2vec_model, idf_scores,
                                    output_f='../results')
                print('Fold=%d, Epoch=%d, Iter=%d, Validation Map=%2.5f, Loss=%2.4f' % (
                    fold_index, epoch, i, map_v, loss))
                if map_v > max_map:
                    best_epoch = epoch
                    best_iter = i
                    patience = 0
                    map_v_test = true_rerank(sess, model, test_q_b_name, dbn, run_to_rerank, mql, mdl,
                                             padding_value,
                                             fold_index, test_qrels_file, epoch, run_name + '_TEST_set',
                                             wn2vec_model, iwi,
                                             ii_dict, wn2vec_model, idf_scores, output_f='../results')
                else:
                    patience += 1
                    if patience == max_patience:
                        print('Early stopping at epoch=%d, iter=%d (patience=%d)!' % (epoch, i, max_patience))
                        done = True
                        break

            print('Fold %d, (presumed) BEST TEST map=%2.5f' % (fold_index, map_v_test))
            print('BEST valid map in fold=%2.4f, at epoch=%d, iter=%d' % (map_v_test, best_epoch, best_iter))


if __name__ == '__main__':
    run()
