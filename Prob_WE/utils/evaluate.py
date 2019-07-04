"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import platform
import subprocess
import time

from tqdm import tqdm
import numpy as np
import os
from wasserstein.operators import BuresProductNormalized, centroid
import torch
from utils import util
from utils import input_output

# choose correct variable values according to what pc I am using
if platform.node() == 'acquario':
    # OS_PYTHON = r'/home/alberto/anaconda3/envs/tensorflow/bin/python3'
    OS_PYTHON = r'/home/marco/anaconda3/envs/prob_ir/bin/python'
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
elif platform.node() == 'DESKTOP-LKU0VTG':
    OS_PYTHON = r'C:\Users\apirp\anaconda\envs\tf\python.exe'
    TREC_EVAL_PATH = r'trec_eval-master\trec_eval.exe'
elif platform.node() == 'hopper':
    OS_PYTHON = '/home/ims/anaconda3/envs/tensorflow_env/bin/python3'
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
elif platform.node() == 'alberto-Alienware-15-R4':
    OS_PYTHON = r'/home/alberto/anaconda3/envs/tf/bin/python3'
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
else:
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
    OS_PYTHON = r'/home/marco/anaconda3/envs/prob_ir/bin/python'


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


def evaluate_ranking(encoded_idf_scores, queries, documents, query_names, doc_names, gt_file, run_to_rerank):
    """
    Documents and queries are just the tokenized versions of queries and documents (not encoded)1
    """
    docs_to_rerank_by_query = compute_docs_to_rerank_by_query(run_to_rerank, query_names)
    rel_docs_by_qry = {}
    sim_scores_by_qry = {}
    for i in tqdm(range(len(queries))):
        print('query: %d/%d' % (i, len(query_names)))
        q_name = query_names[i]
        q = queries[i]
        if len(q) == 0:
            print('skipped zero length query!')
            continue
        d_names_to_rerank = docs_to_rerank_by_query[q_name][:5]
        d_encoded = [documents[doc_names.index(d_name)] for d_name in d_names_to_rerank]
        sims = []
        for d in d_encoded:
            if len(d) == 0:
                sims.append(0)
            else:
                sims.append(compute_q_d_sim(q, d, encoded_idf_scores))
        sims = np.array(sims)
        rel_docs_by_qry[q_name] = (np.array(d_names_to_rerank)[np.argsort(-sims)])
        sim_scores_by_qry[q_name] = sims[np.argsort(-sims)]
    map_v = create_evaluate_ranking(rel_docs_by_qry, sim_scores_by_qry, gt_file)
    print('map=%2.4f' % map_v)
    return map_v


def run_trec_eval(qrels_file, run_to_eval):
    # trec_eval_path = TREC_EVAL_PATH,
    print('using the qrels file: %s' % qrels_file)
    if platform.node() == 'acquario' or platform.node() == 'hopper' or platform.node() == 'alberto-Alienware-15-R4':
        command = os.path.join(os.getcwd(), TREC_EVAL_PATH) + ' ' \
                  + os.path.join(os.getcwd(), qrels_file) + ' ' \
                  + os.path.join(os.getcwd(), run_to_eval) + ' | grep "^map" '
    else:
        command = os.path.join(os.getcwd(), TREC_EVAL_PATH) + ' -m map ' \
                  + os.path.join(os.getcwd(), qrels_file) + ' ' \
                  + os.path.join(os.getcwd(), run_to_eval)
    print(command)
    (map_line, err) = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).communicate()
    map_line = map_line.decode("utf-8")
    map_value = map_line.split('\t')[2]
    return float(map_value)


def compute_q_d_sim(q, d, encoded_idf_scores):
    document_terms_weights = []
    distinct_doc_terms = list(set(d))
    for w in distinct_doc_terms:
        document_terms_weights.append(encoded_idf_scores[w] * d.count(w))

    metric = BuresProductNormalized()
    dict_pt = torch.load('/home/alberto/PycharmProjects/probabilisticir/embeddings_dim_75_margin_2.0')
    w = dict_pt["embeddings"]
    dim = 20  # dimensione dell'embedding

    sim = 0
    for idx_q in q:
        qw_m, qw_c = (w[idx_q, 0:dim].view(1, dim), w[idx_q, dim:].view((1, dim, dim)))
        for i in range(len(distinct_doc_terms)):
            idx_d = distinct_doc_terms[i]
            weight = document_terms_weights[i]
            # for idx_d in distinct_doc_terms:
            dw_m, dw_c = (w[idx_d, 0:dim].view(1, dim), w[idx_d, dim:].view((1, dim, dim)))
            sim += metric(qw_m, dw_m, qw_c, dw_c) * weight
    sim /= len(q) * len(d)
    return float(sim)


# ---------------------------------------------------


def create_evaluate_ranking(rel_docs_by_qry, sim_scores_by_qry, gt_file):
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ranking_WEIR_.txt')
    out = open(output_file, 'w')
    for q, rd in rel_docs_by_qry.items():
        for i in range(len(rd)):
            dname = rd[i]
            sim_score = sim_scores_by_qry[q][i]
            line = str(q) + '\t0\t' + str(dname) + '\t' + str(i) + '\t' + str(sim_score) + '\tWEIR\n'
            out.write(line)
    out.close()
    map_v = run_trec_eval(run_to_eval=output_file, qrels_file=gt_file)
    # os.remove(output_file)
    return map_v


def filter_queries(run_to_evaluate, gt_file, run_to_rerank):
    query_names_to_keep = []
    for line in open(run_to_evaluate):
        qname = line.split()[0]
        if qname not in query_names_to_keep:
            query_names_to_keep.append(qname)

    out = open('gt_subset', 'w')
    for line in open(gt_file):
        qname = line.split()[0]
        if qname in query_names_to_keep:
            out.write(line)
    out.close()
    out = open('new_run_to_rerank', 'w')
    for line in open(run_to_rerank):
        qname = line.split()[0]
        if qname in query_names_to_keep:
            out.write(line)
    out.close()


def run():
    queries_folder = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/topics'
    documents_folder = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/corpus'
    stop_word_path = '/home/alberto/PycharmProjects/probabilisticir/indri_stoplist_eng.txt'
    gt_file = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/qrels.robust2004.txt'
    run_to_rerank = '/home/alberto/PycharmProjects/probabilisticir/robust.terrier.krovetz.qld.2k.run'

    queries, query_names = input_output.load_texts(queries_folder, stop_word_path)
    # documents, doc_names = input_output.load_texts(documents_folder, stop_word_path)

    # idf_scores, word_index = input_output.compute_idf(documents, 10, 0.5)
    # util.save_json(word_index, 'word_index_json')
    idf_scores = util.load_model('idf_scores')
    word_index = util.load_json('word_index_json')
    encoded_idf_scores = {word_index[k]: v for k, v in idf_scores.items() if k in word_index.keys()}

    # encoded_docs = [[word_index[w] for w in d if w in word_index.keys()] for d in documents]
    # encoded_queries = [[word_index[w] for w in q if w in word_index.keys()] for q in queries]

    # util.save_model(encoded_docs, 'encoded_docs')
    # util.save_model(encoded_queries, 'encoded_queries')
    # util.save_model(query_names, 'q_names')
    # util.save_model(doc_names, 'd_names')

    query_names = util.load_model('q_names')
    doc_names = util.load_model('d_names')

    # query_names = util.load_model('q_names')
    # doc_names = util.load_model('d_names')

    query_names = [n.split(r'.txt')[0] for n in query_names][:50]
    doc_names = [n.split(r'.txt')[0] for n in doc_names]
    encoded_queries = util.load_model('encoded_queries')[:50]
    encoded_docs = util.load_model('encoded_docs')
    evaluate_ranking(encoded_idf_scores, encoded_queries, encoded_docs, query_names, doc_names, gt_file, run_to_rerank)


if __name__ == '__main__':
    run_to_evaluate = '/home/alberto/PycharmProjects/probabilisticir/ranking_WEIR_.txt'
    gt_file = '/media/alberto/DATA/ExperimentalCollections/Robust04/original/qrels.robust2004.txt'
    run_to_rerank = '/home/alberto/Dropbox/IPM2019-MPS/EXPERIMENTS/RUNS/robust.terrier.krovetz.qld.2k.run'
    # filter_queries(run_to_evaluate, gt_file, run_to_rerank)
    run()
