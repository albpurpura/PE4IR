"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import csv
import io
import json
import os
import pickle
import platform
import subprocess

import numpy as np
import krovetz

import string

from tqdm import tqdm
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer

kstemmer = krovetz.PyKrovetzStemmer()
# choose correct variable values according to what pc I am using
TREC_EVAL_PATH = '../../trec_eval.8.1/trec_eval'

with open('../data/indri_stoplist_eng.txt', 'r') as slf:
    sw = slf.readlines()
    sw = [word.strip() for word in sw]


def save_json(model, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(model, outfile)


def load_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def save_model(model, output_path):
    with open(output_path, 'wb') as handle:
        pickle.dump(model, handle)
        handle.close()


def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model


def run_trec_eval(trec_eval_path=TREC_EVAL_PATH,
                  qrels_file='data/cran/processed_corpus/cranfield.qrel',
                  run_to_eval='/Users/albertopurpura/PycharmProjects/ml4ir_git/results/re_ranking_output_lmnn.txt'):
    print('using the qrels file: %s' % qrels_file)
    if True or platform.node() == 'Albertos-MacBook-Pro.local' or platform.node() == 'hopper' or platform.node() == 'alberto-Alienware-15-R4':
        command = os.path.join(os.getcwd(), trec_eval_path) + ' ' \
                  + os.path.join(os.getcwd(), qrels_file) + ' ' \
                  + os.path.join(os.getcwd(), run_to_eval) + ' | grep "^map" '
    else:
        command = os.path.join(os.getcwd(), trec_eval_path) + ' -m map ' \
                  + os.path.join(os.getcwd(), qrels_file) + ' ' \
                  + os.path.join(os.getcwd(), run_to_eval)
    print(command)
    (map_line, err) = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).communicate()
    map_line = map_line.decode("utf-8")
    if len(map_line) > 0:
        map_value = map_line.split('\t')[2]
    else:
        print('Error computing the map value!')
        map_value = -1
    return float(map_value)


def contains_digits(token):
    for c in token:
        if c.isdigit():
            return True
    return False


def tokenize(text, stemming=True, stoplist=None):
    # kstemmer = Stemmer()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    text = text.translate(translator)
    text = text.lower()
    text = text.strip()
    table = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    if stemming:
        analyzer = StemmingAnalyzer(stoplist=stoplist, minsize=2, stemfn=kstemmer.stem)
    else:
        analyzer = StandardAnalyzer(stoplist=stoplist, minsize=2)

    tokens = [token.text for token in analyzer(text)]
    tokens = [word for word in tokens if not contains_digits(word)]
    return tokens


def stem(word):
    return kstemmer.stem(word)


def create_evaluate_ranking(step, rel_docs_by_qry, sim_scores_by_qry, gt_file, prog_name,
                            output_folder=os.path.dirname(os.path.realpath(__file__))):
    output_file = os.path.join(output_folder, prog_name + '_' + str(step) + '.txt')
    out = open(output_file, 'w')
    for q, rd in rel_docs_by_qry.items():
        for i in range(len(rd)):
            dname = rd[i]
            sim_score = sim_scores_by_qry[q][i]
            line = str(q) + ' Q0 ' + str(dname) + ' ' + str(i) + ' ' + str(sim_score) + ' ' + prog_name + '\n'
            out.write(line)
    out.close()
    map_v = run_trec_eval(run_to_eval=output_file, qrels_file=gt_file)
    # os.remove(output_file)
    return map_v


def build_inv_index(docs):
    """
    Builds the inverted index of the document collection, i.e. a dictionary with the words in the collection as keys and as values
    the doc ids where the term appears with the corresponding frequency.

    :param docs: a dictionary of tokenized documents accessible by doc id as key
    :return: inverted index. i.e. a dictionary with the words in the collection as keys and as values the doc ids where the
    term appears with the corresponding frequency
    """
    inverted_idx = {}
    for doc_id in tqdm(range(len(docs))):
        d = docs[doc_id]
        set_w_in_doc = set(d)
        for w in set_w_in_doc:
            if w in inverted_idx.keys():
                inverted_idx[w].append((doc_id, d.count(w)))
            else:
                inverted_idx[w] = [(doc_id, d.count(w))]
    return inverted_idx


def invert_wi(wi):
    iwi = {}
    for k, v in wi.items():
        iwi[v] = k
    return iwi


def _compute_idf(inv_idx, doc_n):
    words = list(inv_idx.keys())
    d_freqs = {}
    for k in tqdm(words):
        v = inv_idx[k]
        df = len(v)
        d_freqs[k] = df

    idf_scores = {}
    for w in words:
        idf_scores[w] = np.log((1 + doc_n) / (d_freqs[w] + 1))
    return idf_scores


def load_indri_stopwords():
    fpath = './data/indri_stoplist_eng.txt'
    sws = []
    for line in open(fpath, 'r'):
        sws.append(line.strip())
    return sws
