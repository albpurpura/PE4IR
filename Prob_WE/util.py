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
from krovetzstemmer import Stemmer
import string
import emot
import en_core_web_sm
import gensim
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from numba import jit
from pyspark import SparkContext, SparkConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer

kstemmer = Stemmer()
# choose correct variable values according to what pc I am using
if platform.node() == 'acquario':
    OS_PYTHON = r'/home/alberto/anaconda3/envs/tensorflow/bin/python3'
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
elif platform.node() == 'DESKTOP-LKU0VTG':
    OS_PYTHON = r'C:\Users\apirp\anaconda\envs\tf\python.exe'
    TREC_EVAL_PATH = r'trec_eval-master\trec_eval.exe'
elif platform.node() == 'hopper':
    OS_PYTHON = '/home/ims/anaconda3/envs/tensorflow_env/bin/python3'
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
else:
    TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'

with open('./data/indri_stoplist_eng.txt', 'r') as slf:
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
    if platform.node() == 'acquario' or platform.node() == 'hopper' or platform.node() == 'alberto-Alienware-15-R4':
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


def create_evaluate_ranking(step, rel_docs_by_qry, sim_scores_by_qry, gt_file, prog_name):
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), prog_name + '_' + str(step) + '.txt')
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
