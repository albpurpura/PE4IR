"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import platform
import string

from pyspark import SparkConf, SparkContext
from tqdm import tqdm
from krovetzstemmer import Stemmer
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
import pickle
import os
import json
import numpy as np
from multiprocessing import Pool

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


def load_indri_stopwords(fpath):
    sws = []
    for line in open(fpath, 'r'):
        sws.append(line.strip())
    return sws


def contains_digits(token):
    for c in token:
        if c.isdigit():
            return True
    return False


def tokenize(text, stemming=True, stoplist=None):
    kstemmer = Stemmer()
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


def split_doc_in_frames(collection, max_sentence_size=20, max_sentences_per_doc=None):
    def chunk(l, size):
        chnks = []
        for i in range(0, len(l), size):
            chnks.append(l[i: min(i + size, len(l))])
        return chnks

    fixed_sizes_collections = []
    for i in tqdm(range(len(collection))):
        d = collection[i]
        # a document is a list of sentences
        max_sentence_size_in_words = max_sentence_size
        frames = chunk(d, max_sentence_size_in_words)
        if max_sentences_per_doc is not None:
            frames = frames[:max_sentences_per_doc]
        fixed_sizes_collections.append(frames)
    return fixed_sizes_collections


def read_files_from_folder(input_folder):
    files = []
    fnames = []
    print('reading data')
    for filename in tqdm(os.listdir(input_folder)):
        if filename.startswith('.') or os.path.isdir(os.path.join(input_folder, filename)):
            continue
        with open(os.path.join(input_folder, filename), 'r', encoding='latin-1') as f:  # Reading file
            files.append(f.read())
            fnames.append(filename)
    return files, fnames


def encode_spark(docs, dictionary, stopwords=None):
    os.environ['PYSPARK_PYTHON'] = OS_PYTHON
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '10g').set('spark.executor.memory', '10g')
    sc = SparkContext(conf=conf)
    encoded_docs = sc.parallelize(docs).map(lambda d: encode_doc(d, dictionary, stopwords)).collect()
    sc.stop()
    return encoded_docs


def tokenize_collection(documents, stemming, stoplist):
    os.environ['PYSPARK_PYTHON'] = OS_PYTHON
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '20g').set('spark.driver.maxResultSize', '5g') \
        .set('spark.executor.memory', '20g')
    sc = SparkContext(conf=conf)
    tokenized_docs = sc.parallelize(documents). \
        map(lambda doc: tokenize(doc, stemming=stemming, stoplist=stoplist)).collect()
    sc.stop()
    return tokenized_docs


def encode_doc(d, dictionary, stopwords):
    if stopwords is not None:
        stemmed = tokenize(d, stopwords)
    else:
        stemmed = tokenize(d)
    new_d = []
    for t in stemmed:
        if dictionary.get(t) is not None:
            new_d.append(dictionary[t])
    return new_d


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


def build_inv_index(docs):
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


def _compute_idf(inv_idx, doc_n, min_df=10, d_freq_upper_bound=1.0):
    words = sorted(list(inv_idx.keys()))
    d_freqs = {}
    new_w_index = {}
    for k in tqdm(words):
        v = inv_idx[k]
        df = len(v)
        d_freqs[k] = df
        if df / doc_n < d_freq_upper_bound and df > min_df:
            new_w_index[k] = len(new_w_index)

    idf_scores = {}
    for w in words:
        idf_scores[w] = np.log((1 + doc_n) / (d_freqs[w] + 1))

    return idf_scores, new_w_index


def compute_idf(docs, min_df=1, max_d_freq=1.0):
    doc_n = len(docs)
    print('building inverted index')
    inv_idx = build_inv_index(docs)
    # save_model(inv_idx, 'inv_idx')
    # inv_idx = load_model('inv_idx')
    print('computing idf scores')
    idf_scores, new_w_index = _compute_idf(inv_idx, doc_n, min_df, max_d_freq)
    return idf_scores, new_w_index


def read_data_pipeline(root, collection_folder, stop_word_file):
    collection_path = os.path.join(root, collection_folder)
    stop_word_path = os.path.join(root, stop_word_file)
    docs, d_names = read_files_from_folder(collection_path)
    tokenized_stemmed_docs = tokenize_collection(docs, stemming=True, stoplist=load_indri_stopwords(stop_word_path))
    t_idfs, word_index = compute_idf(tokenized_stemmed_docs, 10, 1.0)
    print('word index size = %d' % len(word_index))
    print('encoding documents (2)')
    encoded_docs = encode_spark(docs, word_index, load_indri_stopwords(stop_word_path))
    save_model(encoded_docs, 'encoded_docs_model')
    save_model(word_index, 'word_index')


def _build_context(doc):
    context = []
    for i in range(len(doc) - 1):
        if i > 1:
            context.append(doc[i - 1:i + 2])
    return np.array(context)


def build_context_dataset(docs, num_workers):
    """
    Occhio che serve ram :)
    :param docs: List of documents
    :param num_workers: Number of threads
    :return: a list with the [context, word, context]
    """
    with Pool(num_workers) as p:
        result = p.map(_build_context, docs)
        result = list(result)
        for x in result:
            x.shape = (-1, 3)
        result = np.concatenate(result)
    return result


def load_dataset(root, encoded_docs_filename, word_index_filename):
    encoded_docs_path = os.path.join(root, encoded_docs_filename)
    word_index_path = os.path.join(root, word_index_filename)
    return load_model(encoded_docs_path), load_model(word_index_path)


def load_texts(queries_folder, stop_word_path):
    queries, q_names = read_files_from_folder(queries_folder)
    tokenized_stemmed_queries = tokenize_collection(queries, stemming=True,
                                                    stoplist=load_indri_stopwords(stop_word_path))
    return tokenized_stemmed_queries, q_names


if __name__ == '__main__':
    coll_folder = '/media/alberto/DATA/ExperimentalCollections/Robust04/processed/corpus'
    stop_file = r'./data/indri_stoplist_eng.txt'
    read_data_pipeline("", coll_folder, stop_file)
