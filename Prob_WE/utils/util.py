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

import emot
# import en_core_web_sm
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
from krovetzstemmer import Stemmer


# choose correct variable values according to what pc I am using
# if platform.node() == 'acquario':
#     OS_PYTHON = r'/home/alberto/anaconda3/envs/tensorflow/bin/python3'
#     TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
# elif platform.node() == 'DESKTOP-LKU0VTG':
#     OS_PYTHON = r'C:\Users\apirp\anaconda\envs\tf\python.exe'
#     TREC_EVAL_PATH = r'trec_eval-master\trec_eval.exe'
# elif platform.node() == 'hopper':
#     OS_PYTHON = '/home/ims/anaconda3/envs/tensorflow_env/bin/python3'
#     TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
# elif platform.node() == 'alberto-Alienware-15-R4':
#     OS_PYTHON = r'/home/alberto/anaconda3/envs/tf/bin/python3'
#     TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
# else:
TREC_EVAL_PATH = 'trec_eval.8.1/trec_eval'
OS_PYTHON = r'/home/alberto/anaconda3/envs/tf/bin/python3'

def create_summary_chart(losses, maps, suff=''):
    plt.switch_backend('agg')
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(losses, c='b', label='loss')
    ax1.legend()
    ax2.plot(maps, c='r', label='map_score')
    ax2.legend()
    plt.savefig('analyses/maps_and_losses_' + suff + '.png')
    plt.close()
    return


def find_first_prime_number_after_k(k):
    if k < 10000000019:
        return 10000000019
    print('finding valid prime number...')
    candidate = k
    while True:
        candidate += 1
        is_prime = True
        print('evaluating candidate=%s' % candidate)
        for i in tqdm(range(2, candidate)):
            if (i % i) == 0:
                is_prime = False
                break
        if is_prime:
            return candidate


def get_hash(x, a, b, p, m):
    """
    a must be in [1, (p-1)]
    b must be in [1, (p-1)]
    p is a prime number that is greater than max possible value of x
    h(x, a, b) = ((ax + b) mod p)mod m
    m is a max possible value you want for hash code + 1
    """
    if a >= p or b >= p:
        print('error in input params of hash function')
        return None
    return ((a * x + b) % p) % m


def read_files_from_folder(input_folder):
    files = []
    fnames = []
    for filename in os.listdir(input_folder):
        if filename.startswith('.') or os.path.isdir(os.path.join(input_folder, filename)):
            continue
        with open(os.path.join(input_folder, filename), 'r', encoding='latin-1') as f:  # Reading file
            files.append(f.read())
            fnames.append(filename)
    return files, fnames


def hash_docs(word_dict, encoded_docs, repr_size, n_hash_functions):
    hashed_docs = []
    dict_size = len(word_dict)
    p = find_first_prime_number_after_k(dict_size)
    print('hashing docs...')
    for d in tqdm(encoded_docs):
        hashed_d = np.zeros(repr_size)
        for w in d:
            for b in range(n_hash_functions):
                i = get_hash(w, 2, b, p, repr_size)
                hashed_d[i] = hashed_d[i] + 1
        hashed_docs.append(hashed_d)
    return np.array(hashed_docs)


def read_lines_data_file(fpath, separator='\t', encoding='latin-1'):
    docs_text = []
    labels_text = []

    for data in read_tsv_file(fpath, delimiter=separator, encoding=encoding):
        if len(data[1].strip()) == 0:
            continue
        labels_text.append(data[1].strip())
        docs_text.append(data[2].strip())
    return docs_text, labels_text


def read_tsv_file(path, delimiter='\t', headers=False, encoding='latin-1'):
    rows = []
    if headers:
        cnt = 0
    else:
        cnt = 1
    with io.open(path, 'r', encoding=encoding, errors='ignore') as f:
        for line in csv.reader((x.replace('\0', '') for x in f), delimiter=delimiter):
            if cnt > 0:
                rows.append(line)
            else:
                cnt += 1
    return rows


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
    map_value = map_line.split('\t')[2]
    return float(map_value)


def chunk(l, size):
    chnks = []
    for i in range(0, len(l), size):
        chnks.append(l[i: min(i + size, len(l))])
    return chnks


def get_embeddings_matrix(word_index, embs_path='models/GoogleNews-vectors-negative300.bin', emb_size=300):
    vocab_size = len(word_index.keys())
    model = KeyedVectors.load_word2vec_format(embs_path, binary=True)
    embeddings = np.zeros((vocab_size + 1, emb_size))

    for k, v in word_index.items():
        if k in model.wv.vocab:
            embeddings[v] = model[k]
        else:
            embeddings[v] = np.zeros(emb_size)  # unk words are mapped to all zero vector

    return np.array(embeddings, dtype=np.float32), vocab_size, emb_size


def split_doc_in_frames(collection, max_sentence_size=20, max_sentences_per_doc=None):
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


@jit
def encode_sequence(sequence, word_dict):
    encoded_sequence = []
    for w in sequence:
        word = w.text
        if word in word_dict.keys():
            encoded_sequence.append([word_dict[word]])
    return encoded_sequence


def load_indri_stopwords():
    fpath = './data/indri_stoplist_eng.txt'
    sws = []
    for line in open(fpath, 'r'):
        sws.append(line.strip())
    return sws


def contains_digits(token):
    for c in token:
        if c.isdigit():
            return True
    return False


def encode_docs_split_in_sentences(docs_split_in_sentences, word_dict):
    encoded_docs = []
    for doc in tqdm(docs_split_in_sentences):
        encoded_doc = []
        for sentence in doc:
            encoded_sentence = []
            for w in sentence:
                if w in word_dict.keys():
                    encoded_sentence.append(word_dict[w])
            if len(encoded_sentence) > 0:
                encoded_doc.append(encoded_sentence)
        encoded_docs.append(encoded_doc)
    return encoded_docs


# def split_in_sents_then_encode(docs, dictionary):
#     nlp = en_core_web_sm.load()
#     # split documents in sentences
#     docs_split_in_sentences = []
#     for d in tqdm(docs):
#         d = d.replace('\n', ' ')
#         d = d.replace('  ', ' ')
#         d = d.strip()
#         doc = nlp(d)
#         docs_split_in_sentences.append([s.text for s in list(doc.sents)])
#
#     # encoded documents
#     encoded_docs = []
#     for doc in docs_split_in_sentences:
#         encoded_doc = []
#         for sent in doc:
#             encoded_sent = []
#             tokens = nlp(sent)
#             for t in tokens:
#                 token = t.text
#                 if token in dictionary.keys():
#                     encoded_sent.append(dictionary[token])
#             if len(encoded_sent) > 0:
#                 encoded_doc.append(encoded_sent)
#         encoded_docs.append(encoded_doc)
#
#     return encoded_docs


def stem_tokens(tokens):
    for i in range(len(tokens)):
        tokens[i] = kstemmer.stem(tokens[i])
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def extract_emojis_emoticons(text):
    extracted = []
    vals = emot.emoticons(text)
    if len(vals) > 1:
        extracted.extend(vals['value'])

    vals = emot.emoji(text)
    if len(vals) > 1:
        extracted.extend(vals['value'])
    return extracted


def encode_spark(docs, dictionary, stopwords=None):
    os.environ['PYSPARK_PYTHON'] = OS_PYTHON
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '10g').set('spark.executor.memory', '10g')
    sc = SparkContext(conf=conf)
    encoded_docs = sc.parallelize(docs).map(lambda d: encode_doc(d, dictionary, stopwords)).collect()
    sc.stop()
    return encoded_docs


def rerank_docs_spark(q, docs_to_rerank, we):
    os.environ['PYSPARK_PYTHON'] = OS_PYTHON
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '10g').set('spark.executor.memory', '10g')
    sc = SparkContext(conf=conf)
    sims = sc.parallelize(docs_to_rerank).map(lambda d: compute_q_d_sim(q, d, we)).collect()
    sc.stop()
    return sims


def compute_q_d_sim(q, d, we):
    exact_match_sim = np.sum([1 for w in d if w in q]) / len(d)
    sim = np.sum(cosine_similarity([q], [we[word] for word in d])) / len(d)
    return sim + exact_match_sim


def encode_doc(d, dictionary, stopwords):
    if stopwords is not None:
        stemmed = tokenize_preprocess_document_standard(d, stopwords)
    else:
        stemmed = tokenize_preprocess_document_standard(d)
    new_d = []
    for t in stemmed:
        if dictionary.get(t) is not None:
            new_d.append(dictionary[t])
    return new_d


def encode(docs, dictionary):
    encoded_docs = []
    for d in tqdm(docs):
        tokenized = tokenize(d)
        new_d = []
        for t in tokenized:
            if dictionary.get(t) is not None:
                new_d.append(dictionary[t])
        encoded_docs.append(new_d)
    return encoded_docs


def encode_collection(docs, max_n_words=None, min_df=1, max_df=1.0):
    print('computing terms frequencies...')
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=max_df, min_df=min_df, max_features=max_n_words)
    vectorizer.fit(docs)
    features_list = vectorizer.get_feature_names()
    dictionary = {}
    for w in features_list:
        dictionary[w] = len(dictionary)
    print('dictionary size=' + str(len(dictionary.keys())))
    print('turning text to sequences...')
    encoded_docs = []
    for d in tqdm(docs):
        tokenized = tokenize(d)
        new_d = []
        for t in tokenized:
            if dictionary.get(t) is not None:
                new_d.append(dictionary[t])
        encoded_docs.append(new_d)
    return np.array(encoded_docs), dictionary


def compute_sim(d, q, sim_scores_index):
    if len(d) == 0:
        doc_score = 0
    else:
        neural_sim_score = [sim_scores_index[t] for t in d]
        sim_score = np.mean(neural_sim_score)
        exact_match_score = np.sum([1 for w in q if w in d]) / len(q)  # add exact match bonus term
        doc_score = sim_score + exact_match_score
    return doc_score


def compute_doc_similarity_spark(documents, sim_scores_index, q):
    os.environ['PYSPARK_PYTHON'] = OS_PYTHON
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '10g').set('spark.executor.memory', '10g')
    sc = SparkContext(conf=conf)
    sims = sc.parallelize(documents).map(lambda doc: compute_sim(doc, q, sim_scores_index)).collect()
    sc.stop()
    return sims


def load_w2v_model(word_dictionary):
    model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
    emb_matrix = []
    word_dict_new = {}
    for k, v in word_dictionary.items():
        if k in model.wv.vocab:
            emb_matrix.append(model.wv[k])
            word_dict_new[k] = len(word_dict_new)
    emb_matrix.append(np.zeros(300))  # for padding values
    return np.array(emb_matrix), word_dict_new


def tokenize(text, stemming=True, stoplist=None):
    text = text.lower()
    text = text.strip()
    if stemming:
        analyzer = StemmingAnalyzer(stoplist=stoplist, minsize=2, stemfn=kstemmer.stem)
    else:
        analyzer = StandardAnalyzer(stoplist=stoplist, minsize=2)

    tokens = [token.text for token in analyzer(text)]
    tokens = [word for word in tokens if not contains_digits(word)]
    return tokens


def tokenize_preprocess_document_standard(doc, stoplist=None):
    # doc = doc.lower()
    # doc = doc.strip()
    # tokenized_doc = doc.split(' ')
    # if stoplist is not None:
    #     tokenized_doc = [w for w in tokenized_doc if w not in sw]
    # tokenized_doc = [word for word in tokenized_doc if not contains_digits(word)]
    # tokenized_doc = stem_tokens(tokenized_doc)
    return tokenize(doc, stoplist)


def train_w2v_model(documents):
    emb_size = 300
    print('tokenizing and stemming collection')
    tokenized_stemmed_docs = []
    for d in tqdm(documents):
        tokenized_stemmed_docs.append(tokenize_preprocess_document_standard(d))

    print('training word embeddings')
    model = gensim.models.Word2Vec(tokenized_stemmed_docs, negative=10, size=emb_size, window=10, min_count=10,
                                   workers=12, sample=1e-4, sg=0)
    model.train(tokenized_stemmed_docs, total_examples=model.corpus_count, epochs=10)
    emb_matrix = []
    word_dict = {}
    for k in model.wv.vocab:
        word_dict[k] = len(word_dict)
        emb_matrix.append(model.wv[k])
    emb_matrix.append(np.zeros(emb_size))  # for padding values
    return np.array(emb_matrix), word_dict


def train_w2v_gensim(documents, word_dictionary):
    print('Training word embeddings')
    emb_size = 300
    docs = [tokenize(d, stemming=True) for d in documents]
    model = gensim.models.Word2Vec(docs, negative=10, size=emb_size, window=10, min_count=10, workers=12, sample=1e-4,
                                   sg=0)
    model.train(docs, total_examples=model.corpus_count, epochs=10)
    emb_matrix = []
    word_dict_new = {}
    for k, v in word_dictionary.items():
        if k in model.wv.vocab:
            emb_matrix.append(model.wv[k])
            word_dict_new[k] = len(word_dict_new)
    emb_matrix.append(np.zeros(emb_size))  # for padding values
    return np.array(emb_matrix), word_dict_new


def split_doc_in_sequences(d, max_seq_len):
    sequences = []
    s = []
    for w in d:
        s.append(w)
        if len(s) == max_seq_len:
            sequences.append(s)
            s = []
    sequences.append(s)
    return sequences


def pad_documents(docs_batch, max_seq_len, padding_value):
    docs_split_in_seqs = []
    for d in docs_batch:
        docs_split_in_seqs.append(split_doc_in_sequences(d, max_seq_len))
    true_seqs_lengths = []
    for d in docs_split_in_seqs:
        true_seqs_lengths.append([len(s) for s in d])

    # pad each sequence
    for d in docs_split_in_seqs:
        new_d = []
        for s in d:
            while len(s) < max_seq_len:
                s.append(padding_value)
            new_d.append(s)
    # add n
    max_n_sequences_in_doc = max([len(d) for d in docs_split_in_seqs])
    for i in range(len(docs_split_in_seqs)):
        d = docs_split_in_seqs[i]
        sequences_lengths = true_seqs_lengths[i]
        while len(d) < max_n_sequences_in_doc:
            d.append([padding_value] * max_seq_len)
            sequences_lengths.append(0)
        docs_split_in_seqs[i] = d
        true_seqs_lengths[i] = sequences_lengths

    return docs_split_in_seqs, true_seqs_lengths


def pad_sequences(sequences, padding_value, ml=-1):
    if ml == -1:
        ml = max([len(s) for s in sequences])
    new_list = np.ones(shape=(len(sequences), ml)) * padding_value
    for i in range(new_list.shape[0]):
        for j in range(len(sequences[i])):
            new_list[i, j] = sequences[i][j]
    return new_list
