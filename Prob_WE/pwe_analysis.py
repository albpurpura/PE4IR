"""
    Author: Alberto Purpura
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import gensim
import krovetz
import matplotlib
import torch
import numpy as np
import json

from adjustText import adjust_text
from fastText import load_model
from sklearn.decomposition import PCA

from utils import util
from wasserstein.operators import BuresProductNormalizedModule
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def plot_pwe_sim_m(w_list, w_list_labels, w, wi):
    fig = plt.figure()
    pwe_sim_m = np.zeros((len(w_list), len(w_list)))
    for i in range(len(w_list)):
        w1 = w_list[i]
        for j in range(len(w_list)):
            w2 = w_list[j]
            pwe_sim_m[i, j] = compute_words_sim_pwe(w1, w2, w, wi)
    with sns.axes_style("white"):
        ax = sns.heatmap(pwe_sim_m, vmin=-1.0, vmax=1.0, cmap=sns.color_palette("Blues"), square=True,
                         xticklabels=w_list_labels, yticklabels=w_list_labels)
        # ax.set_title('PWE similarity matrix')
        plt.show()
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        sns.set(font_scale=1.2)
        plt.savefig('pwe_sim_m.png')
        plt.close(fig)


def plot_ft_sim_m(w_list, w_list_labels, model):
    ftt_sim_m = np.zeros((len(w_list), len(w_list)))
    fig = plt.figure()
    for i in range(len(w_list)):
        w1 = w_list[i]
        for j in range(len(w_list)):
            w2 = w_list[j]
            ftt_sim_m[i, j] = compute_word_sim_ftt(w1, w2, model)
    with sns.axes_style("white"):
        ax = sns.heatmap(ftt_sim_m, vmin=-1.0, vmax=1.0, cmap=sns.color_palette("Blues"), square=True,
                         xticklabels=w_list_labels, yticklabels=w_list_labels)
        # ax.set_title('FTT similarity matrix')
        # plt.show()
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        sns.set(font_scale=1.2)
        plt.savefig('ftt_sim_m.png')
        plt.close(fig)


def plot_wnwe_sim_m(w_list, w_list_labels, model):
    wn_sim_m = np.zeros((len(w_list), len(w_list)))
    fig = plt.figure()
    for i in range(len(w_list)):
        w1 = w_list[i]
        for j in range(len(w_list)):
            w2 = w_list[j]
            if w1 in model.wv.vocab and w2 in model.wv.vocab:
                wn_sim_m[i, j] = compute_word_sim_wnwe(w1, w2, model)
            else:
                wn_sim_m[i, j] = -1.0

    with sns.axes_style("white"):
        ax = sns.heatmap(wn_sim_m, vmin=-1.0, vmax=1.0, cmap=sns.color_palette("Blues"), square=True,
                         xticklabels=w_list_labels, yticklabels=w_list_labels)
        # ax.set_title('WNE similarity matrix')
        # plt.show()
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        sns.set(font_scale=1.2)
        plt.savefig('wnwe_sim_m.png')
        plt.close(fig)


def project(basis, v):
    rval = np.zeros(v.shape)
    for b in basis:
        rval += np.dot(v, b) * b
    return rval


def plot_we_scatterplots(word_list, model, model_name='ft', wi=None):
    labels = word_list
    kstemmer = krovetz.PyKrovetzStemmer()
    reducer = PCA(2)
    if model_name == 'ft':
        vecs = reducer.fit_transform(np.array([model.get_word_vector(kstemmer.stem(w)) for w in word_list]))
    elif model_name == 'wn':
        word_list = [w for w in word_list if w in model.wv.vocab]
        labels = word_list
        vecs = reducer.fit_transform(model[[w for w in word_list]])
    else:
        vecs = reducer.fit_transform([model[wi[kstemmer.stem(w)]] for w in word_list])

    fig = plt.figure()
    plt.grid(True)
    plt.axhspan(0, 0, linewidth=2, color='#1f77b4')
    plt.axvline(0)
    ax = plt.gca()
    ax.set_xlim(left=-10, right=10)
    ax.set_ylim(bottom=-10, top=10)
    texts = []
    xy = []
    for i in range(len(vecs)):
        reduced = vecs[i]
        label = labels[i]
        ax.scatter(reduced[0], reduced[1], s=100, alpha=1.0, color='b', marker='+')
        # texts.append(ax.text(reduced[0], reduced[1], label, fontsize=14))
        if label == 'osteoporosis' and model_name == 'wn':
            ax.annotate(label, (reduced[0], reduced[1]), fontsize=18, ha='right')
        else:
            ax.annotate(label, (reduced[0], reduced[1]), fontsize=18)

        xy.append(reduced)
    # adjust_text(texts, expand_text=(0.01, 0.02), arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.0))

    if model_name == 'ft':
        plt.title('FTE')
    elif model_name == 'wn':
        plt.title('WNE')
    else:
        plt.title('W2V')

    xmax = -100
    xmin = 100
    ymin = 100
    ymax = -100
    for i in range(len(xy)):
        coord = xy[i]
        if coord[0] > xmax:
            xmax = coord[0]
        if coord[0] < xmin:
            xmin = coord[0]

        if coord[1] > ymax:
            ymax = coord[1]
        if coord[1] < ymin:
            ymin = coord[1]

    ax.set_xlim(left=xmin - 1, right=xmax + 1)
    ax.set_ylim(bottom=ymin - 1, top=ymax + 1)

    plt.show()
    plt.close(fig)


def plot_ellipses_alt(words, w, wi):
    kstemmer = krovetz.PyKrovetzStemmer()
    np.random.seed(0)
    dim = 50
    plt.grid(True)
    plt.axhspan(0, 0, linewidth=2, color='#1f77b4')
    plt.axvline(0)
    ax = plt.gca()
    ws = [kstemmer.stem(w.lower()) for w in words]
    labels = []
    mean_vectors = []
    all_eigenvectors_eigenvalues = []
    for i in range(len(words)):
        w1 = ws[i]
        index = wi[w1]
        w1_m, w1_c = (w[index, 0:dim].view(-1, dim), w[index, dim:].view((-1, dim, dim)))
        w1_c = np.reshape(w1_c.detach().numpy(), newshape=(dim, dim))
        w1_m = np.reshape(w1_m.detach().numpy(), newshape=-1)

        prec_matr = np.linalg.inv(w1_c * w1_c.T)
        eigs = np.linalg.eig(prec_matr)
        norms = [np.linalg.norm(v) for v in eigs[1]]
        eigenvalues = np.abs(eigs[0])
        eigenvectors = np.array([eigs[1][i] / norms[i] for i in range(len(norms))])
        sorted_eigenvalues = eigenvalues[np.argsort(-eigenvalues)]
        sorted_v = eigenvectors[np.argsort(-eigenvalues)][0:2]

        all_eigenvectors_eigenvalues.append((sorted_v, sorted_eigenvalues))
        mean_vectors.append(w1_m)

    reducer = PCA(2)

    # mean_vectors_all = np.array([np.reshape(w[index, 0:dim].view(-1, dim).detach().numpy(), -1) for index in range(len(wi.items()))])
    # reducer.fit(mean_vectors_all)
    # centers = reducer.transform(np.array(mean_vectors))

    centers = reducer.fit_transform(np.array(mean_vectors))
    basis = reducer.transform(reducer.components_)

    widths = []
    heights = []
    angles = []
    xy = []
    max_eig0 = np.NINF
    max_eig1 = np.NINF
    for i in range(len(all_eigenvectors_eigenvalues)):
        eigenvectors, eigenvalues = all_eigenvectors_eigenvalues[i]
        if eigenvalues[0] > max_eig0:
            max_eig0 = eigenvalues[0]

        if eigenvalues[1] > max_eig1:
            max_eig1 = eigenvalues[1]
    max_overall = max_eig0
    if max_eig1 > max_overall:
        max_overall = max_eig1

    for i in range(len(all_eigenvectors_eigenvalues)):
        label = words[i]
        eigenvectors, eigenvalues = all_eigenvectors_eigenvalues[i]
        proj_eigs = reducer.transform(np.array(eigenvectors))

        width = np.linalg.norm(proj_eigs[0]) * eigenvalues[0] / max_eig0
        height = np.linalg.norm(proj_eigs[1]) * eigenvalues[1] / max_eig1
        angle = np.arccos(np.dot(proj_eigs[0], basis[0]) / (np.linalg.norm(proj_eigs[0]) * np.linalg.norm(basis[0])))

        widths.append(width)
        heights.append(height)
        angles.append(angle)
        x = centers[i][0]
        y = centers[i][1]
        xy.append((x, y))
        labels.append(label)

    texts = []
    colors = []
    xy = [(xy[i][0] * 1, xy[i][1] * 1) for i in range(len(xy))]
    for i in range(len(widths)):
        # widths = [w / max(widths) for w in widths]
        # heights = [h / max(heights) for h in heights]
        # xy = [(xy[i][0] * 2.2, xy[i][1] * 2.2) for i in range(len(xy))]
        print(labels[i])
        print('width=%2.8f, height=%2.8f, x=%2.5f, y=%2.5f' % (widths[i], heights[i], xy[i][0], xy[i][1]))
        ell_color = np.random.rand(3)
        ell = matplotlib.patches.Ellipse(xy=xy[i], width=widths[i], height=heights[i], angle=angles[i],
                                         facecolor=ell_color, edgecolor=ell_color, fill=True)
        ax.add_patch(ell)
        ell.set_zorder(-1)
        ell.set_alpha(np.random.rand())
        ell.set_alpha(0.5)
        ell.set(label=labels[i], clip_box=ax.bbox)
        colors.append(ell_color)
        ell.set_facecolor(ell_color)

        texts.append(ax.text(xy[i][0], xy[i][1], labels[i], fontsize=18))
        plt.scatter(xy[i][0], xy[i][1], s=80, alpha=0.8, color=ell_color, edgecolor=ell_color, marker='+')

    adjust_text(texts, expand_text=(1.5, 2.5), expand_points=(2.5, 2.5), expand_objects=(1.9, 2.8),
                expand_align=(1.8, 1.7), arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.8))

    xmax = -100
    xmin = 100
    ymin = 100
    ymax = -100
    wmax = -100
    hmax = -100
    for i in range(len(xy)):
        coord = xy[i]
        if wmax < widths[i]:
            wmax = widths[i]
        if hmax < heights[i]:
            hmax = heights[i]

        if coord[0] > xmax:
            xmax = coord[0]
        if coord[0] < xmin:
            xmin = coord[0]

        if coord[1] > ymax:
            ymax = coord[1]
        if coord[1] < ymin:
            ymin = coord[1]

    ax.set_xlim(left=xmin - wmax / 2 - 0.05, right=xmax + wmax / 2 + 0.05)
    ax.set_ylim(bottom=ymin - hmax / 2 - 0.05, top=ymax + hmax / 2 + 0.05)
    ax.legend(prop={'size': 13})
    plt.savefig('ellipses_final.png')
    plt.show()
    plt.close()


def plot_ellipses(word_lists, w, wi):
    kstemmer = krovetz.PyKrovetzStemmer()
    np.random.seed(0)
    dim = 50
    plt.grid(True)
    ax = plt.gca()

    for words in word_lists:
        ws = [kstemmer.stem(w.lower()) for w in words]
        basis = None
        widths = []
        heights = []
        angles = []
        xy = []
        labels = []
        basis_eigenvalues = None
        for i in range(len(words)):
            label = words[i]
            w1 = ws[i]
            index = wi[w1]
            w1_m, w1_c = (w[index, 0:dim].view(-1, dim), w[index, dim:].view((-1, dim, dim)))
            w1_c = np.reshape(w1_c.detach().numpy(), newshape=(dim, dim))
            w1_m = np.reshape(w1_m.detach().numpy(), newshape=-1)

            prec_matr = np.linalg.inv(w1_c * w1_c.T)

            eigs = np.linalg.eig(prec_matr)
            norms = [np.linalg.norm(v) for v in eigs[1]]
            eigenvalues = np.abs(eigs[0])
            eigenvectors = np.array([eigs[1][i] / norms[i] for i in range(len(norms))])

            sorted_v = eigenvectors[np.argsort(-eigenvalues)][0:2]
            width = np.sqrt(np.abs(eigenvalues[np.argsort(-eigenvalues)][0]))
            height = np.sqrt(np.abs(eigenvalues[np.argsort(-eigenvalues)][1]))

            if basis is None:
                sorted_eigenvalues = eigenvalues[np.argsort(-eigenvalues)]
                expl_variance = np.abs(np.sum(sorted_eigenvalues[0:2])) / np.sum(sorted_eigenvalues[2:]) * 100

                print('PWE: explained variance: %2.5f' % expl_variance)
                basis_eigenvalues = (width, height)
                basis = sorted_v
                angle = 0
                # center = (0, 0)
                center = (
                    basis_eigenvalues[0] * np.dot(basis[0], w1_m) / (np.linalg.norm(basis[0]) * np.linalg.norm(w1_m)),
                    basis_eigenvalues[1] * (np.dot(basis[1], w1_m) / (np.linalg.norm(basis[1]) * np.linalg.norm(w1_m))))

            else:
                proj0 = project(basis, sorted_v[0])
                # proj1 = project(basis, sorted_v[1])
                angle = np.arccos(np.dot(proj0, basis[0]) / (np.linalg.norm(proj0) * np.linalg.norm(basis[0])))
                # mean = (np.dot(basis[0], w1_m), np.dot(basis[1], w1_m))

                width *= np.linalg.norm(project(basis, eigs[1][0]))
                height *= np.linalg.norm(project(basis, eigs[1][1]))

                center = (
                    width * np.dot(basis[0], w1_m) / (np.linalg.norm(basis[0]) * np.linalg.norm(w1_m)),
                    height * (np.dot(basis[1], w1_m) / (np.linalg.norm(basis[1]) * np.linalg.norm(w1_m))))

            widths.append(width)
            heights.append(height)
            angles.append(angle)
            xy.append(center)
            labels.append(label)

        texts = []
        colors = []
        xy = [(xy[i][0] * 1, xy[i][1] * 1) for i in range(len(xy))]
        for i in range(len(widths)):
            # widths = [w / max(widths) for w in widths]
            # heights = [h / max(heights) for h in heights]
            # xy = [(xy[i][0] * 2.2, xy[i][1] * 2.2) for i in range(len(xy))]
            print(labels[i])
            print('width=%2.8f, height=%2.8f, x=%2.5f, y=%2.5f' % (widths[i], heights[i], xy[i][0], xy[i][1]))
            ell_color = np.random.rand(3)
            ell = matplotlib.patches.Ellipse(xy=xy[i], width=widths[i], height=heights[i], angle=angles[i],
                                             facecolor=ell_color, edgecolor=ell_color, fill=True)
            ax.add_patch(ell)
            ell.set_zorder(-1)
            ell.set_alpha(np.random.rand())
            ell.set_alpha(0.5)
            ell.set(label=labels[i], clip_box=ax.bbox)
            # ell.set_edgecolor(ell_color)
            colors.append(ell_color)
            ell.set_facecolor(ell_color)

            texts.append(ax.text(xy[i][0], xy[i][1], labels[i], fontsize=18))
            plt.scatter(xy[i][0], xy[i][1], s=80, alpha=0.8, color=ell_color, edgecolor=ell_color, marker='+')

        adjust_text(texts, expand_text=(1.5, 2.5), expand_points=(2.5, 2.5), expand_objects=(1.9, 2.8),
                    expand_align=(1.8, 1.7), arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.8))

        xmax = -100
        xmin = 100
        ymin = 100
        ymax = -100
        wmax = -100
        hmax = -100
        for i in range(len(xy)):
            coord = xy[i]
            if wmax < widths[i]:
                wmax = widths[i]
            if hmax < heights[i]:
                hmax = heights[i]

            if coord[0] > xmax:
                xmax = coord[0]
            if coord[0] < xmin:
                xmin = coord[0]

            if coord[1] > ymax:
                ymax = coord[1]
            if coord[1] < ymin:
                ymin = coord[1]

        ax.set_xlim(left=int(xmin - wmax / 2), right=int(xmax + wmax / 2))
        ax.set_ylim(bottom=int(ymin - hmax / 2), top=int(ymax + hmax / 2))
    ax.legend()
    plt.show()
    plt.close()


def get_top_k_closest_words_w_ftt(w1, k, wi, model):
    keys = list(wi.keys())
    sims = []
    for w2 in tqdm(keys):
        sim = compute_word_sim_ftt(w1, w2, model)
        sims.append(sim)
    sims = np.array(sims)
    keys = np.array(list(keys))
    sorted_keys = keys[np.argsort(-sims)]
    return sorted_keys[:k], sims[np.argsort(-sims)][:k]


def compute_word_sim_ftt(word1, word2, model):
    w1 = model.get_word_vector(word1)
    w2 = model.get_word_vector(word2)

    sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    return sim


def get_top_k_closest_words_w_wne(w1, k, wi, model):
    keys = list(wi.keys())
    sims = []
    for w2 in tqdm(keys):
        if w1 in model.wv.vocab and w2 in model.wv.vocab:
            sim = compute_word_sim_wnwe(w1, w2, model)
        else:
            sim = 0.0
        sims.append(sim)
    sims = np.array(sims)
    keys = np.array(list(keys))
    sorted_keys = keys[np.argsort(-sims)]
    return sorted_keys[:k], sims[np.argsort(-sims)][:k]


def compute_word_sim_wnwe(word1, word2, model):
    w1 = model[word1]
    w2 = model[word2]

    sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    return sim


def compute_top_k_closest_words_to(w1, k, wi, w):
    keys = list(wi.keys())
    sims = []
    for w2 in tqdm(keys):
        sim = compute_words_sim_pwe(w1, w2, w, wi)
        sims.append(sim)
    sims = np.array(sims)
    keys = np.array(list(keys))
    sorted_keys = keys[np.argsort(-sims)]
    return sorted_keys[:k], sims[np.argsort(-sims)][:k]


def compute_words_sim_pwe(word1, word2, w, wi):
    dim = 50
    w1 = wi[word1]
    w2 = wi[word2]

    # print('w1=%s' % word1)
    # print('w2=%s' % word2)

    w1_m, w1_c = (w[w1, 0:dim].view(-1, dim), w[w1, dim:].view((-1, dim, dim)))
    w2_m, w2_c = (w[w2, 0:dim].view(-1, dim), w[w2, dim:].view((-1, dim, dim)))

    metric = torch.nn.DataParallel(BuresProductNormalizedModule())
    similarities = metric(w1_m, w2_m, w1_c, w2_c)
    sim = np.reshape(np.array(similarities.cpu().detach().numpy()), newshape=-1)[0]
    return (sim + 1) * 2 / 3 - 1


def load_w2v_model(wi_path, we_path):
    we = util.load_model(we_path)
    wi = util.load_model(wi_path)
    return we, wi


def get_top_k_closest_words_w2v(w1, k, wi, model):
    keys = list(wi.keys())
    sims = []
    w1_vec = model[wi[w1]]
    for w2 in tqdm(keys):
        w2_vec = model[wi[w2]]
        sim = np.dot(w1_vec, w2_vec) / (np.linalg.norm(w1_vec) * np.linalg.norm(w2_vec))
        sims.append(sim)
    sims = np.array(sims)
    keys = np.array(list(keys))
    sorted_keys = keys[np.argsort(-sims)]
    return sorted_keys[:k], sims[np.argsort(-sims)][:k]


def run():


    plt.rcParams['figure.dpi'] = 300
    kstemmer = krovetz.PyKrovetzStemmer()

    print('loading w2v model')
    wi_path = 'data/w2v/word_index_stemmed'
    we_path = 'data/w2v/word_embeddings_matrix'
    w2v_model, w2v_wi = load_w2v_model(wi_path, we_path)

    print('loading PWE')
    word_dict_path = 'data/word_index_json'
    wi = load_json(word_dict_path)
    embs_path = 'data/embeddings_dim_50_margin_2.0'
    dict_pt = torch.load(embs_path, map_location='cpu')
    pwe_model = dict_pt["embeddings"]

    """
    keys, sims = get_top_k_closest_words_w2v(kstemmer.stem('cuba'), 30, w2v_wi, w2v_model)
    print('closes words to cuba (w2v): %s' % ', '.join(keys))

    keys, sims = compute_top_k_closest_words_to('cuba', 30, wi, w)
    print('closes words to cuba (pwe): %s' % ', '.join(keys))

    keys, sims = get_top_k_closest_words_w2v(kstemmer.stem('sugar'), 30, w2v_wi, w2v_model)
    print('closes words to sugar (w2v): %s' % ', '.join(keys))

    keys, sims = compute_top_k_closest_words_to('sugar', 30, wi, w)
    print('closes words to sugar (pwe): %s' % ', '.join(keys))

    keys, sims = get_top_k_closest_words_w2v(kstemmer.stem('export'), 30, w2v_wi, w2v_model)
    print('closes words to export (w2v): %s' % ', '.join(keys))

    keys, sims = compute_top_k_closest_words_to('export', 30, wi, w)
    print('closes words to export (pwe): %s' % ', '.join(keys))
    """

    # ['disease', 'osteoporosis', 'fracture', 'bone', 'diet', 'health', 'drug']
    # w_lists = [['osteoporosis', 'disease', 'fracture', 'bone', 'magnesium', 'diet']]
    w_list = ['osteoporosis', 'disease', 'fracture', 'bone', 'magnesium', 'diet']
    # plot_ellipses([w_list], pwe_model, wi)
    plot_ellipses_alt(w_list, pwe_model, wi)

    exit()

    plot_pwe_sim_m([kstemmer.stem(w) for w in w_list], w_list, pwe_model, wi)

    for w in w_list:
        stemmed = kstemmer.stem(w)
        sim = compute_words_sim_pwe(kstemmer.stem('osteoporosis'), stemmed, pwe_model, wi)
        print('PWE similarity between osteoporosis and %s: %2.5f' % (w, sim))

    print()

    print('loading fasttext model')
    ftext_model_path = 'data/wiki.en.bin'
    f = load_model(ftext_model_path)

    print('loading wn embs')
    wn2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/wn2vec.txt')

    plot_we_scatterplots(w_list, f, model_name='ft')
    plot_we_scatterplots(w_list, wn2vec_model, model_name='wn')
    plot_we_scatterplots(w_list, w2v_model, model_name='w2v', wi=w2v_wi)

    for w in w_list:
        stemmed = kstemmer.stem(w)
        sim = compute_word_sim_wnwe('osteoporosis', w, wn2vec_model)
        print('WNE similarity between osteoporosis and %s: %2.5f' % (w, sim))

    print()

    for w in w_list:
        stemmed = kstemmer.stem(w)
        sim = compute_word_sim_ftt(kstemmer.stem('osteoporosis'), stemmed, f)
        print('FTE similarity between osteoporosis and %s: %2.5f' % (w, sim))

    print()

    for w in w_list:
        stemmed = kstemmer.stem(w)
        sim = compute_word_sim_wnwe(w2v_wi[kstemmer.stem('osteoporosis')], w2v_wi[kstemmer.stem(w)], w2v_model)
        print('W2V similarity between osteoporosis and %s: %2.5f' % (w, sim))


if __name__ == '__main__':
    run()
