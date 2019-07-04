"""
    Author: Marco Maggipinto
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import torch
from torch.nn.functional import relu
import numpy as np


def hinge_loss(ref_distrib, pos_distrib, neg_distrib, product, opt):
    num_neg, num_pos, batch_size = opt['num_negative'],  opt['num_positive'], opt['batch_size']
    rep_ind_pos = repeat_matrix_indexes(batch_size, num_pos)
    rep_ind_neg = repeat_matrix_indexes(batch_size, num_neg)
    ref_mean, ref_var = ref_distrib
    pos_mean, pos_var = pos_distrib
    neg_mean, neg_var = neg_distrib
    prod_pos = product(ref_mean[rep_ind_pos,:], pos_mean, ref_var[rep_ind_pos, :, :], pos_var)
    prod_neg = product(ref_mean[rep_ind_neg, :], neg_mean, ref_var[rep_ind_neg, :, :], neg_var)
    losses = opt['margin'] - prod_pos.view((-1, num_pos)).mean(dim=1, keepdim=True) + prod_neg.view((-1, num_neg)).mean(dim=1, keepdim=True)
    losses = relu(losses) #this computes max(0, x) as in the paper
    return losses.mean()


def trace(A):
    dim = A.shape[1]
    return torch.sum(A.view(-1, dim*dim)[:, 0:dim*dim:(dim+1)], dim=1)

def w2_loss(ref_distrib, other_distrib,  label_dist, distance):
    ref_mean, ref_var = ref_distrib
    other_mean, other_var = other_distrib
    dist = distance(ref_mean, other_mean, ref_var, other_var)
    loss = ((dist - label_dist)*(dist - label_dist)).mean()
    return loss

def repeat_matrix_indexes(size, n):
    ind = torch.arange(0, size).long()
    ind = ind.view((-1,1))
    ind = ind.expand(-1, n)
    return ind.reshape((-1))


