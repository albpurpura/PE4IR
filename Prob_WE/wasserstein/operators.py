"""
    Author: Marco Maggipinto
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import time

import torch


class BuresProduct:
    def __init__(self, e=1E-8, num_iters=20, reg=2.0):
        self.e = e
        self.num_iters = num_iters
        self.reg = reg

    def __call__(self, a, b, L_A, L_B):
        bures = BuresMetric.apply
        dim = a.shape[1]
        prod = a.view(-1, 1, dim).matmul(b.view(-1, dim, 1)).squeeze(dim=2)
        return prod + bures(L_A, L_B, self.e, self.num_iters, self.reg)


class BuresProductNormalized:
    def __init__(self, e=1E-8, num_iters=20, reg=2.0):
        self.e = e
        self.num_iters = num_iters
        self.reg = reg

    def __call__(self, a, b, L_A, L_B):
        bures = BuresMetricNormalized.apply
        dim = a.shape[1]
        prod = a.view(-1, 1, dim).matmul(b.view(-1, dim, 1)).squeeze(dim=2) / (a.norm(dim=1) * b.norm(dim=1)).view(-1, 1)
        return prod + bures(L_A, L_B, self.e, self.num_iters, self.reg)


class DistanceW2:
    def __init__(self, e=1E-8, num_iters=20, reg=2.0):
        self.e = e
        self.num_iters = num_iters
        self.reg = reg

    def __call__(self, a, b, L_A, L_B):
        bures = BuresMetric2.apply
        dim = a.shape[1]
        diff = a - b
        dist = torch.norm(diff, dim=1, keepdim=True)
        return dist ** 2  # + 1/(dim+2)*bures(L_A, L_B, self.e, self.num_iters, self.reg)


class BuresProductNormalizedModule(torch.nn.Module):
    def __init__(self, e=1E-8, num_iters=20, reg=2.0):
        super().__init__()
        self.e = e
        self.num_iters = num_iters
        self.reg = reg

    def forward(self, a, b, L_A, L_B):
        bures = BuresMetricNormalized.apply
        dim = a.shape[1]
        prod = a.view(-1, 1, dim).matmul(b.view(-1, dim, 1)).squeeze(dim=2) / (a.norm(dim=1) * b.norm(dim=1)).view(-1,
                                                                                                                   1)
        return prod + bures(L_A, L_B, self.e, self.num_iters, self.reg)


class BuresMetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L_A, L_B, e=1E-8, num_iters=20, reg=2.0):
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        if L_A.shape[2] != dim:
            raise Exception("Matrix must be square")

        A = L_A.matmul(L_A.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)
        B = L_B.matmul(L_B.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)

        Y1, Z1 = matrix_square_root(A, num_iters, reg)
        supp = Y1.matmul(B).matmul(Y1)
        Y2, Z2 = matrix_square_root(supp, num_iters, reg)

        T_AB = Z1.matmul(Y2).matmul(Z1)
        T_BA = Y1.matmul(Z2).matmul(Y1)

        output = torch.zeros((batch_size, 1)).to(device)
        for i in range(batch_size):
            output[i, 0] = torch.trace(Y2[i, :, :])

        ctx.save_for_backward(L_A, L_B, T_AB, T_BA)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        L_A, L_B, T_AB, T_BA = ctx.saved_tensors
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        I = torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(device)
        # grad_L_A = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * (I - T_AB).matmul(L_A)
        # grad_L_B = grad_output.expand(-1, dim*dim).view(-1, dim,dim) * (I - T_BA).matmul(L_B)
        grad_L_A = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * T_AB.matmul(L_A)
        grad_L_B = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * T_BA.matmul(L_B)
        return grad_L_A, grad_L_B, None, None, None


class BuresMetricNormalized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L_A, L_B, e=1E-8, num_iters=20, reg=2.0):
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        if L_A.shape[2] != dim:
            raise Exception("Matrix must be square")

        A = L_A.matmul(L_A.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)
        B = L_B.matmul(L_B.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)

        Y1, Z1 = matrix_square_root(A, num_iters, reg)
        supp = Y1.matmul(B).matmul(Y1)
        Y2, Z2 = matrix_square_root(supp, num_iters, reg)

        T_AB = Z1.matmul(Y2).matmul(Z1)
        T_BA = Y1.matmul(Z2).matmul(Y1)

        output = torch.zeros((batch_size, 1)).to(device)
        trA = torch.zeros((batch_size, 1)).to(device)
        trB = torch.zeros((batch_size, 1)).to(device)
        for i in range(batch_size):
            trA[i, 0] = torch.trace(A[i, :, :])
            trB[i, 0] = torch.trace(B[i, :, :])
            output[i, 0] = torch.trace(Y2[i, :, :]) / torch.sqrt(trA[i] * trB[i])

        ctx.save_for_backward(L_A, L_B, T_AB, T_BA, trA, trB, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        L_A, L_B, T_AB, T_BA, trA, trB, output = ctx.saved_tensors
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        I = torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(device)
        # grad_L_A = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * (I - T_AB).matmul(L_A)Normalized
        # grad_L_B = grad_output.expand(-1, dim*dim).view(-1, dim,dim) * (I - T_BA).matmul(L_B)
        grad_L_A = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * T_AB.matmul(L_A) / (trA * trB).sqrt().view(
            batch_size, 1, 1) + \
                   (output * trB / ((trA * trB) ** 3).sqrt()).view(-1, 1, 1) * L_A
        grad_L_B = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * T_BA.matmul(L_B) / (trA * trB).sqrt().view(
            batch_size, 1, 1) + \
                   (output * trA / ((trA * trB) ** 3).sqrt()).view(-1, 1, 1) * L_B

        return grad_L_A, grad_L_B, None, None, None


class BuresMetric2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L_A, L_B, e=1E-8, num_iters=20, reg=2.0):
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        if L_A.shape[2] != dim:
            raise Exception("Matrix must be square")

        A = L_A.matmul(L_A.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)
        B = L_B.matmul(L_B.permute((0, 2, 1))) + e * torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(
            device)

        Y1, Z1 = matrix_square_root(A, num_iters, reg)
        supp = Y1.matmul(B).matmul(Y1)
        Y2, Z2 = matrix_square_root(supp, num_iters, reg)

        T_AB = Z1.matmul(Y2).matmul(Z1)
        T_BA = Y1.matmul(Z2).matmul(Y1)

        output = torch.zeros((batch_size, 1)).to(device)
        for i in range(batch_size):
            output[i, 0] = torch.trace(A[i, :, :] + B[i, :, :] - 2 * Y2[i, :, :])

        ctx.save_for_backward(L_A, L_B, T_AB, T_BA)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        L_A, L_B, T_AB, T_BA = ctx.saved_tensors
        device = L_A.device
        batch_size = L_A.shape[0]
        dim = L_A.shape[1]
        I = torch.eye(dim).view((1, dim, dim)).repeat(batch_size, 1, 1).to(device)
        grad_L_A = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * (I - T_AB).matmul(L_A)
        grad_L_B = grad_output.expand(-1, dim * dim).view(-1, dim, dim) * (I - T_BA).matmul(L_B)
        # grad_L_A = grad_output.expand(-1, dim*dim).view(-1, dim,dim) * T_AB.matmul(L_A)
        # grad_L_B = grad_output.expand(-1, dim*dim).view(-1, dim,dim) * T_BA.matmul(L_B)
        return grad_L_A, grad_L_B, None, None, None


class RelTol:
    def __init__(self, dim):
        self.old_param = torch.zeros((1, dim))

    def __call__(self, param):
        param = param.detach()
        device = param.device
        tol = (param - self.old_param.to(device)).norm()
        self.old_param = param.data.clone()
        return tol / param.norm()


def centroid(means, L, metric, doc_lengths, tol=1E-3, lrd=0.9999):
    device = means.device
    dim = means.shape[1]
    t = RelTol(dim ** 2)
    n_docs = len(doc_lengths)
    Lc = torch.randn((n_docs, dim, dim), requires_grad=True, device=device)
    mc = torch.randn((n_docs, dim), requires_grad=True, device=device)
    tl = float('Inf')
    optimizer = torch.optim.SGD(list((Lc, mc)), lr=0.1)
    ind = get_indeces(doc_lengths)
    # lc_exp = Lc[ind, :, :]

    while tl > tol:
        ind = get_indeces(doc_lengths)
        # start = time.time()
        optimizer.zero_grad()
        # lc_exp = Lc.index_select(0, ind.to(Lc.device))
        # mc_exp = mc.index_select(0, ind.to(Lc.device))
        loss = -metric(mc[ind, :], means, Lc[ind, :, :], L).mean()
        # loss = -metric(mc_exp, means, lc_exp, L).mean()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lrd
        # tl = t(Lc.view(n_docs, -1))
        tl = t(Lc.data.view(n_docs, -1))
        # print('time for one cycle: %2.4s' % (time.time() - start))
    return mc, Lc


def centroid_alt(w, q, metric, doc_lengths, tol=1E-3, lrd=0.9999):
    device = w.device
    dim = 50
    # dim = w.shape[1]
    t = RelTol(dim ** 2)
    n_docs = len(doc_lengths)
    Lc = torch.randn((n_docs, dim, dim), requires_grad=True, device=device)
    mc = torch.randn((n_docs, dim), requires_grad=True, device=device)
    tl = float('Inf')
    optimizer = torch.optim.SGD(list((Lc, mc)), lr=0.1)
    # ind = get_indeces(doc_lengths)
    # lc_exp = Lc[ind, :, :]

    while tl > tol:
        m, v = (w[q, 0:dim].view(-1, dim), w[q, dim:].view((-1, dim, dim)))
        ind = get_indeces(doc_lengths)
        # start = time.time()
        optimizer.zero_grad()
        # lc_exp = Lc.index_select(0, ind.to(Lc.device))
        # mc_exp = mc.index_select(0, ind.to(Lc.device))
        loss = -metric(mc[ind, :], m, Lc[ind, :, :], v).mean()
        # loss = -metric(mc_exp, means, lc_exp, L).mean()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lrd
        # tl = t(Lc.view(n_docs, -1))
        tl = t(Lc.data.view(n_docs, -1))
        # print('time for one cycle: %2.4s' % (time.time() - start))
    return mc, Lc


def get_indeces(doc_lenghts):
    n_docs = len(doc_lenghts)
    l = []
    for i in range(n_docs):
        l.append(torch.ones(doc_lenghts[i]) * i)
    return torch.cat(l).long()


def matrix_square_root(A, num_iters=20, reg=2.0):
    A = A.detach()
    device = A.device
    batch_size = A.shape[0]
    dim = A.shape[1]

    if A.shape[2] != dim:
        raise Exception("Matrix must be square")

    normA = reg * frobenius(A)

    Y = A.view(batch_size, -1).div(normA)
    Y = Y.view(batch_size, dim, dim)
    I = torch.eye(dim).reshape(1, dim, dim).repeat(batch_size, 1, 1).to(device)
    Z = torch.eye(dim).reshape(1, dim, dim).repeat(batch_size, 1, 1).to(device)

    for i in range(num_iters):
        T = 0.5 * (3.0 * I - torch.matmul(Z, Y))
        Y = torch.matmul(Y, T)
        Z = torch.matmul(T, Z)

    sqrtA = Y.view(batch_size, -1) * torch.sqrt(normA)
    sqrtA = sqrtA.view(batch_size, dim, dim)
    sqrtAinv = Z.view(batch_size, -1).div(torch.sqrt(normA))
    sqrtAinv = sqrtAinv.view(batch_size, dim, dim)

    return sqrtA, sqrtAinv


def frobenius(A):
    batch_size = A.shape[0]
    return torch.norm(A.view(batch_size, -1), dim=1).view(batch_size, 1)
