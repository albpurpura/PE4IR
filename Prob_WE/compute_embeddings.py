from utils.input_output import load_dataset, build_context_dataset
import tables
import os
import numpy as np
import torch.nn.init as init
import torch
from utils.parser import Parser
from torch.utils.data import TensorDataset, DataLoader
from wasserstein.operators import BuresProduct
from utils.losses import hinge_loss
from tqdm import tqdm
from visdom import Visdom
from utils.visual import *

args = [
    [('--margin', '-mg'), {'type': float, 'default': 2.0, 'help': 'Margin used for the distance'}],
    [('--dim',), {'type': int, 'default': 2, 'help': 'Dimension of the embedding'}],
    [('--lr',), {'type': float, 'default': 0.01, 'help': 'Learning rate'}],
    [('--num_positive', '-np'), {'type': int, 'default': 2, 'help': 'Number of same class elements per training sample'}],
    [('--num_negative', '-nn'), {'type': int, 'default': 1, 'help': 'Number of different class elements per training sample'}],
    [('--batch_size', '-bs'), {'type': int, 'default': 16*4096, 'help': 'Batch size'}],
    [('--epochs', '-e'), {'type': int, 'default': 15, 'help': 'Number of epochs'}],
    [('--num_iters', '-ni'), {'type': int, 'default': 15, 'help': 'Number of iterations for the matrix sqrt algorithm'}],
    [('--reg', ), {'type': float, 'default': 2.0, 'help': 'Regularization for the matrix sqrt algorithm'}],
    [('--reg2', ), {'type': float, 'default': 1E-8, 'help': 'Regularization for the gradient of the bures metric'}],
    [('--log_period', '-lp'), {'type': int, 'default': 50, 'help': 'Logging period'}]
]

argparser = Parser("Deep Elliptical Embeddings")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

viz = Visdom(port=8098)
vm = VisualManager(viz, 'marco')

root = r'/mnt/DATA/Prob_IR/'
context_dataset_name = r'context_data'
encoded_docs_filename = r'encoded_docs_model'
word_index_filename = r'word_index'
emb_filename = r'embeddings_dim_' + str(opt['dim']) + '_margin_' + str(opt['margin'])
emb_path = os.path.join(root, emb_filename)

context_dataset_path = os.path.join(root, context_dataset_name)
print("Loading data...")

_, words = load_dataset(root, encoded_docs_filename, word_index_filename)
idx_words = np.array(list(range(len(words))))

atom = tables.Int32Atom()

with tables.open_file(context_dataset_path, mode='r') as f:
    train_context = torch.Tensor(np.array(f.root.data[:]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dim = 20
w = torch.zeros(len(words), dim + dim**2, device=device)
init.xavier_uniform_(w)
w.requires_grad = True

ntot = opt['num_positive'] + 1
opt['tot_batch_size'] = ntot*opt['batch_size']




dataset = TensorDataset(train_context)
loader = DataLoader(dataset=dataset, batch_size=opt['batch_size'], num_workers=3, shuffle=True, drop_last=True)


def get_reference_indices(opt):
    num_positive = opt['num_positive']
    tot_batch_size = opt['tot_batch_size']
    indices = torch.zeros(tot_batch_size, dtype=torch.int8)
    indices[0:tot_batch_size:num_positive+1] = 1
    return indices.byte()


product = BuresProduct()

optimizer = torch.optim.SGD(list((w,)),
                            weight_decay=0.0,
                            lr=opt['lr'],
                            momentum=0.9,
                            nesterov=True)

loss_logger = LineLogger(vm, 'Hinge loss', opt['log_period'], None)

for e in range(opt['epochs']):
    print("Epoch: " + str(e) + " of " + str(opt['epochs']))
    for i, batch in enumerate(tqdm(loader)):
        indices = batch[0]
        indices = indices.long().view(opt['batch_size'] * ntot)
        wbatch = w[indices]
        ind_neg = np.random.choice(np.setdiff1d(idx_words, indices.numpy()), opt['batch_size'], replace=True)
        ref_indeces = get_reference_indices(opt)

        emb_ref = wbatch[ref_indeces]
        emb_context = wbatch[1-ref_indeces]
        emb_neg = w[ind_neg]

        ref_distrib = (emb_ref[:, 0:dim], emb_ref[:, dim:].view((-1, dim, dim)))
        context_distrib = (emb_context[:, 0:dim], emb_context[:, dim:].view((-1, dim, dim)))
        neg_distrib = (emb_neg[:, 0:dim], emb_neg[:, dim:].view((-1, dim, dim)))

        optimizer.zero_grad()

        loss = hinge_loss(ref_distrib, context_distrib, neg_distrib, product, opt)
        loss.backward()

        optimizer.step()

        if torch.isnan(loss.data):
            raise Exception

        if i%opt['log_period'] == 0:
            loss_logger.update(loss.data.tolist())

    torch.save({'embeddings': w}, emb_path)



