import os
import numpy as np
import torch.nn.init as init
import torch
from utils.parser import Parser
from torch.utils.data import TensorDataset, DataLoader
from wasserstein.operators import BuresProduct, centroid, BuresProductNormalized
from utils.losses import hinge_loss
from tqdm import tqdm


metric = BuresProductNormalized()

root = r'/mnt/DATA/Prob_IR/'
context_dataset_name = r'context_data'
encoded_docs_filename = r'encoded_docs_model'
word_index_filename = r'word_index'
emb_filename = r'embeddings_dim_2_margin_2.0'
dict = torch.load(os.path.join(root, emb_filename))

w = dict["embeddings"]
idx = 10 #indice della parola
idx2 = 20 #indice della seconda parola parola
dim = 20 #dimensione dell'embedding

# importante avere dimensione (batch_size, dim, dim) anche se batch_size = 1 come in questo caso
mean, L = (w[0:2, 0:dim].view(-1, dim), w[0:2, dim:].view((-1, dim, dim)))
mean1, L1 = (w[0:2, 0:dim].view(-1, dim), w[0:2, dim:].view((-1, dim, dim)))

distance = metric(mean1, mean, L1, L)

means = torch.cat((mean, mean1), dim=0)
covars = torch.cat((L, L1), dim=0)

# Questo richiede la soluzione di un problema di minimizzazione per calcolare il centroide (media)
# magari gioca un po' con il tol occio che quello che chiamo Lc non è la covarianza ma la cholesky della covarianza
# ovvero covar = L^T L ma a te importa poco tutte le metrice lavorano con L
mc, Lc, loss = centroid(means, covars, metric, doc_lengths=[2, 2], tol=1E-6)