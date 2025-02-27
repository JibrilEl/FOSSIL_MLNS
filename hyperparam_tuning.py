import argparse
import random
import sys
import os
import os.path as osp

import numpy as np

from eval_tools import  LRE

import torch

import scipy.sparse as sp
import torch.nn as nn
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.convert import to_scipy_sparse_matrix

import torch.nn.functional as F
import losses
import utils
from collections import defaultdict

from models.model import Model
from datasets import TransductiveNodeLearningDataset
from datasets.seeds import development_seed

import tqdm
import optuna
import logging
from yaml import SafeLoader
from yaml import load as yamlload
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--study_name', type=str, default='cora_tuning')
parser.add_argument('--study_folder', type=str, default='studies')
parser.add_argument('--loss_fn', type=str, default='fgwd')
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--sampler', type=str, choices=['TPE', 'Random'], default='TPE')
parser.add_argument('--sampl', type=float, default=1.0)
parser.add_argument('--ground_metric', type=str, default='cosine')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--savepath', type=str, default='save/state.pt')
parser.add_argument('--logfile', type=str, default='logs.json')
parser.add_argument('--resume', type=bool, default=False)

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(epoch, model, x, adj1, adj2, deg, adj, node_neighbor_cen, optimizer, loss_function, alpha, beta2, beta3, metric="cosine", **kwargs):
    model.train()

    optimizer.zero_grad()

    hs, hc, h_gen, beta = model(x, adj1, adj2, deg)
    h = hs + torch.mul(beta.detach(), hc)
    
    ### train gnn
    loss_gnn = loss_function.compute(h, h_gen, adj, node_neighbor_cen, metric=metric, alpha=alpha, **kwargs)


    loss_gnn.backward()
    optimizer.step()
    optimizer.zero_grad()

    ### train mlp
    hs, hc, h_gen, beta = model(x, adj1, adj2, deg)


    hetero_loss, _ = model.cos_loss(hs.detach(), hc.detach(), beta)
    L2_loss = torch.norm(beta, dim = 0)
    reg = torch.abs(torch.mean(beta) - (1-alpha))
    mlp_loss = hetero_loss + beta2 * reg + beta3 * L2_loss
    mlp_loss.backward()

    optimizer.step()
    
    return loss_gnn.item(), mlp_loss.item()

def test(model, labels, x, adj1, adj2, deg, idx_train, idx_val):

    model.eval()

    with torch.no_grad():
        representations = model.get_embedding(x, adj1, adj2, deg)

    result = LRE(representations, labels, idx_train, idx_val)

    return result

config = yamlload(open(args.config), Loader=SafeLoader)[args.dataset.lower()]
data_root = osp.join(os.getcwd(), 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TransductiveNodeLearningDataset(root=data_root, name=args.dataset)
data = dataset.get(0).to(device)
num_nodes = data.num_nodes
num_features = data.x.shape[1]
edges, _ = remove_self_loops(data.edge_index)
adj1 = sparse_mx_to_torch_sparse_tensor(sp.eye(num_nodes)).coalesce().to(device)

adj = to_scipy_sparse_matrix(edge_index=edges, num_nodes = num_nodes)
adj2 = sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device)

deg = sp.coo_matrix.sum(adj, axis=1)
deg = torch.tensor(deg)
deg = deg.to(device)

adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index.cpu()[0, :], data.edge_index.cpu()[1, :])),
                    shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)

adj = sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device)
adj_lists = defaultdict(set)
for i in range(data.edge_index.size(1)):
    adj_lists[data.edge_index[0][i].item()].add(data.edge_index[1][i].item())

idx_train, idx_val, _ = dataset.random_split(
    num_train_per_class=config['num_train_per_class'],
    num_development=config['num_development'],
    seed=development_seed
)

def objective(trial):
    num_hidden = config['num_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    alpha = trial.suggest_categorical('alpha', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tau = trial.suggest_categorical("tau", [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    num_epochs = config['num_epochs']
    lr = trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001, 0.005, 0.01])
    lr_mlp = trial.suggest_categorical('lr_mlp', [0.0001, 0.0005, 0.001, 0.005, 0.01])
    weight_decay = config['weight_decay']
    wd_mlp = config['wd_mlp']
    num = config['num']
    k1 = trial.suggest_int('k1', 10, 30)
    reg = trial.suggest_categorical('reg', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2])
    drop = trial.suggest_categorical('drop', [0.1, 0.2, 0.3, 0.4])
    dropout_mlp = trial.suggest_categorical('dropout_mlp', [0.1, 0.2, 0.3, 0.4])
    gate = config['gate']
    beta2 = config['beta2']
    beta3 = trial.suggest_categorical('beta3', [1e1, 1e2, 1e3, 1e4, 1e5, 1e6])

    model = Model(num_features, num_hidden, drop, dropout_mlp, gate, activation)
    model = model.to(device)
    loss_function = losses.build_loss(args.loss_fn, tau=tau)
    optimizer = torch.optim.Adam([
        {'params':model.params, 'lr': lr, 'weight_decay': weight_decay},
        {'params':model.params_mlp,'weight_decay': wd_mlp,'lr': lr_mlp}
    ])

    epochs = tqdm.trange(1, num_epochs+1)

    ## SSL training
    for epoch in epochs:

        weights = torch.ones(data.num_nodes)
        nodes_batch = torch.multinomial(weights, num, replacement=False)
        node_neighbor_cen = utils.sub_sam(nodes_batch, adj_lists, k1, p=args.sampl)
        
        train(epoch, model, data.x, adj1, adj2, deg, adj, node_neighbor_cen, optimizer,
                    loss_function=loss_function, alpha=alpha, beta2=beta2, beta3=beta3, metric=args.ground_metric, reg=reg)

    ## SSL downstream performance
    result = test(model, data.y, data.x, adj1, adj2, deg, idx_train, idx_val)

    return result['val_acc']

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = args.study_name
study_folder = args.study_folder
if not Path(study_folder).exists():
    Path(study_folder).mkdir(parents=True)
storage_name = f"sqlite:///{study_folder}/{study_name}.db"
if args.sampler == 'TPE':
    sampler = optuna.samplers.TPESampler(seed=development_seed)
else:
    sampler = optuna.samplers.RandomSampler(seed=development_seed)
study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, sampler=sampler, load_if_exists=True)
if args.resume:
    n_existing_trials = len(study.trials)
    if n_existing_trials > 0:
        if args.trials > n_existing_trials:
            print(f"\033[32mFound some existing trials ({n_existing_trials}), completing to {args.trials} trials...\033[0m")
            study.optimize(objective, n_trials=args.trials - n_existing_trials)
    print(f'\033[33mThe requested number of trials is already done. You may want to execute the program with option "--resume False" to add more trials\033[0m')
else:
    study.optimize(objective, n_trials=args.trials)
