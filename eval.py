""" Adapted from the code of https://arxiv.org/pdf/2307.16026.pdf and https://arxiv.org/pdf/2207.11996.pdf"""

import argparse
import random
import os
import os.path as osp

import numpy as np
import seaborn as sns


from datasets.seeds import test_seeds
from eval_tools import  LRE
from utils.logger import Logger

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
from datasets.seeds import test_seeds

import tqdm
import json
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
parser.add_argument('--loss_fn', type=str, default='fgwd')
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--sampl', type=float, default=1.0)
parser.add_argument('--ground_metric', type=str, default='cosine')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--savepath', type=str, default='save/state.pkl')
parser.add_argument('--logfile', type=str, default='logs.json')
parser.add_argument('--resume', type=bool, default=False)

args = parser.parse_args()

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(epoch, model, x, adj1, adj2, deg, adj, node_neighbor_cen, optimizer, loss_function, alpha, beta2, beta3, metric="cosine", **kwargs):
    model.train()

    optimizer.zero_grad()

    hs, hc, h_gen, beta = model(x, adj1, adj2, deg)
    h = hs + torch.mul(beta.detach(), hc)
    
    ### train gnn
    loss_gnn = loss_function.compute(h, h_gen, adj, node_neighbor_cen, alpha=alpha, metric=metric, **kwargs)


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

def test(model, labels, x, adj1, adj2, deg, idx_train, idx_val, idx_test):

    model.eval()

    with torch.no_grad():
        representations = model.get_embedding(x, adj1, adj2, deg)

    result = LRE(representations, labels, idx_train, idx_val, idx_test)
   
    return result

config = yamlload(open(args.config), Loader=SafeLoader)[args.dataset.lower()]
num_hidden = config['num_hidden']
activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
alpha = config['alpha']
tau = config['tau']
num_epochs = config['num_epochs']
lr = config['lr']
lr_mlp = config['lr_mlp']
weight_decay = config['weight_decay']
wd_mlp = config['wd_mlp']
num, k1 = config['num'], config['k1']
reg = config['reg']
drop = config['drop']
dropout_mlp = config['dropout_mlp']
gate = config['gate']
beta2 = config['beta2']
beta3 = config['beta3']

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



train_accs = []
val_accs = []
test_accs = []

exp_key = None
if args.resume:
    path = Path(f"./logs/{args.logfile}")
    logs = json.load(path.open())
    last_exp_key = list(logs.keys())[-1]
    experiments = logs[last_exp_key]["experiments"]
    last_experiment = experiments[-1]
    last_seed = last_experiment["seed"]
    idx = test_seeds.index(last_seed)
    seeds = test_seeds[idx+1:]
    for exp in experiments:
        train_accs.append(exp["train_accuracy"])
        val_accs.append(exp["val_accuracy"])
        test_accs.append(exp["test_accuracy"])
    
    exp_key = last_exp_key
else:
    exp_key = None
    seeds = test_seeds

logger = Logger(config=config, logfile=args.logfile, exp_key=exp_key, args=args)
logger.update_logs(update={
    'experiments': []
})

for seed in seeds:
    seed_it(seed)

    idx_train, idx_val, idx_test = dataset.random_split(
        num_train_per_class=config['num_train_per_class'],
        num_development=config['num_development'],
        seed=seed
    )

    model = Model(num_features, num_hidden, drop, dropout_mlp, gate, activation)
    model = model.to(device)

    loss_function = losses.build_loss(args.loss_fn, tau=tau)
    optimizer = torch.optim.Adam([
        {'params':model.params, 'lr': lr, 'weight_decay': weight_decay},
        {'params':model.params_mlp,'weight_decay': wd_mlp,'lr': lr_mlp}
    ])

    fixed_subgraph = None
    best_train_acc = .0
    best_val_acc = .0
    best_test_acc = .0
    epochs = tqdm.trange(1, num_epochs+1)
   
    for epoch in epochs:
        wandb_log_data = {}

        weights = torch.ones(data.num_nodes)
        nodes_batch = torch.multinomial(weights, num, replacement=False)
        node_neighbor_cen = utils.sub_sam(nodes_batch, adj_lists, k1, p=args.sampl)
        
        train(epoch, model, data.x, adj1, adj2, deg, adj, node_neighbor_cen, optimizer,
                    loss_function=loss_function, alpha=alpha, reg=reg, beta2=beta2, beta3=beta3, metric=args.ground_metric)

    ## SSL downstream performance
    result = test(model, data.y, data.x, adj1, adj2, deg, idx_train, idx_val, idx_test)
    train_accs.append(result['train_acc'])
    val_accs.append(result['val_acc'])
    test_accs.append(result['test_acc'])

    logger.update_logs(update={
        'experiments': {
            'seed': seed,
            'train_accuracy': result['train_acc'],
            'val_accuracy': result['val_acc'],
            'test_accuracy': result['test_acc']
        }
    })


train_accs = np.array(train_accs)
val_accs = np.array(val_accs)
test_accs = np.array(test_accs)
train_boots_series = sns.algorithms.bootstrap(train_accs, func=np.mean, n_boot=1000)
train_acc_mean  = np.mean(train_boots_series)
train_acc_std = np.max(np.abs(sns.utils.ci(train_boots_series, 95) - np.mean(train_accs)))

val_boots_series = sns.algorithms.bootstrap(val_accs, func=np.mean, n_boot=1000)
val_acc_mean  = np.mean(val_boots_series)
val_acc_std = np.max(np.abs(sns.utils.ci(val_boots_series, 95) - np.mean(val_accs)))

test_boots_series = sns.algorithms.bootstrap(test_accs, func=np.mean, n_boot=1000)
test_acc_mean  = np.mean(test_boots_series)
test_acc_std = np.max(np.abs(sns.utils.ci(test_boots_series, 95) - np.mean(test_accs)))

logger.update_logs(update={
    'train_acc_mean': train_acc_mean,
    'train_acc_std': train_acc_std,
    'val_acc_mean': val_acc_mean,
    'val_acc_std': val_acc_std,
    'test_acc_mean': test_acc_mean,
    'test_acc_std': test_acc_std
})