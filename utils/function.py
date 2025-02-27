import random
import warnings
import copy
import torch
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.transforms import Compose
from torch_geometric.sampler.base import *

warnings.filterwarnings("ignore")



# subgraph sampling
def sub_sam(nodes, adj_lists, k, p=0.7):
    node_neighbor =  [ [] for i in range(nodes.shape[0])]
    node_neighbor_cen =  [ [] for i in range(nodes.shape[0])]
    node_centorr =  [[] for i in range(nodes.shape[0])]
    
    num_nei = 0
    flag = 0
    for node in nodes:
        p1 = random.uniform(1-(1-p)*1.3,1.)
        neighbors = set([int(node)])
        neighs = set(random.sample(list(adj_lists[int(node)]), int(p1*len(adj_lists[int(node)]))))
        node_centorr[num_nei] = [int(node)]
        current1 = neighs
        if len(neighs) >= k:
            neighs -= neighbors
            current1 = random.sample(list(neighs), k-1)
            node_neighbor[num_nei] = [neg_node for neg_node in current1]
            current1.append(int(node))
            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            num_nei += 1
        
        else:
            num_while = 0
            while len(current1) < k:
                current2 = set()
                current1 = sam_nexthop(adj_lists, current1, current2, k, p=p1)
                if num_while > 5:
                    break
                num_while += 1
            if num_while > 5:
                flag += 1
                continue
            
            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            if int(node) in node_neighbor_cen[num_nei]:
                node_neighbor_cen[num_nei].remove(int(node))
            node_neighbor[num_nei] = random.sample(node_neighbor_cen[num_nei], k-1)
            node_neighbor_cen[num_nei] = [neg_node for neg_node in node_neighbor[num_nei]]
            node_neighbor_cen[num_nei].append(int(node))
            num_nei += 1

    if flag > 0:
        node_neighbor_new =  [ [] for i in range(len(node_neighbor)-flag)]
        node_neighbor_cen_new =  [ [] for i in range(len(node_neighbor)-flag)] 
        node_centorr_new =  [ [] for i in range(len(node_neighbor)-flag)] 
        for i in range(len(node_neighbor)-flag):
            node_neighbor_new[i] = node_neighbor[i]   
            node_neighbor_cen_new[i] = node_neighbor_cen[i]   
            node_centorr_new[i] = node_centorr[i]
        return node_neighbor_cen_new
        
    return node_neighbor_cen

def sam_nexthop(adj_lists, sam_current, current2, k, p=1):
    for neigh in sam_current:
        current2 |= set(random.sample(list(adj_lists[int(neigh)]), int(p*len(adj_lists[int(neigh)]))))
    if len(current2) < k: 
        return current2 
    else:
        current2 -= sam_current 
        current2 = random.sample(list(current2), k-len(sam_current))
        [current2.append(nei) for nei in sam_current]
        return current2


class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index

        edge_index, edge_attr = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)