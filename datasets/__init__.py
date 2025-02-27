import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData, Data
from torch_geometric.datasets import Planetoid, Coauthor, Actor, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from .seeds import development_seed
from typing import List, Optional


class TransductiveNodeLearningDataset(InMemoryDataset):
    def __init__(self, root:str, name:str, **kwargs):
        super().__init__(root, **kwargs)
        self.name = name.lower()
        if self.name == 'cora':
            self._dataset = Planetoid(root=root, name='Cora', **kwargs)
        elif self.name == 'citeseer':
            self._dataset = Planetoid(root=root, name='CiteSeer', **kwargs)
        elif self.name == 'pubmed':
            self._dataset = Planetoid(root=root, name='PubMed', **kwargs)
        elif self.name == 'coauthor_cs':
            self._dataset = Coauthor(root=root, name='CS', **kwargs)
        elif self.name == 'coauthor_phy':
            self._dataset = Coauthor(root=root, name='Physics', **kwargs)
        elif self.name == 'actor':
            self._dataset = Actor(root=root, **kwargs)
        elif self.name == 'chameleon':
            self._dataset = WikipediaNetwork(root=root, name=name, **kwargs)
        elif self.name == 'squirrel':
            self._dataset = WikipediaNetwork(root=root, name=name, **kwargs)
        elif self.name == 'ogbn-arxiv':
            self._dataset = PygNodePropPredDataset(name="ogbn-arxiv")
            self._dataset.data.y = self._dataset.data.y.squeeze()
        elif self.name == 'ogbn-proteins':
            self._dataset = PygNodePropPredDataset(name="ogbn-proteins")
            data = self._dataset.get(0)
            loader = NeighborLoader(
                data,
                num_neighbors=[18]*3,
                batch_size=data.num_nodes,
                shuffle=False
            )

            new_data = next(iter(loader))
            edge_feat_adj = torch.sparse_coo_tensor(
                new_data.edge_index,
                new_data.edge_attr
            )

            degs = torch.sparse_coo_tensor(
                new_data.edge_index,
                torch.ones(new_data.edge_index.size(1))
            ).sum(dim=-1).to_dense()

            new_data.x = edge_feat_adj.sum(dim=1).to_dense() / degs.unsqueeze(-1).clamp(min=1.0)
            self._dataset.data = new_data
            self._dataset.data.y = self._dataset.data.y.squeeze()
        else:
            raise NotImplemented(f'Dataset "{name}" not implemented.')

        
        if self.name in ['cornell', 'texas', 'wisconsin', 'actor', 'chameleon', 'squirrel']:
            _data = self._dataset.get(0)
            train_mask, val_mask, test_mask = self.random_split(
                num_train_per_class=10,
                num_development=int(0.8*_data.num_nodes),
                seed=42
            )
            data = Data(
                x=_data.x,
                y=_data.y,
                edge_index=_data.edge_index,
                edge_attr=_data.edge_attr,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )

            self._dataset.data = data

    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes
    
    @property
    def num_egde_features(self) -> int:
        return self._dataset.num_edge_features
    
    @property
    def num_node_features(self) -> int:
        return self._dataset.num_node_features
    
    @property
    def num_features(self) -> int:
        return self.num_node_features
    
    def len(self) -> int:
        return self._dataset.len()
    
    def get(self, idx:int) -> BaseData:
        return self._dataset.get(idx)
    
    def random_split(
        self,
        num_train_per_class:int,
        num_development:int,
        seed:Optional[int]=None
    ):
        data = self.get(0)
        rnd_state = np.random.RandomState(development_seed)
        num_nodes = data.num_nodes

        development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
        test_idx = [i for i in range(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(seed)
        if self.name in ['cora', 'citeseer', 'pubmed']:
            for c in range(self.num_classes):
                class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
                train_idx.extend(rnd_state.choice(class_idx, num_train_per_class, replace=False))
        else:
            num_train = int(0.6*num_nodes)
            train_idx.extend(rnd_state.choice(development_idx, num_train, replace=False))
        
        val_idx = [i for i in development_idx if i not in train_idx]

        train_mask = torch.full((num_nodes,), False)
        train_mask[train_idx] = True
        val_mask = torch.full((num_nodes,), False)
        val_mask[val_idx] = True
        test_mask = torch.full((num_nodes,), False)
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask