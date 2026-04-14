import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData, Data
from torch_geometric.datasets import Planetoid, Coauthor, Actor, WikipediaNetwork, TUDataset, WikiCS, Amazon, HeterophilousGraphDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from .seeds import development_seed
from typing import List, Optional
from torch_geometric.loader import RandomNodeSampler

class TransductiveNodeLearningDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, **kwargs):
        super().__init__(root, **kwargs)
        self.name = name.lower()
        
        # --- Existing Logic ---
        if self.name == 'cora':
            self._dataset = Planetoid(root=root, name='Cora', **kwargs)
        elif self.name == "proteins":
            # 1. Load the full collection of 1113 graphs
            full_tu_dataset = TUDataset(root=root, name='PROTEINS', **kwargs)
            
            # 2. Convert the list of graphs into one giant disconnected graph
            # This is a trick to make graph classification datasets work in node-style code
            from torch_geometric.data import Batch
            giant_graph = Batch.from_data_list(full_tu_dataset)
            
            # 3. Wrap it so the rest of the code thinks it's a single-graph dataset
            self._dataset = full_tu_dataset # keep reference
            self._dataset.data = giant_graph
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
        elif self.name == 'chameleon' or self.name == 'squirrel':
             self._dataset = WikipediaNetwork(root=root, name=self.name, **kwargs)
        
        elif self.name == 'wikics':
            self._dataset = WikiCS(root=root, **kwargs)
        elif self.name in ['amazon_computers', 'amazon_photo']:
            from torch_geometric.utils import subgraph
            subset = 'Computers' if 'computers' in self.name else 'Photo'
            

            full_dataset = self._dataset = Amazon(root=root, name=subset, **kwargs)
            data = full_dataset[0]

            # 1. Manually pick a range of nodes (e.g., the first 5000)
            # This acts like a "cluster" without needing METIS/torch-sparse
            num_to_keep = 5000 
            subset_indices = torch.arange(num_to_keep)
            
            edge_index, edge_attr = subgraph(
                subset_indices, 
                data.edge_index, 
                edge_attr=data.edge_attr, 
                relabel_nodes=True, 
                num_nodes=data.num_nodes
            )

            # 3. Create a new Data object with the sliced features
            new_data = Data(
                x=data.x[subset_indices],
                y=data.y[subset_indices],
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # Overwrite the dataset data
            full_dataset.data = new_data
            self._dataset = full_dataset
            
        elif self.name == 'roman-empire':
            from torch_geometric.utils import subgraph

            full_dataset = HeterophilousGraphDataset(root=root, name='Roman-empire', **kwargs)
            data = full_dataset[0]

            # 1. Manually pick a range of nodes (e.g., the first 5000)
            # This acts like a "cluster" without needing METIS/torch-sparse
            num_to_keep = 5000 
            subset_indices = torch.arange(num_to_keep)
            
            edge_index, edge_attr = subgraph(
                subset_indices, 
                data.edge_index, 
                edge_attr=data.edge_attr, 
                relabel_nodes=True, 
                num_nodes=data.num_nodes
            )

            # 3. Create a new Data object with the sliced features
            new_data = Data(
                x=data.x[subset_indices],
                y=data.y[subset_indices],
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # Overwrite the dataset data
            full_dataset.data = new_data
            self._dataset = full_dataset
        elif self.name == 'ogbn-arxiv':
            self._dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
            self._dataset.data.y = self._dataset.data.y.squeeze()
        elif self.name == 'ogbn-proteins':
            # ... (your existing NeighborLoader logic for ogbn-proteins)
            pass 
        else:
            raise NotImplementedError(f'Dataset "{name}" not implemented.')

        # Trigger random split for datasets that don't have standard fixed splits
        # Added 'wikics', 'amazon_computers', 'amazon_photo', 'roman-empire'
        needs_split = ['cornell', 'texas', 'wisconsin', 'actor', 'chameleon', 
                       'squirrel', 'wikics', 'amazon_computers', 'amazon_photo', 'roman-empire']
        
        if self.name in needs_split:
            _data = self._dataset.get(0)
            train_mask, val_mask, test_mask = self.random_split(
                num_train_per_class=kwargs.get('num_train_per_class', 20),
                num_development=int(0.8 * _data.num_nodes),
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
    num_train_per_class: int,
    num_development: int,
    seed: Optional[int] = None
):
        rnd_state = np.random.RandomState(development_seed)
    
        # Auto-detect: single large graph = node classification
        #              multiple graphs    = graph classification
        is_node_classification = len(self) == 1
    
        if is_node_classification:
            # Split over nodes of the single graph
            data = self.get(0)
            num_items = data.num_nodes
        else:
            # Split over graph indices
            num_items = len(self)
    
        development_idx = rnd_state.choice(num_items, num_development, replace=False)
        test_idx = [i for i in range(num_items) if i not in development_idx]
    
        train_idx = []
        rnd_state = np.random.RandomState(seed)
        if is_node_classification:
            # Sample num_train_per_class nodes per class from development set
            data = self.get(0)
            for c in range(self.num_classes):
                # Find indices belonging to class 'c' within the development set
                class_idx = development_idx[
                    np.where(data.y[development_idx].cpu() == c)[0]
                ]

                # --- ROBUST SAMPLING LOGIC ---
                # Determine how many nodes we can actually take
                num_to_sample = min(len(class_idx), num_train_per_class)

                if num_to_sample > 0:
                    train_idx.extend(
                        rnd_state.choice(class_idx, num_to_sample, replace=False)
                    )
                else:
                    # This handles the case where a class might have 0 nodes in your sample
                    print(f"Warning: Class {c} has no nodes in the development set.")
        else:
            # Sample 60% of development set for training
            num_train = int(0.6 * num_development)
            train_idx.extend(
                rnd_state.choice(development_idx, num_train, replace=False)
            )
    
        val_idx = [i for i in development_idx if i not in train_idx]
    
        train_mask = torch.full((num_items,), False)
        train_mask[train_idx] = True
        val_mask = torch.full((num_items,), False)
        val_mask[val_idx] = True
        test_mask = torch.full((num_items,), False)
        test_mask[test_idx] = True
    
        return train_mask, val_mask, test_mask