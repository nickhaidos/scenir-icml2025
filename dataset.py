import pickle
import torch_geometric
from torch_geometric.utils.convert import from_networkx


class GraphDataset(torch_geometric.data.Dataset):
        def __init__(self, graph_list):
            super().__init__()
            self.dataset = graph_list
            
        def get(self, idx):
            return self.dataset[idx]
            
        def len(self):
            return len(self.dataset)


def get_train_loader(config):

    with open(config['train_dataset_path'], "rb") as f:
        train_dataset = pickle.load(f)

    train_dataset_list = []

    for graph,_ in train_dataset:
        train_dataset_list.append(from_networkx(graph, group_node_attrs=["embedding"], group_edge_attrs=["embedding"]).to(config['device']))
        
    train_graph_dataset = GraphDataset(train_dataset_list)

    train_graph_loader = torch_geometric.loader.DataLoader(train_graph_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    return train_graph_loader

def get_test_dataset(config):
    with open(config['test_dataset_path'], "rb") as f:
        test_dataset = pickle.load(f)

    test_dataset_list = []

    for graph,_ in test_dataset:
        test_dataset_list.append(from_networkx(graph, group_node_attrs=["embedding"], group_edge_attrs=["embedding"]).to(config['device']))

    test_graph_dataset = GraphDataset(test_dataset_list)

    return test_graph_dataset