import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import Linear, HGTConv

from torch_geometric.data import Data, LocalDataset

def test_local_dataset():

    path = osp.join(osp.dirname(osp.realpath(__file__)), './ogbn-mags')
    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    # init edge_index and node_features.
    edge_dict, feature_dict = {}, {}
    for etype in data.edge_types:
      edge_dict[etype] = data[etype]['edge_index']
    for ntype in data.node_types:
      feature_dict[ntype] = data[ntype].x.clone(memory_format=torch.contiguous_format)
  
    # define local_dataset and init_graph() and init_node_features()
    pyg_dataset = LocalDataset()
    pyg_dataset.init_graph(
      edge_index=edge_dict
    )
    pyg_dataset.init_node_features(
      node_feature_data=feature_dict
    )

    # get the graph and node_features from the graphstore/featurestore in LocalDataset
    edge_attrs=pyg_dataset.graph.get_all_edge_attrs()
    edge_index={}
    edge_ids={}
    for item in edge_attrs:
        edge_index[item.edge_type] = pyg_dataset.graph.get_edge_index(item)
        edge_ids[item.edge_type] = pyg_dataset.graph.get_edge_ids(item)


    #verify the edge_index
    assert (edge_index==edge_dict)

    tensor_attrs = pyg_dataset.node_features.get_all_tensor_attrs()
    node_feat={}
    node_ids={}
    node_id2index={}
    for item in tensor_attrs:
        node_feat[item.attr_name] = pyg_dataset.node_features.get_tensor(item.fully_specify())
        node_ids[item.attr_name] = pyg_dataset.node_features.get_global_ids(item.group_name, item.attr_name)
        node_id2index[item.attr_name] = pyg_dataset.node_features.id2index

    # verfiy the node features.
    assert (node_feat==feature_dict)

