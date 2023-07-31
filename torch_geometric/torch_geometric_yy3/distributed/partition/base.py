
import torch
import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from torch_geometric.distributed.local_graph_store import LocalGraphStore as Graph
from torch_geometric.distributed.local_feature_store import LocalFeatureStore as Feature

from torch_geometric.typing import as_str

def load_partition(
    root_dir: str,
    partition_idx: int,
    device: torch.device = torch.device('cpu')
) -> Tuple[Dict, int, int,
    Graph,
    Optional[Feature],
    Optional[Feature],
    torch.Tensor,
    torch.Tensor]:


    # load the partition with PyG format (graphstore/featurestore)
    
    with open(os.path.join(root_dir, 'META.json'), 'rb') as infile:
        meta = json.load(infile)
    num_partitions = meta['num_parts']
    assert partition_idx >= 0
    assert partition_idx < num_partitions
    partition_dir = os.path.join(root_dir, f'part_{partition_idx}')
    assert os.path.exists(partition_dir)
    
    graph_dir = os.path.join(partition_dir, 'graph.pt')
    node_feat_dir = os.path.join(partition_dir, 'node_feats.pt')
    edge_feat_dir = os.path.join(partition_dir, 'edge_feats.pt')
    
    #if meta['hetero_graph']==False:
    if meta['is_hetero']==False:
        if os.path.exists(graph_dir):  
            graph = torch.load(graph_dir)  
        if os.path.exists(node_feat_dir):
            node_feat = torch.load(node_feat_dir)
        if os.path.exists(edge_feat_dir):
            edge_feat = torch.load(edge_feat_dir)
        else:
            edge_feat = None
        node_pb = torch.load(os.path.join(root_dir, 'node_map.pt'), map_location=device)
        edge_pb = torch.load(os.path.join(root_dir, 'edge_map.pt'), map_location=device)
        
        return (
            meta, num_partitions, partition_idx,
            graph, node_feat, edge_feat, node_pb, edge_pb
        )
    else:
    
        print("---------- load partition -------")        
        graph_store = torch.load(graph_dir)
        
        if os.path.exists(node_feat_dir):
            node_feat_store = torch.load(node_feat_dir)
        else:
            node_feat_store = None
        
        if os.path.exists(edge_feat_dir):
            edge_feat_store = torch.load(edge_feat_dir)
        else:
            edge_feat_store = None
        
        node_pb_dict = {}
        node_pb_dir = os.path.join(root_dir, 'node_map')
        for ntype in meta['node_types']:
            node_pb_dict[ntype] = torch.load(
                os.path.join(node_pb_dir, f'{as_str(ntype)}.pt'), map_location=device)
        
        edge_pb_dict = {}
        edge_pb_dir = os.path.join(root_dir, 'edge_map')
        for etype in meta['edge_types']:
            edge_pb_dict[tuple(etype)] = torch.load(
                os.path.join(edge_pb_dir, f'{as_str(etype)}.pt'), map_location=device)
        
        return (
            meta, num_partitions, partition_idx,
            graph_store, node_feat_store, edge_feat_store, node_pb_dict, edge_pb_dict
        )      
    
