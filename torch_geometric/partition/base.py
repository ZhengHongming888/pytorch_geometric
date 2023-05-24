
import torch
import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


from torch_geometric.typing import (
  as_str, 
  GraphPartitionData, HeteroGraphPartitionData,
  FeaturePartitionData, HeteroFeaturePartitionData,
  PartitionBook, HeteroNodePartitionDict, HeteroEdgePartitionDict
)
from torch_geometric.utils import convert_to_tensor, ensure_dir, id2idx



def _load_graph_partition_data(
  graph_data_dir: str,
  device: torch.device
) -> GraphPartitionData:
  r""" Load graph partition data from defined folder.
  """
  if not os.path.exists(graph_data_dir):
    return None
  rows = torch.load(os.path.join(graph_data_dir, 'rows.pt'),
                    map_location=device)
  cols = torch.load(os.path.join(graph_data_dir, 'cols.pt'),
                    map_location=device)
  eids = torch.load(os.path.join(graph_data_dir, 'eids.pt'),
                    map_location=device)
  pdata = GraphPartitionData(edge_index=(rows, cols), eids=eids)
  return pdata


def _load_feature_partition_data(
  feature_data_dir: str,
  device: torch.device
) -> FeaturePartitionData:
  r""" Load a feature partition data from defined folder.
  """
  if not os.path.exists(feature_data_dir):
    return None
  feats = torch.load(os.path.join(feature_data_dir, 'feats.pt'),
                     map_location=device)
  ids = torch.load(os.path.join(feature_data_dir, 'ids.pt'),
                   map_location=device)
  cache_feats_path = os.path.join(feature_data_dir, 'cache_feats.pt')
  cache_ids_path = os.path.join(feature_data_dir, 'cache_ids.pt')
  cache_feats = None
  cache_ids = None
  if os.path.exists(cache_feats_path) and os.path.exists(cache_ids_path):
    cache_feats = torch.load(cache_feats_path, map_location=device)
    cache_ids = torch.load(cache_ids_path, map_location=device)
  pdata = FeaturePartitionData(
    feats=feats, ids=ids, cache_feats=cache_feats, cache_ids=cache_ids
  )
  return pdata


def load_partition(
  root_dir: str,
  partition_idx: int,
  device: torch.device = torch.device('cpu')
) -> Union[Tuple[Dict, int, int,
                 GraphPartitionData,
                 Optional[FeaturePartitionData],
                 Optional[FeaturePartitionData],
                 PartitionBook,
                 PartitionBook],
           Tuple[Dict, int, int,
                 HeteroGraphPartitionData,
                 Optional[HeteroFeaturePartitionData],
                 Optional[HeteroFeaturePartitionData],
                 HeteroNodePartitionDict,
                 HeteroEdgePartitionDict]]:


  # load the partition with PyG format (graphstore/featurestore)

  with open(os.path.join(root_dir, 'META.json'), 'rb') as infile:
    meta = json.load(infile)
  num_partitions = meta['num_parts']
  assert partition_idx >= 0
  assert partition_idx < num_partitions
  partition_dir = os.path.join(root_dir, f'part_{partition_idx}')
  print(partition_dir)
  assert os.path.exists(partition_dir)

  graph_dir = os.path.join(partition_dir, 'graph.pt')
  node_feat_dir = os.path.join(partition_dir, 'node_feats.pt')
  edge_feat_dir = os.path.join(partition_dir, 'edge_feats.pt')

  if meta['hetero_graph']==False:
    if os.path.exists(graph_dir):  
        graph = torch.load(graph_dir)  #_load_graph_partition_data(graph_dir, device)
    if os.path.exists(node_feat_dir):
        node_feat = torch.load(node_feat_dir)  #_load_feature_partition_data(node_feat_dir, device)
    if os.path.exists(edge_feat_dir):
        edge_feat = torch.load(edge_feat_dir)  #_load_feature_partition_data(edge_feat_dir, device)
    else:
        edge_feat = None
    node_pb = torch.load(os.path.join(root_dir, 'node_map.pt'), map_location=device)
    edge_pb = torch.load(os.path.join(root_dir, 'edge_map.pt'), map_location=device)
    
    return (
      meta, num_partitions, partition_idx,
      graph, node_feat, edge_feat, node_pb, edge_pb
    )
  else:

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


def load_partition_glt(
  root_dir: str,
  partition_idx: int,
  device: torch.device = torch.device('cpu')
) -> Union[Tuple[int, int,
                 GraphPartitionData,
                 Optional[FeaturePartitionData],
                 Optional[FeaturePartitionData],
                 PartitionBook,
                 PartitionBook],
           Tuple[int, int,
                 HeteroGraphPartitionData,
                 Optional[HeteroFeaturePartitionData],
                 Optional[HeteroFeaturePartitionData],
                 HeteroNodePartitionDict,
                 HeteroEdgePartitionDict]]:
  r""" Load the partition with glt(graphstore/featurestore) format.

  Args:
    root_dir (str): The root directory for saved files.
    partition_idx (int): The partition idx to load.
    device (torch.device): The device where loaded graph partition data locates.

  """
  with open(os.path.join(root_dir, 'META'), 'rb') as infile:
    meta = pickle.load(infile)
  num_partitions = meta['num_parts']
  assert partition_idx >= 0
  assert partition_idx < num_partitions
  partition_dir = os.path.join(root_dir, f'part{partition_idx}')
  assert os.path.exists(partition_dir)

  graph_dir = os.path.join(partition_dir, 'graph')
  node_feat_dir = os.path.join(partition_dir, 'node_feat')
  edge_feat_dir = os.path.join(partition_dir, 'edge_feat')

  # homogenous

  if meta['data_cls'] == 'homo':
    graph = _load_graph_partition_data(graph_dir, device)
    node_feat = _load_feature_partition_data(node_feat_dir, device)
    edge_feat = _load_feature_partition_data(edge_feat_dir, device)
    node_pb = torch.load(os.path.join(root_dir, 'node_pb.pt'),
                         map_location=device)
    edge_pb = torch.load(os.path.join(root_dir, 'edge_pb.pt'),
                         map_location=device)
    return (
      num_partitions, partition_idx,
      graph, node_feat, edge_feat, node_pb, edge_pb
    )

  # heterogenous

  graph_dict = {}
  for etype in meta['edge_types']:
    graph_dict[etype] = _load_graph_partition_data(
      os.path.join(graph_dir, as_str(etype)), device)

  node_feat_dict = {}
  for ntype in meta['node_types']:
    node_feat = _load_feature_partition_data(
      os.path.join(node_feat_dir, as_str(ntype)), device)
    if node_feat is not None:
      node_feat_dict[ntype] = node_feat
  if len(node_feat_dict) == 0:
    node_feat_dict = None

  edge_feat_dict = {}
  for etype in meta['edge_types']:
    edge_feat = _load_feature_partition_data(
      os.path.join(edge_feat_dir, as_str(etype)), device)
    if edge_feat is not None:
      edge_feat_dict[etype] = edge_feat
  if len(edge_feat_dict) == 0:
    edge_feat_dict = None

  node_pb_dict = {}
  node_pb_dir = os.path.join(root_dir, 'node_pb')
  for ntype in meta['node_types']:
    node_pb_dict[ntype] = torch.load(
      os.path.join(node_pb_dir, f'{as_str(ntype)}.pt'), map_location=device)

  edge_pb_dict = {}
  edge_pb_dir = os.path.join(root_dir, 'edge_pb')
  for etype in meta['edge_types']:
    edge_pb_dict[etype] = torch.load(
      os.path.join(edge_pb_dir, f'{as_str(etype)}.pt'), map_location=device)

  return (
    num_partitions, partition_idx,
    graph_dict, node_feat_dict, edge_feat_dict, node_pb_dict, edge_pb_dict
  )


def _cat_feature_cache(partition_idx, raw_feat_data, raw_feat_pb):
  r""" Cat a feature partition with its cached features.
  """
  if isinstance(raw_feat_data, dict):
    # heterogeneous.
    cache_ratio, feat_data, feat_ids, feat_id2idx, feat_pb = {}, {}, {}, {}
    for graph_type, raw_feat in raw_feat_data.items():
      cache_ratio[graph_type], feat_data[graph_type], \
      feat_ids[graph_type], feat_id2idx[graph_type], feat_pb[graph_type] = \
        cat_feature_cache(partition_idx, raw_feat, raw_feat_pb[graph_type])
  else:
    # homogeneous.
    cache_ratio, feat_data, feat_ids, feat_id2idx, feat_pb = \
      cat_feature_cache(partition_idx, raw_feat_data, raw_feat_pb)
  return cache_ratio, feat_data, feat_ids, feat_id2idx, feat_pb

def cat_feature_cache(
  partition_idx: int,
  feat_pdata: FeaturePartitionData,
  feat_pb: PartitionBook
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, PartitionBook]:
  r""" Concatenate and deduplicate partitioned features and its cached
  features into a new feature patition.

  Returns:
    float: The proportion of cache features.
    torch.Tensor: The new feature tensor, where the cached feature data is
      arranged before the original partition data.
    torch.Tensor: The tensor that indicates the mapping from global node id
      to its local index in new features.
    PartitionBook: The modified partition book for the new feature tensor.
  """
  feats = feat_pdata.feats
  ids = feat_pdata.ids
  cache_feats = feat_pdata.cache_feats
  cache_ids = feat_pdata.cache_ids
  if cache_feats is None or cache_ids is None:
    return 0.0, feats, ids, id2idx(ids), feat_pb
