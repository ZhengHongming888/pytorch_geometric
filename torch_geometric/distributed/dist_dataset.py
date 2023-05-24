
import torch

from typing import Dict, List, Optional, Union

from torch_geometric.data import Data, LocalDataset
from torch_geometric.data import LocalGraphStore as Graph
from torch_geometric.data import LocalFeatureStore as Feature

from torch_geometric.partition import load_partition, load_partition_glt, cat_feature_cache, _cat_feature_cache
from torch_geometric.utils import share_memory, id2idx
from torch_geometric.typing import (
  NodeType, EdgeType, TensorDataType,
  PartitionBook, HeteroNodePartitionDict, HeteroEdgePartitionDict
)




class DistDataset(LocalDataset):
  r""" distributed Graph and feature data for each partiton which is loaded from
  partition files and further initialized by graphstore/featurestore format.
  """
  def __init__(
    self,
    num_partitions: int = 1,
    partition_idx: int = 0,
    graph_partition: Union[Graph, Dict[EdgeType, Graph]] = None,
    node_feature_partition: Union[Feature, Dict[NodeType, Feature]] = None,
    edge_feature_partition: Union[Feature, Dict[EdgeType, Feature]] = None,
    whole_node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
    node_pb: Union[PartitionBook, HeteroNodePartitionDict] = None,
    edge_pb: Union[PartitionBook, HeteroEdgePartitionDict] = None,
    node_feat_pb: Union[PartitionBook, HeteroNodePartitionDict] = None,
    edge_feat_pb: Union[PartitionBook, HeteroEdgePartitionDict] = None,
  ):
    super().__init__(
      graph_partition,
      node_feature_partition,
      edge_feature_partition,
      whole_node_labels
    )

    self.meta = None
    self.num_partitions = num_partitions
    self.partition_idx = partition_idx

    self.node_pb = node_pb
    self.edge_pb = edge_pb

    self._node_feat_pb = node_feat_pb
    self._edge_feat_pb = edge_feat_pb
    self.data = None

    if self.graph is not None:
      assert self.node_pb is not None
    if self.node_features is not None:
      assert self.node_pb is not None or self._node_feat_pb is not None
    if self.edge_features is not None:
      assert self.edge_pb is not None or self._edge_feat_pb is not None

  def load(
    self,
    root_dir: str,
    partition_idx: int,
    node_label_file: Union[str, Dict[NodeType, str]] = None,
    partition_format:  str = "pyg"
  ):
    r""" Load one dataset partition from partitioned files.

    Args:
      root_dir (str): The file path to load the partition data.
      partition_idx (int): Current partition idx.
      whole_node_label_file (str): The path to the whole node labels which are
        not partitioned. (default: ``None``)
    """
    if partition_format=="pyg":
        (
          self.meta,
          self.num_partitions,
          self.partition_idx,
          graph_data,
          node_feat_data,
          edge_feat_data,
          self.node_pb,
          self.edge_pb
        ) = load_partition(root_dir, partition_idx)

        # init graph partition
        if(self.meta["hetero_graph"]):
          # heterogeneous.

          edge_attrs=graph_data.get_all_edge_attrs()
          edge_index={}
          edge_ids={}
          for item in edge_attrs:
            edge_index[item.edge_type] = graph_data.get_edge_index(item)
            edge_ids[item.edge_type] = graph_data.get_edge_ids(item)

          if node_feat_data is not None:
              tensor_attrs = node_feat_data.get_all_tensor_attrs()
              node_feat={}
              node_ids={}
              node_id2index={}
              for item in tensor_attrs:
                  node_feat[item.attr_name] = node_feat_data.get_tensor(item.fully_specify())
                  node_ids[item.attr_name] = node_feat_data.get_global_ids(item.group_name, item.attr_name)
                  node_id2index[item.attr_name] = node_feat_data.id2index

          if edge_feat_data is not None:
              edge_attrs=graph_data.get_all_edge_attrs()
              edge_feat={}
              edge_ids={}
              edge_id2index={}
              for item in edge_attrs:
                  edge_feat[item.edge_type] = edge_feat_data.get_tensor(item.fully_specify())
                  edge_ids[item.edge_type] = edge_feat_data.get_global_ids(item.group_name, item.attr_name)
                  edge_id2index[item.edge_type] = edge_feat_data.id2index
          #self.data = Data(x=node_feat, edge_index=edge_index, num_nodes=node_feat.size(0))


        else:
          # homogeneous.
          
          edge_attrs=graph_data.get_all_edge_attrs()
          for item in edge_attrs:
              edge_index = graph_data.get_edge_index(item)
              edge_ids = graph_data.get_edge_ids(item)

          if node_feat_data is not None:
              tensor_attrs = node_feat_data.get_all_tensor_attrs()
              for item in tensor_attrs:
                  node_feat = node_feat_data.get_tensor(item.fully_specify())
                  node_ids = node_feat_data.get_global_ids(item.group_name, item.attr_name)
                  #node_feat_data.set_id2index(item.group_name, item.attr_name)
                  node_id2index = node_feat_data.id2index

          if edge_feat_data is not None:
              tensor_attrs = edge_feat_data.get_all_tensor_attrs()
              for item in tensor_attrs:
                  edge_feat = edge_feat_data.get_tensor(item.fully_specify())
                  edge_ids = edge_feat_data.get_global_ids(item.group_name, item.attr_name)
                  edge_id2index = edge_feat_data.id2index

          self.data = Data(x=node_feat, edge_index=edge_index, num_nodes=node_feat.size(0))

        # init graph/node feature/edge feature by graphstore/featurestore
        self.graph = graph_data  
  
        # load node feature partition
        if node_feat_data is not None:
          self._node_feat_pb = self.node_pb
          self.node_features = node_feat_data
  
        # load edge feature partition
        if edge_feat_data is not None:
          self._edge_feat_pb = eself.edge_pb
          self.edge_features = edge_feat_data

    # for glt partition format ..
    else:  
        (
          self.num_partitions,
          self.partition_idx,
          graph_data,
          node_feat_data,
          edge_feat_data,
          self.node_pb,
          self.edge_pb
        ) = load_partition_glt(root_dir, partition_idx)
        
        # init graph partition
        if isinstance(graph_data, dict):
          # heterogeneous.
          edge_index, edge_ids = {}, {}
          for k, v in graph_data.items():
            edge_index[k] = v.edge_index
            edge_ids[k] = v.eids
        else:
          # homogeneous.
          edge_index = graph_data.edge_index
          edge_ids = graph_data.eids

        self.init_graph(edge_index, edge_ids, layout='COO')

        # load node feature partition
        if node_feat_data is not None:
          node_feat_pb = self.node_pb
          node_cache_ratio, node_feat, node_ids, node_feat_id2idx, node_feat_pb = \
            _cat_feature_cache(partition_idx, node_feat_data, self.node_pb)
          self.init_node_features(node_feat, node_ids, node_feat_id2idx, self.partition_idx, dtype=None)
          self._node_feat_pb = node_feat_pb

        # load edge feature partition
        if edge_feat_data is not None:
          edge_cache_ratio, edge_feat, edge_ids, edge_feat_id2idx, edge_feat_pb = \
            _cat_feature_cache(partition_idx, edge_feat_data, self.edge_pb)
          self.init_edge_features(edge_feat, edge_ids, edge_feat_id2idx, self.partition_idx, dtype=None)
          self._edge_feat_pb = edge_feat_pb
     
        self.data = Data(x=node_feat, edge_index=edge_index, num_nodes=node_feat.size(0)) 
        
    # init for labels
    if node_label_file is not None:
      if isinstance(node_label_file, dict):
        whole_node_labels = {}
        for ntype, file in node_label_file.items():
          whole_node_labels[ntype] = torch.load(file)
      else:
        whole_node_labels = torch.load(node_label_file)
      self.init_node_labels(whole_node_labels)
  
  @property
  def node_feat_pb(self):
    if self._node_feat_pb is None:
      return self.node_pb
    return self._node_feat_pb

  @property
  def edge_feat_pb(self):
    if self._edge_feat_pb is None:
      return self.edge_pb
    return self._edge_feat_pb

