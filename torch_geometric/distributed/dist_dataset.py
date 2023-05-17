
import torch

from typing import Dict, List, Optional, Union

from torch_geometric.data import Data, LocalDataset
from torch_geometric.data import LocalGraphStore as Graph
from torch_geometric.data import LocalFeatureStore as Feature

from torch_geometric.partition import load_partition, cat_feature_cache
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
    graph_mode: str = 'ZERO_COPY',
    feature_with_gpu: bool = True,
    whole_node_label_file: Union[str, Dict[NodeType, str]] = None,
    device: Optional[int] = None
  ):
    r""" Load a certain dataset partition from partitioned files and create
    in-memory objects (``Graph``, ``Feature`` or ``torch.Tensor``).

    Args:
      root_dir (str): The directory path to load the graph and feature
        partition data.
      partition_idx (int): Partition idx to load.
      graph_mode (str): Mode for creating graphlearn_torch's `Graph`, including
        'CPU', 'ZERO_COPY' or 'CUDA'. (default: 'ZERO_COPY')
      feature_with_gpu (bool): A Boolean value indicating whether the created
        ``Feature`` objects of node/edge features use ``UnifiedTensor``.
        If True, it means ``Feature`` consists of ``UnifiedTensor``, otherwise
        ``Feature`` is a PyTorch CPU Tensor, the ``device_group_list`` and
        ``device`` will be invliad. (default: ``True``)
      device_group_list (List[DeviceGroup], optional): A list of device groups
        used for feature lookups, the GPU part of feature data will be
        replicated on each device group in this list during the initialization.
        GPUs with peer-to-peer access to each other should be set in the same
        device group properly.  (default: ``None``)
      whole_node_label_file (str): The path to the whole node labels which are
        not partitioned. (default: ``None``)
      device: The target cuda device rank used for graph operations when graph
        mode is not "CPU" and feature lookups when the GPU part is not None.
        (default: ``None``)
    """
    (
      self.num_partitions,
      self.partition_idx,
      graph_data,
      node_feat_data,
      edge_feat_data,
      self.node_pb,
      self.edge_pb
    ) = load_partition(root_dir, partition_idx)

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

    print(f"----------- edge_index={edge_index}----------")

    self.data = Data(x=node_feat_data.feats, edge_index=edge_index, num_nodes=node_feat_data.feats.size(0))

    print(f"-------------- self.data ={self.data}--------------")
    print(f"-------------- self.data.edge_index ={self.data.edge_index}--------------")
    print(f"edge_index={edge_index}, edge_ids={edge_ids}, num_partition={self.num_partitions}, self.partition_idx={self.partition_idx}, graph_data={graph_data}, node_feat_data={node_feat_data}, edge_feat_data={edge_feat_data}, self.node_pb={self.node_pb}, self.edge_pb={self.edge_pb}")
    print(f"--------- edge_index_len ={len(edge_index[0])}, edge_ids_size={edge_ids.size()}")
    print(f"--------- node_feat_data.feats.size={node_feat_data.feats.size()}, node_feat_data_ids_len={len(node_feat_data.ids)} ")
    print(f"--------- node_pb_sum ={self.node_pb.sum()}, node_pb_len ={len(self.node_pb)}, edge_pb_sum ={self.edge_pb.sum()}, edge_pb_len ={len(self.edge_pb)}")
    

    # init graph/node feature/edge feature by graphstore/featurestore
    self.init_graph(edge_index, edge_ids, layout='COO')

    if node_feat_data is not None:

      node_feat = node_feat_data.feats
      node_feat_id2idx = id2idx(node_feat_data.ids)
      node_feat_pb = self.node_pb
      
      self.init_node_features(node_feat, node_feat_id2idx, self.partition_idx, dtype=None)
      self._node_feat_pb = node_feat_pb

    # load edge feature partition
    if edge_feat_data is not None:

      edge_feat = edge_feat_data.feats
      edge_feat_id2idx = id2idx(edge_feat_data.ids)
      edge_feat_pb = self.edge_pb

      self.init_edge_features(edge_feat, edge_feat_id2idx, self.partition_idx, type=None)
      self._edge_feat_pb = edge_feat_pb

    print(f"======== init_node_features node_feat_data={node_feat_data}, node_feat_pb={node_feat_pb}, node_feat_id2idx={node_feat_id2idx} ")
    print(f"-------- self._node_feat_pb={self._node_feat_pb}, self._edge_feat_pb={self._edge_feat_pb}")
    print(f"-------- self.node_features={self.node_features}, self.edge_features={self.edge_features}")


    # init for labels
    if whole_node_label_file is not None:
      if isinstance(whole_node_label_file, dict):
        whole_node_labels = {}
        for ntype, file in whole_node_label_file.items():
          whole_node_labels[ntype] = torch.load(file)
      else:
        whole_node_labels = torch.load(whole_node_label_file)
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

