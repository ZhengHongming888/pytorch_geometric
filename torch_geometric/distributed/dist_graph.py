

import torch

from typing import Dict, Optional, Union
from torch_geometric.distributed import LocalGraphStore as Graph

from torch_geometric.typing import (
  NodeType, EdgeType, PartitionBook,
  HeteroNodePartitionDict, HeteroEdgePartitionDict
)


class DistGraph(object):
  r""" 
  Distributed Graph with partition information

  Args:
    num_partitions: Number of partitions.
    partition_id: partition idx for current process.
    local_graph: local Graph data.
    node_pb: node partition book between node ids and partition ids.
    edge_pb: edge partition book between edge ids and partition ids..

  """
  def __init__(self,
               num_partitions: int,
               partition_idx: int,
               local_graph: Union[Graph, Dict[EdgeType, Graph]],
               node_pb: Union[PartitionBook, HeteroNodePartitionDict],
               edge_pb: Union[PartitionBook, HeteroEdgePartitionDict]):
    
    self.num_partitions = num_partitions
    self.partition_idx = partition_idx
    self.local_graph = local_graph
    
    if isinstance(self.local_graph, dict):
      self.data_cls = 'hetero'
    elif isinstance(self.local_graph, Graph):
      self.data_cls = 'homo'
    else:
      raise ValueError("found invalid input with mismatched graph type")

    self.node_pb = node_pb
    if self.node_pb is not None:
      if isinstance(self.node_pb, dict):
        assert self.data_cls == 'hetero'
      elif isinstance(self.node_pb, PartitionBook):
        assert self.data_cls == 'homo'
      else:
        raise ValueError("found invalid input with mismatched graph type")

    self.edge_pb = edge_pb
    if self.edge_pb is not None:
      if isinstance(self.edge_pb, dict):
        assert self.data_cls == 'hetero'
      elif isinstance(self.edge_pb, PartitionBook):
        assert self.data_cls == 'homo'
      else:
        raise ValueError("found invalid input with mismatched graph type")

  def get_local_graph(self, etype: Optional[EdgeType]=None):
    # Get the local graph object by edge type.
    
    if self.data_cls == 'hetero':
      assert etype is not None
      return self.local_graph[etype]
    return self.local_graph

  def get_node_partitions(self, ids: torch.Tensor,
                          ntype: Optional[NodeType]=None):
    # Get the local partition ids of node ids with a specific node type.
    
    if self.data_cls == 'hetero':
      assert ntype is not None
      return self.node_pb[ntype][ids]
    return self.node_pb[ids]

  def get_edge_partitions(self, eids: torch.Tensor,
                          etype: Optional[EdgeType]=None):
    # Get the partition ids of edge ids with a specific edge type.
    
    if self.data_cls == 'hetero':
      assert etype is not None
      return self.edge_pb[etype][eids]
    return self.edge_pb[eids]

