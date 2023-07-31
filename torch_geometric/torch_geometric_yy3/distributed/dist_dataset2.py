
import torch

from typing import Dict, List, Optional, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed.local_dataset import LocalDataset
from torch_geometric.distributed.local_graph_store import LocalGraphStore as Graph
from torch_geometric.distributed.local_feature_store import LocalFeatureStore as Feature

from torch_geometric.distributed.partition.base import load_partition
from torch_geometric.typing import (
  NodeType, EdgeType, TensorDataType,
)




class DistDataset():
    r""" distributed Graph and feature data for each partiton which is loaded from
    partition files and further initialized by graphstore/featurestore format.
    """
    def __init__(
        self,
        num_partitions: int = 1,
        partition_idx: int = 0,
        graph_partition: Union[Graph, Dict[EdgeType, Graph]] = None,
        node_feat_partition: Union[Feature, Dict[NodeType, Feature]] = None,
        edge_feat_partition: Union[Feature, Dict[EdgeType, Feature]] = None,
        node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
        node_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor]] = None,
        edge_pb: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]] = None,
        node_feat_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor]] = None,
        edge_feat_pb: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]] = None,
    ):
        self.graph = graph_partition
        self.node_features = node_feat_partition
        self.edge_features = edge_feat_partition
        self.node_labels = node_labels
        
        self.meta = None
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        
        self.node_pb = node_pb
        self.edge_pb = edge_pb
        
        self._node_feat_pb = node_feat_pb
        self._edge_feat_pb = edge_feat_pb
        self.data = None
        
    
    def load(
        self,
        root_dir: str,
        partition_idx: int,
        node_label_file: Union[str, Dict[NodeType, str]] = None,
        partition_format:  str = "pyg",
        keep_pyg_data:  bool = True
    ):
        r""" Load one dataset partition from partitioned files.
        
        Args:
            root_dir (str): The file path to load the partition data.
            partition_idx (int): Current partition idx.
            node_label_file (str): The path to the node labels
            partition_format:  pyg/dgl/glt
            keep_pyg_data:  keep the original pyg data besides graphstore/featurestore.
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
   
            if keep_pyg_data:
                # This will also generate the PyG Data format from Store format for back compatibility
                # besides the graphstore/featurestore format.
                
                if(self.meta["hetero_graph"]):
                    # heterogeneous.
        
                    data = HeteroData()

                    r"""
                    edge_attrs=graph_data.get_all_edge_attrs()
                    print(f"---------------edge_attrs={edge_attrs}----")

                    ntypes = set()
                    etypes = []

                    for attrs in edge_attrs:
                        print(f"-------- attrs={attrs}    ")
                        ntypes.add(attrs.edge_type[0])
                        ntypes.add(attrs.edge_type[2])
                        etypes.append(attrs.edge_type)

                    print(f"----------- ntypes={ntypes} ")
                    print(f"----------- etypes={etypes} ")
                    #self._node_types = list(ntypes)
                    """

    
                    edge_index={}
                    edge_ids={}
                    for item in edge_attrs:
                        edge_index[item.edge_type] = graph_data.get_edge_index(item)
                        edge_ids[item.edge_type] = graph_data.get_edge_id(item)
                        data[item.edge_type].edge_index = edge_index[item.edge_type]

                    if node_feat_data is not None:
                        tensor_attrs = node_feat_data.get_all_tensor_attrs()
                        print(f"---------------tensor_attrs={tensor_attrs}----")
                        node_feat={}
                        node_ids={}
                        node_id2index={}
                        for item in tensor_attrs:
                            node_feat[item.attr_name] = node_feat_data.get_tensor(item.fully_specify())
                            node_ids[item.attr_name] = node_feat_data.get_global_id(item.group_name) #, item.attr_name)
                            data[item.attr_name].x = node_feat[item.attr_name]
        
                    if edge_feat_data is not None:
                        edge_attrs=edge_feat_data.get_all_edge_attrs()
                        edge_feat={}
                        edge_ids={}
                        edge_id2index={}
                        for item in edge_attrs:
                            edge_feat[item.edge_type] = edge_feat_data.get_tensor(item.fully_specify())
                            edge_ids[item.edge_type] = edge_feat_data.get_global_id(item.group_name) #, item.attr_name)
                            data[item.edge_type].edge_attr = edge_feat[item.edge_type]

                    self.data = data        
        
                else:
                    # homogeneous.
                    
                    edge_attrs=graph_data.get_all_edge_attrs()

                    print(f"---------------edge_attrs={edge_attrs}----")
                    for item in edge_attrs:
                        edge_index = graph_data.get_edge_index(item)
                        edge_ids = graph_data.get_edge_id(item)
        
                    if node_feat_data is not None:
                        tensor_attrs = node_feat_data.get_all_tensor_attrs()
                        print(f"---------------tensor_attrs={tensor_attrs}----")
                        for item in tensor_attrs:
                            node_feat = node_feat_data.get_tensor(item.fully_specify())
                            node_ids = node_feat_data.get_global_id(item.group_name) #, item.attr_name)
        
                    if edge_feat_data is not None:
                        tensor_attrs = edge_feat_data.get_all_tensor_attrs()
                        for item in tensor_attrs:
                            edge_feat = edge_feat_data.get_tensor(item.fully_specify())
                            edge_ids = edge_feat_data.get_global_id(item.group_name, item.attr_name)
        
                    self.data = Data(x=node_feat, edge_index=edge_index, num_nodes=node_feat.size(0))
        
            # init graph/node feature/edge feature by graphstore/featurestore
            self.graph = graph_data  
        
            # load node feature partition
            if node_feat_data is not None:
                self._node_feat_pb = self.node_pb
                self.node_features = node_feat_data
        
            # load edge feature partition
            if edge_feat_data is not None:
                self._edge_feat_pb = self.edge_pb
                self.edge_features = edge_feat_data
        
        else:  
            # including other partition format, like dgl/glt ..
            # use LocalGraphStore.from_data() and LocalFeatureStore.from_data() api for initialization ..
            pass
        
        # init for labels
        r"""
        if node_label_file is not None:
            if isinstance(node_label_file, dict):
                whole_node_labels = {}
                for ntype, file in node_label_file.items():
                    whole_node_labels[ntype] = torch.load(file)
            else:
                whole_node_labels = torch.load(node_label_file)
            self.node_labels = whole_node_labels
        """

    
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

    def get_node_types(self):
        #if isinstance(self.graph, dict):
        if(self.meta["hetero_graph"]):
            edge_attrs=self.graph.get_all_edge_attrs()
            print(f"---------------edge_attrs={edge_attrs}----")

            if not hasattr(self, '_node_types'):
                ntypes = set()
                for etype in self.graph.keys():
                    ntypes.add(etype[0])
                    ntypes.add(etype[2])
                self._node_types = list(ntypes)
            return self._node_types
        return None

    def get_edge_types(self):
        #if isinstance(self.graph, dict):
        if(self.meta["hetero_graph"]):
            if not hasattr(self, '_edge_types'):
                self._edge_types = list(self.graph.keys())
            return self._edge_types
        return None

    def get_node_label(self, ntype: Optional[NodeType] = None):
        if isinstance(self.node_labels, torch.Tensor):
            return self.node_labels
        if isinstance(self.node_labels, dict):
            assert ntype is not None
            return self.node_labels.get(ntype, None)
        return None

