
import torch

from typing import Dict, List, Optional, Union

from torch_geometric.data import TensorAttr, EdgeAttr
from torch_geometric.distributed.local_graph_store import LocalGraphStore as Graph
from torch_geometric.distributed.local_feature_store import LocalFeatureStore as Feature

from torch_geometric.typing import NodeType, EdgeType, TensorDataType
from torch_geometric.utils import convert_to_tensor, share_memory, squeeze


class LocalDataset(object):
    r""" Local data manager to initialize the graph topology and feature data 
    from each local partition by using the LocalGraphStore/LocalFeatureStore
    """

    def __init__(
        self,
        graph: Union[Graph, Dict[EdgeType, Graph]] = None,
        node_features: Union[Feature, Dict[NodeType, Feature]] = None,
        edge_features: Union[Feature, Dict[EdgeType, Feature]] = None,
        node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
        ):
        self.graph = graph
        self.node_features = node_features
        self.edge_features = edge_features
        self.node_labels = squeeze(convert_to_tensor(node_labels))

    def init_graph(
        self,
        edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        edge_ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        layout: Union[str, Dict[EdgeType, str]] = 'COO',
        directed: bool = False
        ):
        r""" Initialize the graph data storage and build the Graph.
        
        Args:
        edge_index (torch.Tensor or numpy.ndarray): Edge index for graph topo,
        edge_ids (torch.Tensor or numpy.ndarray): Edge ids for graph edges, 
        layout (str): The edge layout representation for the input edge index,
        should be 'COO', 'CSR' or 'CSC'. (default: 'COO')
        
        directed (bool):indicate graph topology is directed or not.(default: ``False``)
        """
        edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
        edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)
        self._directed = directed
        
        #print(isinstance(edge_index, dict))    
        if edge_index is not None:
            if isinstance(edge_index, dict):
                # heterogeneous.
                if edge_ids is not None:
                    assert isinstance(edge_ids, dict)
                else:
                    edge_ids = {}
                if not isinstance(layout, dict):
                    layout = {etype: layout for etype in edge_index.keys()}

                graph_store = Graph() 
                for etype, e_idx in edge_index.items():
                    node_num = e_idx[0].size()
                    graph_store.put_edge_index(
                        edge_index=e_idx,
                        edge_type=etype, layout='coo',
                        size=(node_num, node_num))            
                    graph_store.set_edge_ids(
                        edge_ids=edge_ids,
                        edge_attr=EdgeAttr(edge_type=etype, layout='coo',
                        size=(node_num, node_num)))
                    self.graph = graph_store
            else:
                # homogeneous.
                graph_store = Graph()
                node_num = edge_index[0].size()
                graph_store.put_edge_index(
                    edge_index=edge_index,
                    edge_type=None, layout='coo', size=(node_num, node_num))
                graph_store.set_edge_ids(edge_ids=edge_ids,
    				    edge_attr=EdgeAttr(edge_type=None, layout='coo', size=(node_num, node_num)))
                self.graph = graph_store


    def init_node_features(
        self,
        node_feature_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
        ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        id2idx: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
        partition_idx: int = 1,
        dtype: Optional[torch.dtype] = None
        ):
        r"""
        Initialize the node feature data storage.
        Args:
        node_feature_data (torch.Tensor or numpy.ndarray): raw node feature data,
        id2idx (torch.Tensor or numpy.ndarray): mapping between node id to index
        	(default: ``None``)
        """
        
        if node_feature_data is not None:
            node_feature_data = convert_to_tensor(node_feature_data, dtype)
            id2idx = convert_to_tensor(id2idx)
            self.node_features = create_features(
                node_feature_data, ids, id2idx, partition_idx, "node_feat"
                )
    
    def init_edge_features(
        self,
        edge_feature_data: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        id2idx: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        partition_idx: int = 1,
        dtype: Optional[torch.dtype] = None
        ):
        r""" 
        Initialize the edge feature data storage.
        Args:
        edge_feature_data (torch.Tensor or numpy.ndarray): raw edge feature data, 
        id2idx (torch.Tensor or numpy.ndarray): mapping between edge id to index.
        """    
        
        if edge_feature_data is not None:
            self.edge_features = create_features(
            convert_to_tensor(edge_feature_data, dtype), convert_to_tensor(id2idx), partition_idx,
      		"edge_feat"
        )
    
    def init_node_labels(
        self,
        node_label_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
        ):
        #Initialize the node labels.
        if node_label_data is not None:
    	    self.node_labels = squeeze(convert_to_tensor(node_label_data))
    
    def get_graph(self, etype: Optional[EdgeType] = None):
        if isinstance(self.graph, Graph):
            return self.graph
        if isinstance(self.graph, dict):
            assert etype is not None
            return self.graph.get(etype, None)
        return None

    def get_node_types(self):
        if isinstance(self.graph, dict):
            if not hasattr(self, '_node_types'):
                ntypes = set()
                for etype in self.graph.keys():
                    ntypes.add(etype[0])
                    ntypes.add(etype[2])
                self._node_types = list(ntypes)
            return self._node_types
        return None
    
    def get_edge_types(self):
        if isinstance(self.graph, dict):
            if not hasattr(self, '_edge_types'):
                self._edge_types = list(self.graph.keys())
            return self._edge_types
        return None
    
    def get_node_feature(self, ntype: Optional[NodeType] = None):
        if isinstance(self.node_features, Feature):
            return self.node_features
        if isinstance(self.node_features, dict):
            assert ntype is not None
            return self.node_features.get(ntype, None)
        return None
    
    def get_edge_feature(self, etype: Optional[EdgeType] = None):
        if isinstance(self.edge_features, Feature):
            return self.edge_features
        if isinstance(self.edge_features, dict):
            assert etype is not None
            return self.edge_features.get(etype, None)
        return None
    
    def get_node_label(self, ntype: Optional[NodeType] = None):
        if isinstance(self.node_labels, torch.Tensor):
            return self.node_labels
        if isinstance(self.node_labels, dict):
            assert ntype is not None
            return self.node_labels.get(ntype, None)
        return None
    
    def __getitem__(self, key):
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    
def create_features(feature_data, ids, id2idx, partition_idx, attr_name):
    # Initialize the node/edge feature by FeatureStore.
    if feature_data is not None:    	    
        if isinstance(feature_data, dict):
            # heterogeneous.    
            if id2idx is not None:
                assert isinstance(id2idx, dict)
            else:
                id2idx = {}
            features = Feature()
            for graph_type, feat in feature_data.items():
                features.put_tensor(feat, group_name=f'part_{partition_idx}', attr_name=graph_type, index=None)    
                if ids is not None:
                    features.set_global_ids_plus_id2index(ids[graph_type], group_name=f'part_{partition_idx}', attr_name=graph_type)
        else:
            # homogeneous.    
                features = Feature()
                features.put_tensor(feature_data, group_name=f'part_{partition_idx}', attr_name=None, index=None)
                if ids is not None:
                    features.set_global_ids_plus_id2index(ids, group_name=f'part_{partition_idx}', attr_name=None)
    else:
        features = None

    return features
