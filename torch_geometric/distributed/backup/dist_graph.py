import torch
from typing import Dict, Optional, Union
from torch_geometric.distributed import LocalGraphStore as Graph
from torch_geometric.typing import NodeType, EdgeType


class DistGraph(object):
    r"""  Distributed Graph with partition information
    Args:
        num_partitions: Number of partitions.
        partition_id: partition idx for current process.
        local_graph: local Graph data based on LocalGraphStore format
        node_pb: node partition book between node ids and partition ids.
        edge_pb: edge partition book between edge ids and partition ids..
    """
    def __init__(self,
            num_partitions: int,
            partition_idx: int,
            local_graph: Graph,
            node_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor]],
            edge_pb: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
            meta: Optional[Dict] = None):

        self.meta = meta        
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        self.local_graph = local_graph
        self.node_pb = node_pb
        self.edge_pb = edge_pb

    def get_partition_ids_from_nids(self, ids: torch.Tensor,
                            ntype: Optional[NodeType]=None):
        # Get the local partition ids of node ids with a specific node type.
        
        if(self.meta["is_hetero"]):
            assert ntype is not None
            return self.node_pb[ntype][ids]
        return self.node_pb[ids]

    def get_partition_ids_from_eids(self, eids: torch.Tensor,
                            etype: Optional[EdgeType]=None):
        # Get the partition ids of edge ids with a specific edge type.
        
        if(self.meta["is_hetero"]):
            assert etype is not None
            return self.edge_pb[etype][eids]
        return self.edge_pb[eids]

