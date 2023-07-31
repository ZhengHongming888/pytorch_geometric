

import torch
from typing import Dict, List, Optional, Tuple, Union

from torch_geometric.data import TensorAttr
from torch_geometric.distributed import LocalFeatureStore as Feature
from torch_geometric.typing import EdgeType, NodeType

from torch_geometric.utils import get_available_device, ensure_device

from torch_geometric.distributed import (
    RpcRouter, RpcCallBase, rpc_register, rpc_request_async
)



class RpcCallFeatureLookup(RpcCallBase):
    r""" A wrapper for rpc remote call to get the feature data from remote"""

    def __init__(self, dist_feature):
        super().__init__()
        self.dist_feature = dist_feature
    
    def rpc_async(self, *args, **kwargs):
        return self.dist_feature.rpc_local_feature_get(*args, **kwargs)
    
    def rpc_sync(self, *args, **kwargs):
        pass


class DistFeature(object):
    r""" Distributed feature with partition information and also with
    the global feature lookups after node sampling.
    
    Args:
        meta: partition meta information
        num_partitions: Number of partitions.
        partition_id: partition idx for current process.
        local_feature: Local Feature instance.
        feature_pb: feature partition book between feature ids and partition id
        local_only: Use this to switch on/off only local feature lookup or stitching.
    """
    def __init__(self,
            num_partitions: int,
            partition_index: int,
            local_feature: Feature,
            feature_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor],
                                Dict[EdgeType, torch.Tensor]],
            local_only: bool = False,
            rpc_router: Optional[RpcRouter] = None,
            device: Optional[torch.device] = None,
            meta: Optional[Dict] = None,
            ):

        self.meta = meta
        self.num_partitions = num_partitions
        self.partition_idx = partition_index
        
        self.device = torch.device(type='cpu')

        self.local_feature = local_feature
        self.feature_pb = feature_pb
        
        self.rpc_router = rpc_router
        if not local_only:
            if self.rpc_router is None:
                raise ValueError("A rpc router must be provided")
            rpc_callee = RpcCallFeatureLookup(self)
            self.rpc_callee_id = rpc_register(rpc_callee)
        else:
            self.rpc_callee_id = None
    
    def lookup_features(
        self,
        ids: torch.Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.futures.Future:
        r""" Lookup the local/remote features based on node/edge ids """
    
        remote_fut = self._remote_lookup_features(ids, is_node_feat, input_type)
        local_feature = self._local_lookup_features(ids, is_node_feat, input_type)
        res_fut = torch.futures.Future()
        def when_finish(*_):
            try:
                remote_feature_list = remote_fut.wait()
                # combine the feature from remote and local
                result = torch.zeros(ids.shape[0], local_feature[0].shape[1], dtype=local_feature[0].dtype) 
                result[local_feature[1]] = local_feature[0]
                for remote in remote_feature_list:
                    result[remote[1]] = remote[0]
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)
        remote_fut.add_done_callback(when_finish)
        return res_fut
    
    def _local_lookup_features(
        self,
        ids: torch.Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.Tensor:
        # Collect the features locally based on node/edge ids
    
        if(self.meta["is_hetero"]):
            feat = self.local_feature[input_type]
            pb = self.feature_pb[input_type]
        else:
            feat = self.local_feature
            pb = self.feature_pb
        
        input_order= torch.arange(ids.size(0), dtype=torch.long) 
        partition_ids = pb[ids]
    
        local_mask = (partition_ids == self.partition_idx)
        local_ids = torch.masked_select(ids, local_mask)
        local_index = torch.masked_select(input_order, local_mask)
   
        if(self.meta["is_hetero"]):
            if is_node_feat:
                kwargs = dict(group_name=NodeType, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(index=local_ids, **kwargs)
            else:
                kwargs = dict(group_name=EdgeType, attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(index=local_ids, **kwargs)
        else:
            if is_node_feat:
                kwargs = dict(group_name=None, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(index=local_ids, **kwargs)
            else:
                kwargs = dict(group_name=(None, None), attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(index=local_ids, **kwargs)
    
        return ret_feat, local_index
    
    def _remote_lookup_features(
        self,
        ids: torch.Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.futures.Future:
        r""" fetch the remote features with the remote node/edge ids"""

        if(self.meta["is_hetero"]):
            pb = self.feature_pb[input_type]
        else:
            pb = self.feature_pb

        input_order= torch.arange(ids.size(0), dtype=torch.long) 
        partition_ids = pb[ids]
        futs, indexes = [], []
        for pidx in range(0, self.num_partitions):
            if pidx == self.partition_idx:
                continue
            remote_mask = (partition_ids == pidx)
            remote_ids = torch.masked_select(ids, remote_mask)
            if remote_ids.shape[0] > 0:
                to_worker = self.rpc_router.get_to_worker(pidx)
                futs.append(rpc_request_async(to_worker,
                                            self.rpc_callee_id,
                                            args=(remote_ids.cpu(), is_node_feat, input_type)))
                indexes.append(torch.masked_select(input_order, remote_mask))
        collect_fut = torch.futures.collect_all(futs)
        res_fut = torch.futures.Future()
        def when_finish(*_):
            try:
                fut_list = collect_fut.wait()
                result = []
                for i, fut in enumerate(fut_list):
                    result.append((fut.wait(), indexes[i]))
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)
        collect_fut.add_done_callback(when_finish)
        return res_fut
    
    def rpc_local_feature_get(
        self,
        ids: torch.Tensor,
        is_node_feat: bool = True,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.Tensor:
        r""" Lookup the features in local feature store
        input node/edge ids should be in local store. """
      
        if(self.meta["is_hetero"]):
            feat = self.local_feature[input_type]
            if is_node_feat:
                kwargs = dict(group_name=NodeType, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(index=ids, **kwargs)
            else:
                kwargs = dict(group_name=EdgeType, attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(index=ids, **kwargs)
        else:
            feat = self.local_feature
            if is_node_feat:
                kwargs = dict(group_name=None, attr_name='x')
                ret_feat = feat.get_tensor_from_global_id(index=ids, **kwargs)
            else:
                kwargs = dict(group_name=(None, None), attr_name='edge_attr')
                ret_feat = feat.get_tensor_from_global_id(index=ids, **kwargs)
      
        return ret_feat  
