

import torch
from typing import Dict, List, Optional, Tuple, Union

from torch_geometric.data import TensorAttr
from torch_geometric.distributed import LocalFeatureStore as Feature
from torch_geometric.typing import EdgeType, NodeType

from torch_geometric.utils import get_available_device, ensure_device

from torch_geometric.distributed import (
    RpcDataPartitionRouter, RpcCalleeBase, rpc_register, rpc_request_async
)


# Given a set of node ids, the `PartialFeature` stores the feature info
# of a subset of the original ids, the first tensor is the features of the
# subset node ids, and the second tensor records the index of the subset
# node ids.
PartialFeature = Tuple[torch.Tensor, torch.Tensor]


class RpcFeatureLookupCallee(RpcCalleeBase):
    r""" A wrapper for rpc callee that will perform feature lookup from
    remote processes.
    """
    def __init__(self, dist_feature):
        super().__init__()
        self.dist_feature = dist_feature
    
    def call(self, *args, **kwargs):
        return self.dist_feature.local_get(*args, **kwargs)


class DistFeature(object):
    r""" Distributed feature with partition information and also with
    the global feature lookups after node sampling.
    
    Args:
        num_partitions: Number of partitions.
        partition_id: partition idx for current process.
        local_feature: Local Feature instance.
        feature_pb: feature partition book between feature ids and partition id
        local_only: Use this to switch on/off only local feature lookup or stitching.
    """
    def __init__(self,
            meta: Dict,
            num_partitions: int,
            partition_idx: int,
            local_feature: Feature,
            feature_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor],
                                Dict[EdgeType, torch.Tensor]],
            local_only: bool = False,
            rpc_router: Optional[RpcDataPartitionRouter] = None,
            device: Optional[torch.device] = None):

        self.meta = meta
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        
        self.device = get_available_device(device)
        ensure_device(self.device)
        
        self.local_feature = local_feature
        self.feature_pb = feature_pb
        
        
        
        self.rpc_router = rpc_router
        if not local_only:
            if self.rpc_router is None:
                raise ValueError(f"'{self.__class__.__name__}': a rpc router must be "
                             f"provided when `local_only` set to `False`")
            rpc_callee = RpcFeatureLookupCallee(self)
            self.rpc_callee_id = rpc_register(rpc_callee)
        else:
            self.rpc_callee_id = None
        
    def _get_local_store(self, input_type: Optional[Union[NodeType, EdgeType]]):
        if(self.meta["is_hetero"]):
            assert input_type is not None
            return self.local_feature[input_type], self.feature_pb[input_type]
        return self.local_feature, self.feature_pb
    
    
    def local_get(
        self,
        ids: torch.Tensor,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.Tensor:
        r""" Lookup the features in local feature store
        input node/edge ids should be in local store. """
      
        feat, _ = self._get_local_store(input_type)
    
        tensor_attrs = feat.get_all_tensor_attrs()
        #*print(f"----------- 444.1 ----- dist_feature:  feat={feat}, tensor_attrs={tensor_attrs}  ------------ ")
        for item in tensor_attrs:
            #*print(f"---------- 444.2 ----- dist_feature:---- item={item}------------- ")
            #node_feats = node_feat_data.get_tensor(item)
            node_feats = feat.get_tensor(item.fully_specify())
            node_ids = feat.get_global_id(item.group_name) ##, item.attr_name)
    
            #print(f"------- 444.3 ----- dist_feature: node_feats ={node_feats}, node_ids={node_ids}, ids={ids}, max_ids={max(ids)} ----- ")
          
            kwargs = dict(group_name=item.group_name, attr_name=item.attr_name)
            ret_feat = feat.get_tensor_from_global_id(index=ids, **kwargs)
            #print(f"------- 444.4 ----- dist_feature: id2index={id2index} ----- ")
            #ret_feat = feat.get_tensor(item.group_name, item.attr_name, index=id2index[ids])
    
      
        #*print(f"------- 444.5 ----- dist_feature: end of local_get ...     --- ")
        #group_name = 'partition' + str(self.partition_idx)
        #attr_name = 'node_feat'
      
        return ret_feat  
    
    def async_get(
        self,
        ids: torch.Tensor,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.futures.Future:
        r""" Lookup the local/remote features asynchronously based on node ids 
        and return a future.    """
    
        device = torch.device(type='cpu')
        ids = ids.to(device)
        remote_fut = self._remote_selecting_get(ids, input_type)
        local_feature = self._local_selecting_get(ids, input_type)
        res_fut = torch.futures.Future()
        def on_done(*_):
            try:
                remote_feature_list = remote_fut.wait()
                result = self._stitch(ids, local_feature, remote_feature_list)
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)
        remote_fut.add_done_callback(on_done)
        return res_fut
    
    def __getitem__(
        self,
        input: Union[torch.Tensor, Tuple[Union[NodeType, EdgeType], torch.Tensor]]
    ) -> torch.Tensor:
        r""" Lookup features synchronously in a '__getitem__' way.
        """
        if isinstance(input, torch.Tensor):
            input_type, ids = None, input
        elif isinstance(input, tuple):
            input_type, ids = ids[0], ids[1]
        else:
            raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                           f"type for feature lookup: '{type(input)}'")
        fut = self.async_get(ids, input_type)
        return fut.wait()
    
    def _local_selecting_get(
        self,
        ids: torch.Tensor,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.Tensor:
        # Collect the local features based on local node ids
    
        device = torch.device(type='cpu')
    
        feat, pb = self._get_local_store(input_type)
        ids = ids.to(device)
        input_order= torch.arange(ids.size(0),
                                  dtype=torch.long,
                                  device=device)
        partition_ids = pb[ids].to(device)
    
        local_mask = (partition_ids == self.partition_idx)
        local_ids = torch.masked_select(ids, local_mask)
        local_index = torch.masked_select(input_order, local_mask)
     
        tensor_attrs = feat.get_all_tensor_attrs()
        #*print(f"----------- 555.1 ----- dist_feature:  feat={feat} , tensor_attrs={tensor_attrs}   ")
        for item in tensor_attrs:
            #print(f"---------- 555.2 ----- dist_feature:---- item={item}------------- ")
            #node_feats = node_feat_data.get_tensor(item)
            node_feats = feat.get_tensor(item.fully_specify())
            node_ids = feat.get_global_id(item.group_name) #, item.attr_name)
    
            #*print(f"------- 555.3 ----- dist_feature: node_feats ={node_feats}, node_ids={node_ids}, local_ids={local_ids}, max_local_ids={max(local_ids)} ----- ")
            
            kwargs = dict(group_name=item.group_name, attr_name=item.attr_name)
            ret_feat = feat.get_tensor_from_global_id(index=local_ids, **kwargs)
            #*print(f"------- 555.4 ----- dist_feature: id2index={id2index} ----- ")
    
            #ret_feat = feat.get_tensor(item.group_name, item.attr_name, index=id2index[local_ids])
    
        #*print(f"------- 555.5 ----- dist_feature: end of local_select_get() ...     --- ")
    
        #group_name = 'part_' + str(self.partition_idx)
        #attr_name = 'node_feat'
        #ind = feat.id2index[local_ids]
    
    
        r"""
        tensor_attrs = feat.get_all_tensor_attrs()
        for item in tensor_attrs:
            print(f"------------- item={item}------------- ")
            #node_feats = node_feat_data.get_tensor(item)
            node_feats = node_feat_data.get_tensor(item.fully_specify())
        """
        return ret_feat, local_index
    
    def _remote_selecting_get(
        self,
        ids: torch.Tensor,
        input_type: Optional[Union[NodeType, EdgeType]] = None
    ) -> torch.futures.Future:
        r""" fetch the remote features with the remote node ids
        Args:
            ids: input node/edge ids.
            input_type: input node/edge type for heterogeneous feature lookup.
        Return:
            torch.futures.Future: a torch future with a list of PartialFeature
        """
        assert (
            self.rpc_callee_id is not None
        ), "Remote feature collection is disabled by local_only."
    
        _, pb = self._get_local_store(input_type)
        device = torch.device(type='cpu')
        ids = ids.to(device)
    
        input_order= torch.arange(ids.size(0),
                                  dtype=torch.long,
                                  device=device)
        partition_ids = pb[ids].to(device)
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
                                            args=(remote_ids.cpu(), input_type)))
                indexes.append(torch.masked_select(input_order, remote_mask))
        collect_fut = torch.futures.collect_all(futs)
        res_fut = torch.futures.Future()
        def on_done(*_):
            try:
                fut_list = collect_fut.wait()
                result = []
                for i, fut in enumerate(fut_list):
                    result.append((fut.wait(), indexes[i]))
            except Exception as e:
                res_fut.set_exception(e)
            else:
                res_fut.set_result(result)
        collect_fut.add_done_callback(on_done)
        return res_fut
    
    def _stitch(
        self,
        ids: torch.Tensor,
        local: PartialFeature,
        remotes: List[PartialFeature]
    ) -> torch.Tensor:
        r""" Stitch local and remote features together.
        Args:
            ids: complete input node/edge ids.
            local: local partial feature.
            remotes: remote partial feature list.
        """
        feat = torch.zeros(ids.shape[0],
                           local[0].shape[1],
                           dtype=local[0].dtype,
                           device=self.device)
        feat[local[1].to(self.device)] = local[0].to(self.device)
        for remote in remotes:
            feat[remote[1].to(self.device)] = remote[0].to(self.device)
        return feat

