# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, List, Optional, Tuple, Union

import torch

#from torch_geometric.data import LocalGraphStore as Graph
from torch_geometric.data import LocalFeatureStore as Feature

#from ..data.local_featurestore import Feature

#from torch_geometric.testing import MyFeatureStore as Feature
#from ..data import Feature
from ..typing import (
  EdgeType, NodeType,
  PartitionBook, HeteroNodePartitionDict, HeteroEdgePartitionDict
)
from ..utils import get_available_device, ensure_device

from .rpc import (
  RpcDataPartitionRouter, RpcCalleeBase, rpc_register, rpc_request_async
)
##
from torch_geometric.data import TensorAttr

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
  r""" Distributed feature data manager for global feature lookups.

  Args:
    num_partitions: Number of data partitions.
    partition_id: Data partition idx of current process.
    local_feature: Local ``Feature`` instance.
    feature_pb: Partition book which records node/edge ids to worker node
      ids mapping on feature store.
    local_only: Use this instance only for local feature lookup or stitching.
      If set to ``True``, the related rpc callee will not be registered and
      users should ensure that lookups for remote features are not invoked
      through this instance. Default to ``False``.
    device: Device used for computing. Default to ``None``.

  Note that`local_feature` and `feature_pb` should be a dictionary
  for hetero data.
  """
  def __init__(self,
               num_partitions: int,
               partition_idx: int,
               local_feature: Union[Feature,
                                    Dict[Union[NodeType, EdgeType], Feature]],
               feature_pb: Union[PartitionBook,
                                 HeteroNodePartitionDict,
                                 HeteroEdgePartitionDict],
               local_only: bool = False,
               rpc_router: Optional[RpcDataPartitionRouter] = None,
               device: Optional[torch.device] = None):
    self.num_partitions = num_partitions
    self.partition_idx = partition_idx

    self.device = get_available_device(device)
    ensure_device(self.device)

    self.local_feature = local_feature
    if isinstance(self.local_feature, dict):
      self.data_cls = 'hetero'
      for _, feat in self.local_feature.items():
        feat.lazy_init_with_ipc_handle()
    elif isinstance(self.local_feature, Feature):
      self.data_cls = 'homo'
      #self.local_feature.lazy_init_with_ipc_handle()
    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"feature type '{type(self.local_feature)}'")
    self.feature_pb = feature_pb
    if isinstance(self.feature_pb, dict):
      assert self.data_cls == 'hetero'
    elif isinstance(self.feature_pb, PartitionBook):
      assert self.data_cls == 'homo'
    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"patition book type '{type(self.feature_pb)}'")

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
    if self.data_cls == 'hetero':
      assert input_type is not None
      return self.local_feature[input_type], self.feature_pb[input_type]
    return self.local_feature, self.feature_pb

  def local_get(
    self,
    ids: torch.Tensor,
    input_type: Optional[Union[NodeType, EdgeType]] = None
  ) -> torch.Tensor:
    r""" Lookup features in the local feature store, the input node/edge ids
    should be guaranteed to be all local to the current feature store.
    """
    feat, _ = self._get_local_store(input_type)
    # TODO: check performance with `return feat[ids].cpu()`

    index = torch.tensor([0, 1, 2])
    #index = ids   #torch.arange(0, feature_data.size(0), 1)
    #*print(f"---888 ---- index={index}   ")

    #attr = TensorAttr(group_name, attr_name, index)
    #assert TensorAttr(group_name).update(attr) == attr

    #features.put_tensor(feature_data, attr)


    #ret_feat = feat.get_tensor(group_name, attr_name, index[0])
    #ret_feat = feat[group_name, attr_name, :]
    
    
    group_name = 'partition' + str(self.partition_idx)
    attr_name = 'node_feat'
    #t = feat.store[(group_name, 'node_feat')]

    #ret_feat = t[1][feat.id2index[ids]]


    ret_feat = feat.get_tensor(group_name, attr_name, index=feat.id2index[ids])
    
    #ret_feat = feat.get_feature(ids)
    #*print(f"\n\n-----999------ DistFeature:  local_get(), ret_feat={ret_feat} , ret_feat.shape={ret_feat.shape} ------")
    #print(f"\n\n-----999------ DistFeature:  local_get(), feat={feat}, ret_feat={ret_feat} ------")
    
    return ret_feat  #feat.cpu_get(ids)

  def async_get(
    self,
    ids: torch.Tensor,
    input_type: Optional[Union[NodeType, EdgeType]] = None
  ) -> torch.futures.Future:
    r""" Lookup features asynchronously and return a future.
    """
    #ids = ids.to(self.device)
    device = torch.device(type='cpu')
    ids = ids.to(device)
    #print(f"---------- DistFeature:   async_get() -- before self._remote_selecting_get --------------- ")
    remote_fut = self._remote_selecting_get(ids, input_type)
    #print(f"---------- DistFeature:   async_get() -- after self._remote_selecting_get, ------------- ")
    #print(f"---------- DistFeature:   async_get() -- after self._remote_selecting_get, remote_fut.wait()={remote_fut.wait()}------------- ")
    local_feature = self._local_selecting_get(ids, input_type)
    #print(f"---------- DistFeature:   async_get() -- after -_local_selecting_get -------------- ")
    res_fut = torch.futures.Future()
    def on_done(*_):
      try:
        remote_feature_list = remote_fut.wait()
        #print(f"----999.1--- DistFeature:   async_get() -- local_feature={local_feature} ") #local_feature.shape={local_feature.shape} -------------- ")
        #print(f"----999.2--- DistFeature:   async_get() -- remote_feature_list={remote_feature_list} ") # ,remote_feature_list.shape={remote_feature_list.shape} -------------- ")
        #print(f"----999.3--- DistFeature:   async_get() -- remote_feature_list[0][1].shape={remote_feature_list[0][1].shape} ") # ,remote_feature_list.shape={remote_feature_list.shape} -------------- ")
        #print(f"----999.4--- DistFeature:   async_get() -- remote_feature_list[0][0].shape={remote_feature_list[0][0].shape} ") # ,remote_feature_list.shape={remote_feature_list.shape} -------------- ")
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
    r""" Select node/edge ids only in the local feature store and lookup
    features of them.

    Args:
      ids: input node/edge ids.
      input_type: input node/edge type for heterogeneous feature lookup.

    Return:
      PartialFeature: features and index for local node/edge ids.
    """
    device = torch.device(type='cpu')

    feat, pb = self._get_local_store(input_type)
    ids = ids.to(device)
    #ids = ids.to(self.device)
    input_order= torch.arange(ids.size(0),
                              dtype=torch.long,
                              device=device)
                              #device=self.device)
    partition_ids = pb[ids].to(device)

    #partition_ids = pb[ids].to(self.device)
    local_mask = (partition_ids == self.partition_idx)
    local_ids = torch.masked_select(ids, local_mask)
    local_index = torch.masked_select(input_order, local_mask)
    #print(f"----888.1--- DistFeature:   _local_selecting_get() -- ids={ids}, ids.shape={ids.shape}, local_ids={local_ids}, local_ids.shape={local_ids.shape}, feat[local_ids]={feat[local_ids]} ") #local_feature.shape={local_feature.shape} -------------- ")
    
    group_name = 'partition' + str(self.partition_idx)
    attr_name = 'node_feat'
    
    #print(f"---- 888.2 ---- self.partition_idx={self.partition_idx}, feat.id2index={feat.id2index}, local_ids={local_ids}, feat.id2index[local_ids]={feat.id2index[local_ids]}, max = {max(feat.id2index[local_ids])}--- ")
    #print(f"---- 888.3 ---- feat.__dict__ = {feat.__dict__} --- ")
    #print(f"---- 888.4 ---- feat.store.keys() = {feat.store.keys(), feat.store.values()}, feat.store[]={feat.store[(group_name, 'node_feat')]} --- ")

    t = feat.store[(group_name, 'node_feat')]
    #print(f"---- t={len(t)}---")
    #print(f"---- t={t[0].size()}---")
    #print(f"---- t={t[1].size()}---")

    #print(f"---- 888.5 ---- feat._tensor_attr_cls = {feat._tensor_attr_cls.group_name} --- ")
    #print(f"---- 888.6 ---- feat.get_tensor = {feat.get_tensor(group_name, attr_name, index=torch.tensor([0, 2]))} --- ")
    ind = feat.id2index[local_ids]

    #ret_feat = feat.get_tensor(group_name, attr_name, index=ind[0:100000])
    ret_feat = feat.get_tensor(group_name, attr_name, index=feat.id2index[local_ids])
    
    #ret_feat = t[1][feat.id2index[local_ids]]
    #print(f"---- 888.7 ---- ret_feat={t[1][feat.id2index[local_ids]]}- ")
    #print(f"---- 888.7 ---- ret_feat={ret_feat}--- ")
    return ret_feat, local_index
    #return feat[local_ids], local_index

  def _remote_selecting_get(
    self,
    ids: torch.Tensor,
    input_type: Optional[Union[NodeType, EdgeType]] = None
  ) -> torch.futures.Future:
    r""" Select node/edge ids only in the remote feature stores and fetch
    their features.

    Args:
      ids: input node/edge ids.
      input_type: input node/edge type for heterogeneous feature lookup.

    Return:
      torch.futures.Future: a torch future with a list of `PartialFeature`,
        which corresponds to partial features on different remote workers.
    """
    assert (
      self.rpc_callee_id is not None
    ), "Remote feature lookup is disabled in 'local_only' mode."

    _, pb = self._get_local_store(input_type)
    #ids = ids.to(self.device)
    device = torch.device(type='cpu')
    ids = ids.to(device)

    input_order= torch.arange(ids.size(0),
                              dtype=torch.long,
                              device=device)
                              #device=self.device)
    partition_ids = pb[ids].to(device)
    #partition_ids = pb[ids].to(self.device)
    futs, indexes = [], []
    for pidx in range(0, self.num_partitions):
      if pidx == self.partition_idx:
        continue
      remote_mask = (partition_ids == pidx)
      remote_ids = torch.masked_select(ids, remote_mask)
      #*print(f"----888.3--- DistFeature:   _remote_selecting_get() -- ids={ids}, ids.shape={ids.shape}, remote_ids={remote_ids}, remote_ids.shape={remote_ids.shape} ") #local_feature.shape={local_feature.shape} -------------- ")
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
    r""" Stitch local and remote partial features into a complete one.

    Args:
      ids: the complete input node/edge ids.
      local: partial feature of local node/edge ids.
      remotes: partial feature list of remote node/edge ids.
    """
    feat = torch.zeros(ids.shape[0],
                       local[0].shape[1],
                       dtype=local[0].dtype,
                       device=self.device)
    feat[local[1].to(self.device)] = local[0].to(self.device)
    for remote in remotes:
      feat[remote[1].to(self.device)] = remote[0].to(self.device)
    return feat

