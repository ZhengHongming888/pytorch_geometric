import torch.multiprocessing as mp
import torch

from math import ceil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, Tuple
from pyparsing import Any


from ..channel import ChannelBase, SampleMessage
from torch_geometric.sampler import (
  NodeSamplerInput, EdgeSamplerInput,
  NeighborOutput, SamplerOutput, HeteroSamplerOutput,
  NeighborSampler, NegativeSampling, edge_sample_async
)
from torch_geometric.sampler.base import SubgraphType
from ..typing import EdgeType, as_str, NumNeighbors, OptTensor
from ..utils import (
    get_available_device, ensure_device, merge_dict, id2idx,
    merge_hetero_sampler_output, format_hetero_sampler_output,
    id2idx_v2
)

from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore
    )

from .event_loop import ConcurrentEventLoop, wrap_torch_future
from .rpc import (
  RpcCallBase, rpc_register, rpc_request_async,
  RpcRouter, rpc_partition2workers, shutdown_rpc
)
##
from torch_geometric.data import Data
from torch_geometric.data import TensorAttr
from torch_geometric.utils.map import map_index
from .dist_context import DistRole, DistContext

@dataclass
class PartialNeighborOutput:
  r""" The sampled neighbor output of a subset of the original ids.

  * index: the index of the subset vertex ids.
  * output: the sampled neighbor output.
  """
  index: torch.Tensor
  output: SamplerOutput


class RpcSamplingCallee(RpcCallBase):
  r""" A wrapper for rpc callee that will perform rpc sampling from
  remote processes.
  """
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def rpc_async(self, *args, **kwargs):
    ensure_device(self.device)
    output = self.sampler.sample_one_hop(*args, **kwargs)

    #if(output.device=='cpu'):
    return output
    #else:
    #    return output.to(torch.device('cpu'))
    
  def rpc_sync(self, *args, **kwargs):
      pass

class RpcSubGraphCallee(RpcCallBase):
  r""" A wrapper for rpc callee that will perform rpc sampling from
  remote processes.
  """
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def rpc_async(self, *args, **kwargs):
    ensure_device(self.device)
    with_edge = kwargs['with_edge']
    output = self.sampler.subgraph_op.node_subgraph(args[0].to(self.device),
                                                    with_edge)
    eids = output.eids.to('cpu') if with_edge else None
    return output.nodes.to('cpu'), output.rows.to('cpu'), output.cols.to('cpu'), eids

  def rpc_sync(self, *args, **kwargs):
      pass
    
class DistNeighborSampler():
  r""" Asynchronized and distributed neighbor sampler.

  Args:
    data (DistDataset): The graph and feature data with partition info.
    num_neighbors (NumNeighbors): The number of sampling neighbors on each hop.
    with_edge (bool): Whether to sample with edge ids. (default: ``None``).
    collect_features (bool): Whether collect features for sampled results.
      (default: ``None``).
    channel (ChannelBase, optional): The message channel to send sampled
      results. If set to `None`, the sampled results will be returned
      directly with `sample_from_nodes`. (default: ``None``).
    concurrency (int): The max number of concurrent seed batches processed by
      the current sampler. (default: ``1``).
    device: The device to use for sampling. If set to ``None``, the current
      cuda device (got by ``torch.cuda.current_device``) will be used if
      available, otherwise, the cpu device will be used. (default: ``None``).
  """
  def __init__(self,
               current_ctx: DistContext,
               rpc_worker_names: Dict[DistRole, List[str]],
               data: Tuple[LocalGraphStore, LocalFeatureStore],
               num_neighbors: Optional[NumNeighbors] = None,
               replace: bool = False,
               subgraph_type: Union[SubgraphType, str] = 'directional',
               disjoint: bool = False,
               with_edge: bool = True,
               collect_features: bool = False,
               channel: mp.Queue() = None, # change to pyg msg channel when ready
               concurrency: int = 1,
               device: Optional[torch.device] = None,
               **kwargs,
               ):
    self.current_ctx = current_ctx
    self.rpc_worker_names = rpc_worker_names

    self.data = data
    self.graph = data[0]
    self.feature = data[1]

    self.num_neighbors = num_neighbors
    self.with_edge = with_edge
    self.collect_features = collect_features
    self.channel = channel
    self.concurrency = concurrency
    self.device = get_available_device(device)
    self.event_loop = None
    self.replace = replace
    self.subgraph_type = subgraph_type
    self.disjoint = disjoint
    self.temporal_strategy = kwargs.pop('temporal_strategy', 'uniform')
    self.time_attr = kwargs.pop('time_attr', None)


  def register_sampler_rpc(self):
    #TODO: Check if all steps are executed correctly and inlcude edge\node info
  
    partition2workers = rpc_partition2workers(
      current_ctx = self.current_ctx,
      num_data_partitions=self.graph.num_partitions,
      current_partition_idx=self.graph.partition_idx
    )
    
    self.rpc_router = RpcRouter(partition2workers)
    self.dist_graph = self.graph

    
    edge_index = self.graph.get_edge_index(edge_type=None, layout='coo')

    self.dist_node_feature = None
    self.dist_edge_feature = None
    
    if self.collect_features:
      attrs = self.feature.get_all_tensor_attrs()
      #print(f"----{repr(self)}: attrs={attrs}  ")
      
      node_features = self.feature.get_tensor(group_name=None, attr_name='x')
      #print(f"----{repr(self)}: node_features={node_features} ")

      if node_features is not None:
          local_feature=self.feature
          local_feature.set_local_only(local_only=False)
          local_feature.set_rpc_router(self.rpc_router)
          
          self.dist_node_feature = local_feature # TODO: check this?
          
      edge_features = self.feature.get_tensor(group_name=attrs[0].group_name, attr_name=attrs[0].attr_name)
      #group_name=(None, None), attr_name='edge_attr'
      #print(f"----{repr(self)}: edge_features={edge_features} ")

      if self.with_edge and edge_features is not None:
        self.dist_edge_feature = None
        #TODO: edge feats
    # print(f"----{repr(self)}: data0={data0} ")
    
    self._sampler = NeighborSampler(
      data=(self.dist_node_feature, self.dist_graph),
      num_neighbors=self.num_neighbors,
      subgraph_type=self.subgraph_type,
      replace=self.replace,
      disjoint=self.disjoint,
      temporal_strategy=self.temporal_strategy,
      time_attr=self.time_attr,
    ) #TODO: BaseSampler
    
    rpc_sample_callee = RpcSamplingCallee(self._sampler, self.device)
    self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)
    
    rpc_subgraph_callee = RpcSubGraphCallee(self._sampler, self.device)
    self.rpc_subgraph_callee_id = rpc_register(rpc_subgraph_callee)

    if(self.dist_graph.meta["is_hetero"]):
    #if self.dist_graph.s_hetero: #self.dist_graph.data_cls == 'hetero':
      self.num_neighbors = self._sampler.num_neighbors
      self.num_hops = self._sampler.num_hops
      self.edge_types = self._sampler.edge_types


    
  def init_event_loop(self):
    

    self.event_loop = ConcurrentEventLoop(self.concurrency)
    self.event_loop._loop.call_soon_threadsafe(ensure_device, self.device)
    self.event_loop.start_loop()
    

  def sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[SampleMessage]:
    r""" Sample multi-hop neighbors from nodes, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (NodeSamplerInput): The input data with node indices to start
        sampling from.
    """
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.event_loop.run_task(coro=self._send_adapter(self.node_sample,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(self.node_sample, inputs),
                  callback=cb)
    return None
    
  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    neg_sampling: Optional[NegativeSampling] = None,
    **kwargs,
  ) -> Optional[SampleMessage]:
    r""" Sample multi-hop neighbors from edges, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (EdgeSamplerInput): The input data for sampling from edges
        including the (1) source node indices, the (2) destination node
        indices, the (3) optional edge labels and the (4) input edge type.
    """
    if self.channel is None:
      return self.event_loop.run_task(coro=self._send_adapter(edge_sample_async, inputs, self.node_sample,
                                                    self.sampler.num_nodes, self.disjoint,
                                                    self.sampler.node_time,
                                                    neg_sampling, distributed=True))
    cb = kwargs.get('callback', None)
    self.event_loop.add_task(coro=self._send_adapter(edge_sample_async, inputs, self.node_sample,
                                                    self.sampler.num_nodes, self.disjoint,
                                                    self.sampler.node_time,
                                                    neg_sampling, distributed=True),
                                                    callback=cb)
    return None

  def subgraph(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[SampleMessage]:
    r""" Induce an enclosing subgraph based on inputs and their neighbors(if
      self.num_neighbors is not None).
    """
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.run_task(coro=self._send_adapter(self._subgraph, inputs))
    cb = kwargs.get('callback', None)
    self.add_task(coro=self._send_adapter(self._subgraph, inputs), callback=cb)
    return None

  async def _send_adapter(
    self,
    async_func,
    *args, **kwargs
  ) -> Optional[SampleMessage]:
    
    sampler_output = await async_func(*args, **kwargs)
    # print(f'0.1 Sampler PID-{mp.current_process().pid} async sample_from nodes returned sampler output')
    res = await self._colloate_fn(sampler_output)
    # print(f'0.2 Sampler PID-{mp.current_process().pid} async _colloate_fn returned result')
    # print(f"\n\n--------------   dist_neighbor_sampler-PID{mp.current_process().pid}:   res={res} ----------- \n\n")
    #torch.save(res, "sample_output_for_channel.pt")

    if self.channel is None:
      return res
    self.channel.put(res)
    # print(f'0.3 Sampler PID-{mp.current_process().pid} sent result to PyG MSG channel')
    return None

  async def node_sample(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[SampleMessage]:
    
    # TODO: rm inducer and refactor sampling for hetero

    
    seed = inputs.node.to(self.device)
    seed_time = inputs.time.to(self.device) if inputs.time is not None else None
    input_type = inputs.input_type

    #*print(f" ----777.1 -------- distNSampler:  node_sample, self.dist_graph.data_cls={self.dist_graph.data_cls}, input_seeds={input_seeds}, input_type={input_type} ")
    if(self.dist_graph.meta["is_hetero"]):
      assert input_type is not None
      src_dict = inducer.init_node({input_type: input_seeds})
      batch_size = src_dict[input_type].numel()

      out_nodes, out_rows, out_cols, out_edges = {}, {}, {}, {}
      merge_dict(src_dict, out_nodes)

      for i in range(self.num_hops):
        task_dict, nbr_dict, edge_dict = {}, {}, {}
        for etype in self.edge_types:
          srcs = src_dict.get(etype[0], None)
          req_num = self.num_neighbors[etype][i]
          if srcs is not None:
            task_dict[etype] = self._loop.create_task(
              self._sample_one_hop(inputs, req_num, etype))
        for etype, task in task_dict.items():
          output: NeighborOutput = await task
          nbr_dict[etype] = [src_dict[etype[0]], output.nbr, output.nbr_num]
          if output.edge is not None:
            edge_dict[etype] = output.edge
        nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)
        merge_dict(nodes_dict, out_nodes)
        merge_dict(rows_dict, out_rows)
        merge_dict(cols_dict, out_cols)
        merge_dict(edge_dict, out_edges)
        src_dict = nodes_dict

      sample_output = HeteroSamplerOutput(
        node={ntype: torch.cat(nodes) for ntype, nodes in out_nodes.items()},
        row={etype: torch.cat(rows) for etype, rows in out_rows.items()},
        col={etype: torch.cat(cols) for etype, cols in out_cols.items()},
        edge=(
          {etype: torch.cat(eids) for etype, eids in out_edges.items()}
          if self.with_edge else None
        ),
        metadata={'input_type': input_type, 'bs': batch_size}
      )
    else:

      srcs = seed
      batch_size = seed.numel()
      
      out_nodes = [srcs]
      out_rows, out_cols, out_edge_ids, out_batch = [], [], [], []
      out_num_sampled_nodes_per_hop = [seed.numel()]
      out_num_sampled_edges_per_hop = [0]

      for one_hop_num in self.num_neighbors:
        out = await self._sample_one_hop(srcs, one_hop_num, seed_time)
        '''print("Debug,==========================================================")
        #print("self.graph:", self.graph)
        #print("self.feature global id 0:", self.feature._global_id)
        #print("self.feature global id 1:", self.feature._global_id_to_index)
        #test_node = self.feature.get_global_id(None)[out.node]
        print(f"Debug,===================srcs max: {srcs.max()}, shape: {srcs.shape}")
        print(f"Debug,===================out node max: {out.node.max()}, shape: {out.node.shape}")
        print(f"Debug,===================out row shape: {out.row.shape}")
        print(f"Debug,===================out row: {out.row.sort().values}")
        print(f"Debug,===================out col: {out.col.sort().values}")'''
        srcs = out.node
        out_nodes.append(out.node)
        out_rows.append(out.row)
        out_cols.append(out.col)
        out_edge_ids.append(out.edge)
        if self.disjoint:
          out_batch.append(out.batch)
        if out_num_sampled_nodes_per_hop != None:
          out_num_sampled_nodes_per_hop.append(len(out.node))
          out_num_sampled_edges_per_hop.append(len(out.row))

      sample_output = SamplerOutput(
        node=torch.cat(out_nodes),
        row=torch.cat(out_rows),
        col=torch.cat(out_cols),
        edge=(torch.cat(out_edge_ids) if self.with_edge else None),
        batch=(torch.cat(out_batch) if self.disjoint else None),
        num_sampled_nodes=out_num_sampled_nodes_per_hop if out_num_sampled_nodes_per_hop != None else None,
        num_sampled_edges=out_num_sampled_edges_per_hop if out_num_sampled_edges_per_hop != None else None,
        metadata={'input_type': None, 'bs': batch_size}
       )

    return sample_output

  async def _subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[SampleMessage]:
    #TODO: refactor
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    is_hetero = (self.dist_graph.data_cls == 'hetero')
    if is_hetero:
      raise NotImplementedError
    else:
      # neighbor sampling.
      if self.num_neighbors is not None:
        nodes = [input_seeds]
        for num in self.num_neighbors:
          nbr = await self._sample_one_hop(nodes[-1], num)
          nodes.append(torch.unique(nbr.nbr))
        nodes = torch.cat(nodes)
      else:
        nodes = input_seeds
      nodes, mapping = torch.unique(nodes, return_inverse=True)
      nid2idx = id2idx(nodes)
      # subgraph inducing.
      partition_ids = self.dist_graph.get_node_partitions(nodes)
      partition_ids = partition_ids.to(self.device)
      rows, cols, eids, futs = [], [], [], []
      for i in range(self.data.num_partitions):
        pidx = (self.data.partition_idx + i) % self.data.num_partitions
        p_ids = torch.masked_select(nodes, (partition_ids == pidx))
        if p_ids.shape[0] > 0:
          if pidx == self.data.partition_idx:
            subgraph = self._sampler.subgraph_op.node_subgraph(nodes, self.with_edge)
            # relabel row and col indices.
            rows.append(nid2idx[subgraph.nodes[subgraph.rows]])
            cols.append(nid2idx[subgraph.nodes[subgraph.cols]])
            if self.with_edge:
              eids.append(subgraph.eids.to(self.device))
          else:
            to_worker = self.rpc_router.get_to_worker(pidx)
            futs.append(rpc_request_async(to_worker,
                                          self.rpc_subgraph_callee_id,
                                          args=(nodes.cpu(),),
                                          kwargs={'with_edge': self.with_edge}))
      if not len(futs) == 0:
        res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
        for res_fut in res_fut_list:
          res_nodes, res_rows, res_cols, res_eids = res_fut.wait()
          res_nodes = res_nodes.to(self.device)
          rows.append(nid2idx[res_nodes[res_rows]])
          cols.append(nid2idx[res_nodes[res_cols]])
          if self.with_edge:
            eids.append(res_eids.to(self.device))

      sample_output = SamplerOutput(
        node=nodes,
        row=torch.cat(rows),
        col=torch.cat(cols),
        edge=torch.cat(eids) if self.with_edge else None,
        device=self.device,
        metadata={'mapping': mapping[:input_seeds.numel()]})

      return sample_output

  def merge_results(
    self,
    results: List[PartialNeighborOutput]
  ) -> NeighborOutput:
    r""" Merge partitioned neighbor outputs into a complete one.
    """
    out_nodes = torch.cat([r.output.node for r in results])
    out_nodes = out_nodes.unique()

    rows_list = []
    cols_list = []
    for r in results:
      global_rows = r.output.node[r.output.row]
      global_cols = r.output.node[r.output.col]
      local_rows,_ = map_index(global_rows, out_nodes, inclusive=True)
      local_cols,_ = map_index(global_cols, out_nodes, inclusive=True)
      rows_list.append(local_rows)
      cols_list.append(local_cols)

    out_rows = torch.cat([row for row in rows_list])
    out_cols = torch.cat([col for col in cols_list])
    out_edge_ids = torch.cat([r.output.edge for r in results]) if self.with_edge else None
    out_batch = torch.cat([r.output.batch for r in results]) if self.disjoint else None
    out_num_sampled_nodes_per_hop = len(out_nodes)
    out_num_sampled_edges_per_hop = len(out_rows)

    return SamplerOutput(
      out_nodes,
      out_rows,
      out_cols,
      out_edge_ids,
      out_batch,
      out_num_sampled_nodes_per_hop,
      out_num_sampled_edges_per_hop)

  async def _sample_one_hop(
    self,
    srcs: torch.Tensor,
    one_hop_num: int,
    seed_time: OptTensor = None,
    src_etype: Optional[EdgeType] = None,
  ) -> SamplerOutput:
  #) -> NeighborOutput:
    r""" Sample one-hop neighbors and induce the coo format subgraph.

    Args:
      srcs: input ids, 1D tensor.
      num_nbr: request(max) number of neighbors for one hop.
      etype: edge type to sample from input ids.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: unique node ids and edge_index.
    """
    device = torch.device(type='cpu')

    srcs = srcs.to(device) # all src nodes
    seed_time = seed_time.to(device) if seed_time is not None else None

    nodes = torch.arange(srcs.size(0), dtype=torch.long, device=device)
    src_ntype = src_etype[0] if src_etype is not None else None
    
    partition_ids = self.dist_graph.get_partition_ids_from_nids(srcs, src_ntype) # na których partycjach jest dany node {0,1}
    partition_ids = partition_ids.to(self.device)

    partition_results: List[PartialNeighborOutput] = []
    remote_nodes: List[torch.Tensor] = []
    futs: List[torch.futures.Future] = []

    #*print(f"-------- DistNSampler: async _sample_one_hop() enter multi-partition sample_one_hop() -------")

    for i in range(self.graph.num_partitions): # 2 - number of machines
      pidx = (
        (self.graph.partition_idx + i) % self.graph.num_partitions
        #(self.data.partition_idx + i) % self.data.num_partitions
      )
      p_mask = (partition_ids == pidx) # [true, false]
      p_srcs = torch.masked_select(srcs, p_mask) # id nodów, które są True w masce

      if p_srcs.shape[0] > 0:
        p_nodes = torch.masked_select(nodes, p_mask) # indeksy nodów z srcs, które są na danej partycji
        if pidx == self.graph.partition_idx:
          
          p_nbr_out = self._sampler.sample_one_hop(p_srcs, one_hop_num, seed_time, src_etype)
          partition_results.append(PartialNeighborOutput(p_nodes, p_nbr_out))
          #print("p_nbr_out:", p_nbr_out)
        else:
          remote_nodes.append(p_nodes)
          to_worker = self.rpc_router.get_to_worker(pidx)
          futs.append(rpc_request_async(to_worker,
                                        self.rpc_sample_callee_id,
                                        args=(p_srcs.cpu(), one_hop_num, src_etype)))
    
    # Without remote sampling results.
    if len(remote_nodes) == 0:
      return partition_results[0].output
    # With remote sampling results.
    res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
    for i, res_fut in enumerate(res_fut_list):
      #*print(f"-------- DistNSampler: async _sample_one_hop() res_fut={res_fut.wait()} -------")
      remote_neighbor_out = PartialNeighborOutput(
                              index=remote_nodes[i],
                              output=res_fut.wait())

      partition_results.append(remote_neighbor_out)
      '''partition_results.append(
        PartialNeighborOutput(
          index=remote_nodes[i],
          output=res_fut.wait()
          #output=res_fut.wait().to(device)
        )
      )'''
    
    #*print(f"-------- DistNSampler: async _sample_one_hop() before stitching -----------------  partition_results={partition_results}-------")
    #*print("\n\n\n\n")
    return self.merge_results(partition_results)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> Union[SamplerOutput, HeteroSamplerOutput]:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    result_map = {}
    # if isinstance(output.metadata, dict):
      #scan kv and add metadata
    # TODO: check input type
    input_type = output.metadata.get('input_type', '')
    # print(f"input_type from output.metadata={input_type}")
      # batch_size = output.metadata.get('bs', 1)
      # result_map['meta'] = torch.LongTensor([int(is_hetero), batch_size])
      # # output.metadata.pop('input_type', '')
      # # output.metadata.pop('bs', 1)
      # for k, v in output.metadata.items():
      #   result_map[k] = v
  
    if self.dist_graph.meta["is_hetero"]:
      for ntype, nodes in output.node.items():
        result_map[f'{as_str(ntype)}.ids'] = nodes
      for etype, rows in output.row.items():
        etype_str = as_str(etype)
        result_map[f'{etype_str}.rows'] = rows
        result_map[f'{etype_str}.cols'] = output.col[etype]
        if self.with_edge:
          result_map[f'{etype_str}.eids'] = output.edge[etype]
          
      # Collect node labels of input node type.
      if not isinstance(input_type, Tuple):
        node_labels = self.data.get_node_label(input_type)
        if node_labels is not None:
          result_map[f'{as_str(input_type)}.nlabels'] = \
            node_labels[output.node[input_type]]
      # Collect node features.
      if self.dist_node_feature is not None:
        nfeat_fut_dict = {}
        for ntype, nodes in output.node.items():
          nodes = nodes.to(torch.long)
          nfeat_fut_dict[ntype] = self.dist_node_feature.async_get(nodes, ntype)
        for ntype, fut in nfeat_fut_dict.items():
          nfeats = await wrap_torch_future(fut)
          result_map[f'{as_str(ntype)}.nfeats'] = nfeats
      # Collect edge features
      if self.dist_edge_feature is not None and self.with_edge:
        efeat_fut_dict = {}
        for etype in self.edge_types:
          eids = result_map.get(f'{as_str(etype)}.eids', None).to(torch.long)
          if eids is not None:
            efeat_fut_dict[etype] = self.dist_edge_feature.async_get(eids, etype)
        for etype, fut in efeat_fut_dict.items():
          efeats = await wrap_torch_future(fut)
          result_map[f'{as_str(etype)}.efeats'] = efeats
    else:
      # result_map['ids'] = output.node
      # result_map['rows'] = output.row
      # result_map['cols'] = output.col
      # if self.with_edge:
      #   result_map['eids'] = output.edge
      # Collect node labels.
      node_labels = self.graph.labels #self.data.get_node_label()
      if node_labels is not None:
        #result_map['nlabels'] = node_labels[output.node]
        output.metadata['nlabels'] = node_labels[output.node]
      # Collect node features.
      if self.dist_node_feature is not None:
        #print(f"------- 777.3---- DistNSampler: _colloate_fn()   dist_node_feature.async_get(),   output.node={output.node}, self.dist_node_feature={self.dist_node_feature}, self.dist_edge_feature={self.dist_edge_feature} -------")
        fut = self.dist_node_feature.lookup_features(is_node_feat=True, ids=output.node)
        #print(f"Sampler PID-{mp.current_process().pid} 1: async_get returned {fut}")
        nfeats = await wrap_torch_future(fut) #torch.Tensor([])
        #print(f"Sampler PID-{mp.current_process().pid} 2: wrap_torch_feature returned {nfeats}")
        #result_map['nfeats'] = nfeats
        output.metadata['nfeats'] = nfeats.to(torch.device('cpu'))
        output.edge=torch.empty(0) # ! HOTFIX for pygmsg channel format - rm later!

      # Collect edge features.
      if self.dist_edge_feature is not None:
        eids = result_map['eids']
        fut = self.dist_edge_feature.lookup_features(is_node_feat=False, ids=eids)
        efeats = await wrap_torch_future(fut)
        output.metadata['efeats'] = efeats
      else:
        output.metadata['efeats'] = None
      
      #print(f"------- 777.4 ----- DistNSampler: _colloate_fn()  return -------")

    return output #result_map
  
  def __repr__(self):
    return f"{self.__class__.__name__}()-PID{mp.current_process().pid}"
  
# Sampling Utilities ##########################################################

def sample(
    inputs: NodeSamplerInput,
    sample_fn: Callable,
    _sample_fn: Callable,
    ) -> Optional[Union[SamplerOutput, HeteroSamplerOutput]]:
  return sample_fn(inputs, _sample_fn)


def close_sampler(worker_id, sampler):
  try:
    sampler.event_loop.shutdown_loop()
  except AttributeError:
    pass
  shutdown_rpc(graceful=True)
