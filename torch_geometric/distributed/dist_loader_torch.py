from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, Dict
import torch
import time
import os
import atexit
import copy

import multiprocessing as mp
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from ..sampler import NodeSamplerInput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor
from torch_geometric.loader.mixin import AffinityMixin
#from .dist_mixin import RPCMixin

from .transformer import to_data, to_hetero_data
from .dist_neighbor_sampler import DistNeighborSampler, close_sampler
#from .dist_dataset import DistDataset
from .rpc import init_rpc, global_barrier
#from .dist_context import get_context
from .dist_context import DistRole, DistContext
from torch_geometric.loader.base import DataLoaderIterator

#from PyG_MessageQueue import pyg_messagequeue as pygmsg
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.distributed.local_feature_store import LocalFeatureStore


from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from ..typing import (
  NodeType, EdgeType, as_str, reverse_edge_type
)

SAMPLERS = [] # global registry of samplers created by this dataloader
#DATALOADER_ID = 0

class DistNeighborLoaderTorch(torch.utils.data.DataLoader, AffinityMixin): #, RPCMixin):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, up to :obj:`num_neighbors`
    neighbors will be sampled for each :obj:`edge_type`.
    However, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. The
    :class:`~torch_geometric.loader.NeighborLoader` will include this mapping
    as part of the :obj:`data` object:

    .. code-block:: python

        loader = NeighborLoader(data, ...)
        sampled_data = next(iter(loader))
        print(sampled_data.n_id)  # Global node index of each node in batch.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        input_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        disjoint (bool, optional): If set to :obj: `True`, each seed node will
            create its own disjoint subgraph.
            If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
            vector holding the mapping of nodes to their respective subgraph.
            Will get automatically set to :obj:`True` in case of temporal
            sampling. (default: :obj:`False`)
        temporal_strategy (str, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            If set to :obj:`"uniform"`, will sample uniformly across neighbors
            that fulfill temporal constraints.
            If set to :obj:`"last"`, will sample the last `num_neighbors` that
            fulfill temporal constraints.
            (default: :obj:`"uniform"`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the center node.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column.
            If :obj:`time_attr` is set, additionally requires that rows are
            sorted according to time within individual neighborhoods.
            This avoids internal re-sorting of the data and can improve
            runtime and memory efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` for in-memory datasets is generally not
            recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        data: Tuple[LocalGraphStore, LocalFeatureStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes,
        batch_size: int = 128,
        input_time: OptTensor = None,
        num_workers: int = 0,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        master_addr = None,
        master_port = None,
        device: torch.device = None,
        neighbor_sampler = None,
        async_sampling: bool = False,
        **kwargs,
    ):
        self.pid = mp.current_process().pid
        #self.current_ctx = get_context()     
        global SAMPLERS
        if isinstance(input_nodes, tuple):
            self.input_type, self.input_seeds = input_nodes
        else:
            self.input_type, self.input_seeds = None, input_nodes
        print(f"------- distNeighborLoader:  input_type={self.input_type}, input_seeds={self.input_seeds}   ---")

        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")
        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names
        self.data = data
        self.device = device # 'cuda' or 'cpu' not an exact device
        self.num_workers = num_workers
        self.master_addr = str(master_addr)
        self.master_port = int(master_port)
        self.batch_size = batch_size
        self.filter_per_worker = filter_per_worker
        self.channel = None # pygmsg.PyGMessageQueue(shm_size=134217728*32) if async_sampling else None
        #mp.set_start_method("spawn") # assert: context should be 'spawn'
        self.input_data = NodeSamplerInput(node=self.input_seeds, input_type=self.input_type, input_id=None)
        # dummy init sampler will be registered in the _worker_loop, but each subprocess opens a copy
        if neighbor_sampler is None:

            self.neighbor_sampler = DistNeighborSampler(
                self.current_ctx,
                self.rpc_worker_names,
                self.data, 
                num_neighbors=num_neighbors, 
                device=self.device,
                channel=self.channel,
                concurrency=kwargs.pop('worker_concurrency', 4),
                collect_features=kwargs.pop('collect_features', True),
                with_edge=kwargs.pop('with_edge', False)
            )
        if self.num_workers > 0:
            self.input_data = self.input_data.share_memory()
        else:
            self.init_fn(0)
        iterator = range(input_nodes.size(0))
        print("register_rpc torch backend")
        super().__init__(
            iterator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=self.init_fn,
            **kwargs)
        print(f"--------444.1 super.init ------- ----end-----")
        
        #del self.neighbor_sampler
        
    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        input_data: NodeSamplerInput = self.input_data[index]
        
        out = self.neighbor_sampler.sample_from_nodes(input_data)
        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out
    
    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
        ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.channel and not self.filter_per_worker:
            out = self.channel.get()
            print(f'{repr(self)} retrieved Sampler result from PyG MSG channel')
        is_hetero = False #bool(out['meta'][0])
        # Heterogeneous sampling results
        if is_hetero:
            node_dict, row_dict, col_dict, edge_dict = {}, {}, {}, {}
            nfeat_dict, efeat_dict = {}, {}

            for ntype in self._node_types:
                ids_key = f'{as_str(ntype)}.ids'
                if ids_key in out:
                    node_dict[ntype] = out[ids_key].to(self.to_device)
                    nfeat_key = f'{as_str(ntype)}.nfeats'
                if nfeat_key in out:
                    nfeat_dict[ntype] = out[nfeat_key].to(self.to_device)

            for etype_str, rev_etype in self._etype_str_to_rev.items():
                rows_key = f'{etype_str}.rows'
                cols_key = f'{etype_str}.cols'
                if rows_key in out:
                    # The edge index should be reversed.
                    row_dict[rev_etype] = out[cols_key].to(self.to_device)
                    col_dict[rev_etype] = out[rows_key].to(self.to_device)
                    eids_key = f'{etype_str}.eids'
                if eids_key in out:
                    edge_dict[rev_etype] = out[eids_key].to(self.to_device)
                    efeat_key = f'{etype_str}.efeats'
                if efeat_key in out:
                    efeat_dict[rev_etype] = out[efeat_key].to(self.to_device)

            batch_dict = {self.input_type: node_dict[self.input_type][:self.batch_size]}
            output = HeteroSamplerOutput(node_dict, row_dict, col_dict,
                                        edge_dict if len(edge_dict) else None,
                                        batch_dict, edge_types=self._edge_types,
                                        device=self.to_device)

            if len(nfeat_dict) == 0:
                nfeat_dict = None
            if len(efeat_dict) == 0:
                efeat_dict = None

            batch_labels_key = f'{self.input_type}.nlabels'
            if batch_labels_key in out:
                batch_labels = out[batch_labels_key].to(self.to_device)
            else:
                batch_labels = None
            label_dict = {self.input_type: batch_labels}

            res_data = to_hetero_data(output, label_dict, nfeat_dict, efeat_dict)

        # Homogeneous sampling results
        else:
            # ids = out['ids'].to(device)
            # rows = out['rows'].to(device)
            # cols = out['cols'].to(device)
            # eids = out['eids'].to(device) if 'eids' in out else None
            # #ids = out['ids'].to(self.to_device)
            # #rows = out['rows'].to(self.to_device)
            # #cols = out['cols'].to(self.to_device)
            # #eids = out['eids'].to(self.to_device) if 'eids' in out else None
            # batch = ids[:self.batch_size]
            # # The edge index should be reversed.
            # output = SamplerOutput(ids, cols, rows, eids, batch)
            #                        #device=device)
            #                        #device=self.to_device)

            # batch_labels = out['nlabels'].to(self.to_device) if 'nlabels' in out else None

            # nfeats = out['nfeats'].to(self.to_device) if 'nfeats' in out else None
            # efeats = out['efeats'].to(self.to_device) if 'efeats' in out else None

            res_data = to_data(out, out.metadata['nlabels'], out.metadata['nfeats'], out.metadata['efeats'], batch_size=self.batch_size) #filter_fn

        return res_data

    def _get_iterator(self) -> Iterator:

        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return  f"{self.__class__.__name__}()-PID{self.pid}@{self.device}"
    
    def init_fn(self, worker_id):
        try:
            if self.num_workers > 0:
                print(f"init_worker_group in {repr(self.neighbor_sampler)}._worker_loop(), worker_id-{worker_id}")
                
                self.current_ctx_for_mploader = DistContext(world_size=self.current_ctx.world_size * self.num_workers,
                                    rank=self.current_ctx.rank * self.num_workers + worker_id,
                                    global_world_size=self.current_ctx.world_size * self.num_workers,
                                    global_rank=self.current_ctx.rank * self.num_workers + worker_id,
                                    group_name='mp_sampling_worker')
            
            print(f"init_rpc in {repr(self.neighbor_sampler)}._worker_loop(), worker_id-{worker_id}")

            self.rpc_worker_names_for_mploader: Dict[DistRole, List[str]] = {}
            init_rpc(
                current_ctx=self.current_ctx_for_mploader,
                rpc_worker_names=self.rpc_worker_names_for_mploader,    
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=16,
                rpc_timeout=180
            )
            self.neighbor_sampler.register_rpc()     
            self.neighbor_sampler.init_event_loop()
           
            print(f"------- 66666.1--------  init_fn() ")
            atexit.register(close_sampler, worker_id, self.neighbor_sampler) # register clean exit for samplers
            global_barrier()
            
        except RuntimeError:
            raise RuntimeError("Something went wrong in init_fn")
