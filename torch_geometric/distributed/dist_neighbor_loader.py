import torch
from typing import Callable, Optional, Tuple, Dict, Union, List

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, as_str
from torch_geometric.loader.node_loader import NodeLoader

from .dist_context import DistRole, DistContext
from .dist_loader import DistLoader
from .dist_neighbor_sampler import DistNeighborSampler
from .local_graph_store import LocalGraphStore
from .local_feature_store import LocalFeatureStore



class DistNeighborLoader(NodeLoader, DistLoader):
    r""" A distributed loader that preform sampling from nodes.

    Args:
      data (DistDataset, optional): The ``DistDataset`` object of a partition of
        graph data and feature data, along with distributed patition books. The
        input dataset must be provided in non-server distribution mode.
      num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
        The number of neighbors to sample for each node in each iteration.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
      input_nodes (torch.Tensor or Tuple[str, torch.Tensor]): The node seeds for
        which neighbors are sampled to create mini-batches. In heterogeneous
        graphs, needs to be passed as a tuple that holds the node type and
        node seeds.
      batch_size (int): How many samples per batch to load (default: ``1``).
      shuffle (bool): Set to ``True`` to have the data reshuffled at every
        epoch (default: ``False``).
      drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
        the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last
        batch will be smaller. (default: ``False``).
      with_edge (bool): Set to ``True`` to sample with edge ids and also include
        them in the sampled results. (default: ``False``).
      collect_features (bool): Set to ``True`` to collect features for nodes
        of each sampled subgraph. (default: ``False``).
      to_device (torch.device, optional): The target device that the sampled
        results should be copied to. If set to ``None``, the current cuda device
        (got by ``torch.cuda.current_device``) will be used if available,
        otherwise, the cpu device will be used. (default: ``None``).
      worker_options (optional): The options for launching sampling workers.
        (1) If set to ``None`` or provided with a ``CollocatedDistWorkerOptions``
        object, a single collocated sampler will be launched on the current
        process, while the separate sampling mode will be disabled . (2) If
        provided with a ``MpDistWorkerOptions`` object, the sampling workers will
        be launched on spawned subprocesses, and a share-memory based channel
        will be created for sample message passing from multiprocessing workers
        to the current loader. (3) If provided with a ``RemoteDistWorkerOptions``
        object, the sampling workers will be launched on remote sampling server
        nodes, and a remote channel will be created for cross-machine message
        passing. (default: ``None``).
    """

    def __init__(self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        data: Tuple[LocalGraphStore, LocalFeatureStore],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        device: torch.device = None,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[DistNeighborSampler] = None,
        async_sampling: bool = True,
        master_addr=None,
        master_port=None,
        **kwargs,
        ):

        DistLoader.__init__(self,
          current_ctx = current_ctx,
          rpc_worker_names = rpc_worker_names,
          data = data, 
          num_neighbors = num_neighbors,
          neighbor_sampler = neighbor_sampler, 
          async_sampling = async_sampling,
          filter_per_worker = filter_per_worker,
          device = device, 
          master_addr = master_addr, 
          master_port = master_port, 
          **kwargs
        )
        
        # rm sampler & dist kwargs
        sampler_args = ['strategy','worker_concurrency', 'collect_features', 'with_edge', 'with_neg']
        rpc_args = ['num_rpc_threads', 'rpc_timeout']
        for k in (sampler_args + rpc_args):
          kwargs.pop(k, None)
        
        NodeLoader.__init__(self,
            data=data,
            node_sampler=self.neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            custom_init=self.init_fn,
            custom_filter=self.filter_fn,
            **kwargs
        )
        
    def __repr__(self) -> str:
      return DistLoader.__repr__(self)
  
