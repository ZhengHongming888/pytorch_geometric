import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import torch
from torch_geometric.loader.mixin import WorkerInitWrapper
from .dist_neighbor_sampler import DistNeighborSampler
from .rpc import init_rpc, shutdown_rpc
from .dist_context import init_worker_group, get_context
import copy
neighbor_sampler = None

class RPCMixin:
    
    
    @contextmanager
    def enable_rpc(self): #master_addr, master_port):
        print('RPC Context manager')

        neighbor_sampler = copy.deepcopy(neighbor_sampler_old)
        worker_init_fn_old = WorkerInitWrapper(self.worker_init_fn)
        def init_fn(worker_id):
            try:
                print(f"init_worker_group in worker loop worker_id-{worker_id}")
                init_worker_group(
                    world_size=2, #current_ctx.world_size * self.num_workers
                    rank=worker_id,
                    group_name='mp_sampling_worker'
                )
                print(f"init_rpc in worker loop worker_id-{worker_id}")
                init_rpc(
                    master_addr=os.getenv("MASTER_ADDR"),
                    master_port=int(os.getenv("MASTER_PORT")),
                    num_rpc_threads=16,
                    rpc_timeout=180
                )
                neighbor_sampler.init()
                neighbor_sampler.init_concurrent_event_loop()
                worker_init_fn_old(worker_id)

            except RuntimeError:
                pass
        try:
            print("changing worker_init_fn")
            self.worker_init_fn = init_fn
            self.neighbor_sampler = neighbor_sampler
            yield
            
        finally:
            self.worker_init_fn = worker_init_fn_old
            shutdown_rpc()            