import socket
from typing import Dict, List

import torch

import torch_geometric.distributed.rpc as rpc
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.rpc import RPCRouter
from torch_geometric.testing import onlyLinux

import torch_geometric.distributed as pyg_dist

import os.path as osp

import pytest
import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.typing import EdgeTypeStr



def run_rpc_feature_test(
    world_size: int,
    rank: int,
    root_dir: str,
    master_port: int,
):
    print(f"------------ rank={rank}, root_dir={root_dir} ")

    graph = LocalGraphStore.from_partition(root_dir, rank)
    print(f"-------- graph={graph} ")
    edge_attrs = graph.get_all_edge_attrs()
    print(f"-----777---- edge_attrs ={edge_attrs}")

    feature = LocalFeatureStore.from_partition(root_dir, rank)
    print(f"-------- feature={feature} ")

    meta = {
        'edge_types': None,
        'is_hetero': False,
        'node_types': None,
        'num_parts': 2
    }
    graph.num_partitions = world_size
    graph.partition_idx = rank
    #graph.node_pb = node_pb
    #graph.edge_pb = edge_pb
    graph.meta = meta


    feature.num_partitions = world_size
    feature.partition_idx = rank
    #feature.feature_pb = partition_book
    feature.meta = meta
    #feature.set_rpc_router(rpc_router)

    partition_data = (graph, feature)


    # 1) Initialize the context info:
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-feature-test',
    )
    rpc_worker_names: Dict[DistRole, List[str]] = {}

    print(f"----------- 222 ------------- ")
    
    batch_size = 256

    # Create distributed neighbor loader for training
    train_idx = torch.arange(128 * 2) + 128 * rank
    #train_idx = train_idx.split(train_idx.size(0) // num_training_procs_per_node)[local_proc_rank]
    num_workers=2
    train_loader = pyg_dist.DistNeighborLoader(
      current_ctx,
      rpc_worker_names,
      data=partition_data,

      num_neighbors=[15, 10, 5],
      input_nodes=train_idx,
      batch_size=batch_size,
      shuffle=True,
      collect_features=True,
      device=torch.device('cpu'),
      num_workers=num_workers,
      worker_concurrency=4,
      master_addr='127.0.0.1',
      master_port=master_port,
      async_sampling = True,
      filter_per_worker = False,
    )

    print("---------- done ------------ ")
    r"""
    rpc.init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names=rpc_worker_names,
        master_addr='localhost',
        master_port=master_port,
    )

    # 2) Collect all workers:
    partition_to_workers = rpc.rpc_partition_to_workers(
        current_ctx, world_size, rank)

    assert partition_to_workers == [
        ['dist-feature-test-0'],
        ['dist-feature-test-1'],
    ]

    # 3) Find the mapping between worker and partition ID:
    rpc_router = RPCRouter(partition_to_workers)

    assert rpc_router.get_to_worker(partition_idx=0) == 'dist-feature-test-0'
    assert rpc_router.get_to_worker(partition_idx=1) == 'dist-feature-test-1'
    """

    r"""
    meta = {
        'edge_types': None,
        'is_hetero': False,
        'node_types': None,
        'num_parts': 2
    }

    feature.num_partitions = world_size
    feature.partition_idx = rank
    feature.feature_pb = partition_book
    feature.meta = meta
    feature.set_local_only(local_only=False)
    feature.set_rpc_router(rpc_router)

    # Global node IDs:
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128 * 2

    # Lookup the features from stores including locally and remotely:
    tensor0 = feature.lookup_features(global_id0)
    tensor1 = feature.lookup_features(global_id1)

    # Expected searched results:
    cpu_tensor0 = torch.cat([torch.ones(128, 1024), torch.ones(128, 1024) * 2])
    cpu_tensor1 = torch.cat([torch.zeros(128, 1024), torch.zeros(128, 1024)])

    # Verify..
    assert torch.allclose(cpu_tensor0, tensor0.wait())
    assert torch.allclose(cpu_tensor1, tensor1.wait())

    rpc.shutdown_rpc()
    """
    rpc.shutdown_rpc()


@onlyLinux
def test_dist_feature_lookup():
    r"""
    cpu_tensor0 = torch.cat([torch.ones(128, 1024), torch.ones(128, 1024) * 2])
    cpu_tensor1 = torch.cat([torch.zeros(128, 1024), torch.zeros(128, 1024)])

    # Global node IDs:
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128 * 2

    # Set the partition book for two features (partition 0 and 1):
    partition_book = torch.cat([
        torch.zeros(128 * 2, dtype=torch.long),
        torch.ones(128 * 2, dtype=torch.long)
    ])

    # Put the test tensor into the different feature stores with IDs:
    feature0 = LocalFeatureStore()
    feature0.put_global_id(global_id0, group_name=None)
    feature0.put_tensor(cpu_tensor0, group_name=None, attr_name='x')

    feature1 = LocalFeatureStore()
    feature1.put_global_id(global_id1, group_name=None)
    feature1.put_tensor(cpu_tensor1, group_name=None, attr_name='x')
    """

    data = FakeDataset()[0]
    num_parts = 2
    root_dir = "./partition"

    partitioner = Partitioner(data, num_parts, root_dir)
    partitioner.generate_partition()


    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    w0 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 0, root_dir, port))
    w1 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 1, root_dir, port))

    w0.start()
    w1.start()
    w0.join()
    w1.join()
