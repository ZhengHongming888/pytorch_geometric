
import unittest
import socket

import torch

from torch_geometric.distributed import (
    RpcDataPartitionRouter, RpcCalleeBase, rpc_register, rpc_request_async
)

import torch_geometric.distributed as pyg_dist
import torch_geometric.distributed.rpc

from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed import DistFeature


def run_dist_feature_test(world_size: int, rank: int, feature: LocalFeatureStore,
                          partition_book: torch.Tensor, master_port: int):
    # init the workers
    pyg_dist.init_worker_group(world_size, rank, 'dist-feature-test')
    pyg_dist.init_rpc(master_addr='localhost', master_port=master_port)

    # collect all workers
    partition2workers = pyg_dist.rpc_sync_data_partitions(world_size, rank)
    print(f"------rank={rank}------- partition2workers={partition2workers}")
    
    # find worker with partition id 
    rpc_router = pyg_dist.RpcDataPartitionRouter(partition2workers)
    print(f"------rank={rank}------- rpc_router={rpc_router.get_to_worker(1)} ")

    current_device = torch.device('cpu')

    meta = {
        "edge_types": None,
        "is_hetero": False,
        "node_types": None,
        "num_parts": 2
    }

    dist_feature = pyg_dist.DistFeature(
      num_partitions=world_size, partition_index=rank, local_feature=feature, feature_pb=partition_book,
      local_only=False, rpc_router=rpc_router, meta = meta
    )

    # global node ids
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128*2

    # lookup the features from stores including locally and remotely
    tensor0 = dist_feature.lookup_features(global_id0)
    tensor1 = dist_feature.lookup_features(global_id1)

    # expected searched results
    cpu_tensor0 = torch.cat([
      torch.ones(128, 1024, dtype=torch.float32),
      torch.ones(128, 1024, dtype=torch.float32)*2
    ])
    cpu_tensor1 = torch.cat([
      torch.zeros(128, 1024, dtype=torch.float32),
      torch.zeros(128, 1024, dtype=torch.float32)
    ])

    # verify..
    assert torch.allclose(cpu_tensor0, tensor0.wait())
    assert torch.allclose(cpu_tensor1, tensor1.wait())
    
    pyg_dist.shutdown_rpc()


def test_dist_feature_lookup():

    cpu_tensor0 = torch.cat([
      torch.ones(128, 1024, dtype=torch.float32),
      torch.ones(128, 1024, dtype=torch.float32)*2
    ])
    cpu_tensor1 = torch.cat([
      torch.zeros(128, 1024, dtype=torch.float32),
      torch.zeros(128, 1024, dtype=torch.float32)
    ])

    # global node ids
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128*2

    # set the partition book for two features, partition 0, 1
    partition_book = torch.cat([
      torch.zeros(128*2, dtype=torch.long),
      torch.ones(128*2, dtype=torch.long)
    ])

    # put the test tensor into the different feature stores with ids
    feature0 = LocalFeatureStore()
    feature0.put_global_id(global_id0, group_name=None)
    feature0.put_tensor(cpu_tensor0, group_name=None, attr_name='x')

    feature1 = LocalFeatureStore()
    feature1.put_global_id(global_id1, group_name=None)
    feature1.put_tensor(cpu_tensor1, group_name=None, attr_name='x')


    mp_context = torch.multiprocessing.get_context('spawn')
    # get free port
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    w0 = mp_context.Process(
      target=run_dist_feature_test,
      args=(2, 0, feature0, partition_book, port)
    )
    w1 = mp_context.Process(
      target=run_dist_feature_test,
      args=(2, 1, feature1, partition_book, port)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

if __name__ == "__main__":
    test_dist_feature_lookup()

