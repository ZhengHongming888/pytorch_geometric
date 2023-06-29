
import unittest
import socket

import torch
import graphlearn_torch as glt

from torch_geometric.distributed import (
    RpcDataPartitionRouter, RpcCalleeBase, rpc_register, rpc_request_async
)

import torch_geometric.distributed as pyg_dist
import torch_geometric.distributed.rpc

from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed import DistFeature

def run_dist_feature_test(world_size: int, rank: int, feature: glt.data.Feature,
                          partition_book: glt.PartitionBook, master_port: int):
    #pass

    pyg_dist.init_worker_group(world_size, rank, 'dist-feature-test')
    pyg_dist.init_rpc(master_addr='localhost', master_port=master_port)

    partition2workers = pyg_dist.rpc_sync_data_partitions(world_size, rank)

    print(f"------rank={rank}------- partition2workers={partition2workers}")
    rpc_router = pyg_dist.RpcDataPartitionRouter(partition2workers)
    print(f"------rank={rank}------- rpc_router={rpc_router.get_to_worker(1)} ")

    current_device = torch.device('cuda', rank % 2)

    dist_feature = pyg_dist.DistFeature(
      world_size, rank, feature, partition_book,
      local_only=False, rpc_router=rpc_router
    )

    print("---------- done --------- ")

    r"""
    current_device = torch.device('cuda', rank % 2)

    dist_feature = glt.distributed.DistFeature(
      world_size, rank, feature, partition_book,
      local_only=False, rpc_router=rpc_router,
      device=current_device
    )

    input = torch.tensor(
      [10, 20, 260, 360, 200, 210, 420, 430],
      dtype=torch.int64,
      device=current_device
    )
    expected_features = torch.cat([
      torch.ones(2, 1024, dtype=torch.float32, device=current_device),
      torch.zeros(2, 1024, dtype=torch.float32, device=current_device),
      torch.ones(2, 1024, dtype=torch.float32, device=current_device)*2,
      torch.zeros(2, 1024, dtype=torch.float32, device=current_device)
    ])
    res = dist_feature[input]

    tc = unittest.TestCase()
    tc.assertTrue(glt.utils.tensor_equal_with_device(res, expected_features))

    glt.distributed.shutdown_rpc()
    """


def test_dist_feature_lookup():

    cpu_tensor0 = torch.cat([
      torch.ones(128, 1024, dtype=torch.float32),
      torch.ones(128, 1024, dtype=torch.float32)*2
    ])
    cpu_tensor1 = torch.cat([
      torch.zeros(128, 1024, dtype=torch.float32),
      torch.zeros(128, 1024, dtype=torch.float32)
    ])

    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128*2

    #print(f"----------- global_id0={global_id0}, global_id1={global_id1}  ")
    #print(f"--------cpu_tensor0={cpu_tensor0}, id2index={id2index}  ")

    partition_book = torch.cat([
      torch.zeros(128*2, dtype=torch.long),
      torch.ones(128*2, dtype=torch.long)
    ])


    feature0 = LocalFeatureStore()
    feature0.put_global_id(global_id0, group_name=None)
    feature0.put_tensor(cpu_tensor0, group_name=None, attr_name='x')

    feature1 = LocalFeatureStore()
    feature1.put_global_id(global_id1, group_name=None)
    feature1.put_tensor(cpu_tensor1, group_name=None, attr_name='x')

    #out = store.get_tensor_from_global_id(group_name='paper', attr_name='feat',
    #                                      index=torch.tensor([3, 8, 4]))

    mp_context = torch.multiprocessing.get_context('spawn')
    # get free port
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    #port = glt.utils.get_free_port()
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

