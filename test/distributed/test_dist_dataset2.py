import os.path as osp

import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import Partitioner
from torch_geometric.typing import EdgeTypeStr
from torch_geometric.distributed import DistDataset

def test_dist_dataset():
    num_partitions = 2
    save_dir = osp.join('./homo', 'partition')
    dataset = FakeDataset()
    data = dataset[0]

    partitioner = Partitioner(root=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()

    data_pidx=0
    dataset0 = DistDataset()
    dataset0.load(
        root_dir=osp.join('./homo', 'partition'),
        partition_idx=data_pidx,
    )
    data_pidx=1
    dataset1 = DistDataset()
    dataset1.load(
        root_dir=osp.join('./homo', 'partition'),
        partition_idx=data_pidx,
    )

    assert data.num_nodes==dataset0.data.num_nodes+dataset1.data.num_nodes
    
    graph_store1 = dataset0.graph
    graph_store2 = dataset1.graph
    attr1 = graph_store1.get_all_edge_attrs()
    graph1 = graph_store1.get_edge_index(attr1[0])
    attr2 = graph_store2.get_all_edge_attrs()
    graph2 = graph_store2.get_edge_index(attr2[0])
    assert graph1[0].size(0) + graph2[0].size(0) == data.num_edges


    feature_store1 = dataset0.node_features
    node_attrs1 = feature_store1.get_all_tensor_attrs()
    node_feat1 = feature_store1.get_tensor(node_attrs1[0])
    node_id1 = feature_store1.get_global_id(node_attrs1[0])
    feature_store2 = dataset1.node_features
    node_attrs2 = feature_store2.get_all_tensor_attrs()
    node_feat2 = feature_store2.get_tensor(node_attrs2[0])
    node_id2 = feature_store2.get_global_id(node_attrs2[0])
    assert node_feat1.size(0) + node_feat2.size(0) == data.num_nodes
    assert torch.allclose(data.x[node_id1], node_feat1)
    assert torch.allclose(data.x[node_id2], node_feat2)




