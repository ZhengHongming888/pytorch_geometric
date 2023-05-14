from dataclasses import dataclass

import pytest
import torch

from torch_geometric.data import TensorAttr
from torch_geometric.data.feature_store import AttrView, _field_status
from torch_geometric.data import LocalFeatureStore


@dataclass
def test_feature_store():
    store = LocalFeatureStore()
    tensor = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    group_name = 'A'
    attr_name = 'feat'
    index = torch.tensor([0, 1, 2])
    attr = TensorAttr(group_name, attr_name, index)
    assert TensorAttr(group_name).update(attr) == attr

    # Normal API:
    store.put_tensor(tensor, attr)
    assert torch.equal(store.get_tensor(attr), tensor)
    assert torch.equal(
        store.get_tensor(group_name, attr_name, index=torch.tensor([0, 2])),
        tensor[torch.tensor([0, 2])],
    )

    assert store.update_tensor(tensor + 1, attr)
    assert torch.equal(store.get_tensor(attr), tensor + 1)

    store.remove_tensor(attr)
    with pytest.raises(KeyError):
        _ = store.get_tensor(attr)

    # Views:
    view = store.view(group_name=group_name)
    view.attr_name = attr_name
    view['index'] = index
    assert view != "not a 'AttrView' object"
    assert view == AttrView(store, TensorAttr(group_name, attr_name, index))
    assert str(view) == ("AttrView(store=LocalFeatureStore(), "
                         "attr=TensorAttr(group_name='A', attr_name='feat', "
                         "index=tensor([0, 1, 2])))")

    # Indexing:
    store[group_name, attr_name, index] = tensor

    # Fully-specified forms, all of which produce a tensor output
    assert torch.equal(store[group_name, attr_name, index], tensor)
    assert torch.equal(store[group_name, attr_name, None], tensor)
    assert torch.equal(store[group_name, attr_name, :], tensor)
    assert torch.equal(store[group_name][attr_name][:], tensor)
    assert torch.equal(store[group_name].feat[:], tensor)
    assert torch.equal(store.view().A.feat[:], tensor)

    with pytest.raises(AttributeError) as exc_info:
        _ = store.view(group_name=group_name, index=None).feat.A
        print(exc_info)

    # Partially-specified forms, which produce an AttrView object
    assert store[group_name] == store.view(TensorAttr(group_name=group_name))
    assert store[group_name].feat == store.view(
        TensorAttr(group_name=group_name, attr_name=attr_name))

    # Partially-specified forms, when called, produce a Tensor output
    # from the `TensorAttr` that has been partially specified.
    store[group_name] = tensor
    assert isinstance(store[group_name], AttrView)
    assert torch.equal(store[group_name](), tensor)

    # Deletion:
    del store[group_name, attr_name, index]
    with pytest.raises(KeyError):
        _ = store[group_name, attr_name, index]
    del store[group_name]
    with pytest.raises(KeyError):
        _ = store[group_name]()

def test_feature_lookup_by_id2index():

    # this test can verify the feature lookup based on the
    # ID mapping bewteen global ID and local ID.

    store = LocalFeatureStore()

    group_name = 'part1'
    attr_name = 'node_feat'

    whole_feat_data= torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2],[3, 3, 3], [4, 4, 4], [5, 5, 5],[6, 6, 6], [7, 7, 7], [8, 8, 8]])
    # for part1 node ids
    ids_part1 = torch.Tensor([1, 2, 3, 5, 8, 4])
    part1_feat_data = whole_feat_data[ids_part1]

    index = torch.arange(0, part1_feat_data.size(0), 1)
    attr = TensorAttr(group_name, attr_name, index)


    # construct ip mapping between global ids and local index
    max_id = torch.max(ids_part1).item()
    id2idx = torch.zeros(max_id + 1, dtype=torch.int64)
    id2idx[ids_part1] = torch.arange(ids_part1.size(0), dtype=torch.int64)

    # put the id mapping into feature store for future lookup
    if id2idx is not None:
        store.init_id2index(id2idx)

    # put the feature in rows [1, 2, 3, 5, 8, 4] into part 1 feature store with increasing order [0, 1, 2 ... 5]
    store.put_tensor(part1_feat_data, attr)


    # lookup the features by global ids like [3, 8, 4]
    local_ids = torch.tensor([3, 8, 4])

    assert torch.equal(store.get_tensor(group_name, attr_name, index=store.id2index[local_ids]), torch.Tensor([[3, 3, 3], [8, 8, 8], [4, 4, 4]]))
