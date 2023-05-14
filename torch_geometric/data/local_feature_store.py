from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType


class LocalFeatureStore(FeatureStore):
    def __init__(self,
        id2index: Optional[torch.Tensor] = None
        ):

        self.store: Dict[Tuple[str, str], Tensor] = {}

        # save the the mapping from global sampled node ids to local node ids
        self.id2index: Optional[torch.Tensor] = id2index

        super().__init__()

    @staticmethod
    def key(attr: TensorAttr) -> str:
        return (attr.group_name, attr.attr_name)

    def init_id2index(self, id2index: torch.Tensor):
        self.id2index = id2index

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        index = attr.index

        # None indices define the obvious index:
        if index is None:
            index = torch.arange(0, tensor.shape[0])

        # Store the index:
        self.store[LocalFeatureStore.key(attr)] = (index, tensor)

        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        index, tensor = self.store.get(LocalFeatureStore.key(attr), (None, None))
        if tensor is None:
            return None

        # None indices return the whole tensor:
        if attr.index is None:
            return tensor

        # Empty slices return the whole tensor:
        if (isinstance(attr.index, slice)
                and attr.index == slice(None, None, None)):
            return tensor

        #idx = (torch.cat([(index == v).nonzero() for v in attr.index]).view(-1)
        #       if attr.index.numel() > 0 else [])
        return tensor[attr.index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        del self.store[LocalFeatureStore.key(attr)]
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[str]:
        return [TensorAttr(*key) for key in self.store.keys()]

    def __len__(self):
        raise NotImplementedError
