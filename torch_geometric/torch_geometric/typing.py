import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

WITH_PT2 = int(torch.__version__.split('.')[0]) >= 2

try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_GMM = WITH_PT2 and hasattr(pyg_lib.ops, 'grouped_matmul')
    WITH_SAMPLED_OP = hasattr(pyg_lib.ops, 'sampled_add')
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
                      f"Disabling its usage. Stacktrace: {e}")
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_GMM = False
    WITH_SAMPLED_OP = False
    WITH_INDEX_SORT = False

try:
    import torch_scatter  # noqa
    WITH_TORCH_SCATTER = True
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
                      f"Disabling its usage. Stacktrace: {e}")
    torch_scatter = object
    WITH_TORCH_SCATTER = False

try:
    import torch_sparse  # noqa
    from torch_sparse import SparseTensor
    WITH_TORCH_SPARSE = True
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
                      f"Disabling its usage. Stacktrace: {e}")
    torch_sparse = object
    WITH_TORCH_SPARSE = False

    class SparseTensor:
        def __init__(self, *args, **kwargs):
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        @classmethod
        def from_edge_index(cls, *args, **kwargs) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")


# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

DEFAULT_REL = 'to'
EDGE_TYPE_STR_SPLIT = '__'


class EdgeTypeStr(str):
    r"""A helper class to construct serializable edge types by merging an edge
    type tuple into a single string."""
    def __new__(cls, *args):
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            args = args[0]  # An edge type string was passed.

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            args = (args[0], DEFAULT_REL, args[1])
            args = EDGE_TYPE_STR_SPLIT.join(args)

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            args = EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        return str.__new__(cls, args)

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        out = tuple(self.split(EDGE_TYPE_STR_SPLIT))
        if len(out) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return out


# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# A representation of a feature tensor
FeatureTensorType = Union[Tensor, np.ndarray]

# A representation of an edge index, following the possible formats:
#   * COO: (row, col)
#   * CSC: (row, colptr)
#   * CSR: (rowptr, col)
EdgeTensorType = Tuple[Tensor, Tensor]

# Types for message passing ###################################################

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

MaybeHeteroNodeTensor = Union[Tensor, Dict[NodeType, Tensor]]
MaybeHeteroEdgeTensor = Union[Tensor, Dict[EdgeType, Tensor]]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]






from typing import Dict, List, NamedTuple, Optional, Tuple, Union


############## GLT ..
# A representation of tensor data
#TensorDataType = Union[torch.Tensor, np.ndarray]

# Types for basic graph entity #################################################

# Node-types are denoted by a single string
NodeType = str

# Edge-types are denotes by a triplet of strings.
EdgeType = Tuple[str, str, str]

EDGE_TYPE_STR_SPLIT = '__'

def as_str(type: Union[NodeType, EdgeType]) -> str:
  if isinstance(type, NodeType):
    return type
  elif isinstance(type, (list, tuple)) and len(type) == 3:
    return EDGE_TYPE_STR_SPLIT.join(type)
  return ''

def reverse_edge_type(etype: EdgeType):
  src, edge, dst = etype
  if not src == dst:
    if edge.split("_", 1)[0] == 'rev': # undirected edge with `rev_` prefix.
      edge = edge.split("_", 1)[1]
    else:
      edge = 'rev_' + edge
  return (dst, edge, src)

# A representation of tensor data
TensorDataType = Union[torch.Tensor, np.ndarray]

# Types for partition data #####################################################

class GraphPartitionData(NamedTuple):
  r""" Data and indexing info of a graph partition.
  """
  # edge index (rows, cols)
  edge_index: Tuple[torch.Tensor, torch.Tensor]
  # edge ids tensor corresponding to `edge_index`
  eids: torch.Tensor

class FeaturePartitionData(NamedTuple):
  r""" Data and indexing info of a node/edge feature partition.
  """
  # node/edge feature tensor
  feats: torch.Tensor
  # node/edge ids tensor corresponding to `feats`
  ids: torch.Tensor
  # feature cache tensor
  cache_feats: Optional[torch.Tensor]
  # cached node/edge ids tensor corresponding to `cache_feats`
  cache_ids: Optional[torch.Tensor]

HeteroGraphPartitionData = Dict[EdgeType, GraphPartitionData]
HeteroFeaturePartitionData = Dict[Union[NodeType, EdgeType], FeaturePartitionData]

# Types for partition book #####################################################

PartitionBook = torch.Tensor
HeteroNodePartitionDict = Dict[NodeType, PartitionBook]
HeteroEdgePartitionDict = Dict[EdgeType, PartitionBook]

# Types for neighbor sampling ##################################################

InputNodes = Union[torch.Tensor, NodeType, Tuple[NodeType, torch.Tensor]]
EdgeIndexTensor = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
InputEdges = Union[EdgeIndexTensor, EdgeType, Tuple[EdgeType, EdgeIndexTensor]]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

