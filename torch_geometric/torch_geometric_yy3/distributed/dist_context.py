
from enum import Enum
from typing import Optional


class DistRole(Enum):
    r""" Role types for distributed groups."""
    WORKER = 1  # As worker for worker mode
    SERVER = 2  # As server for server-client mode 
    CLIENT = 3  # As client for server-client mode 


_DEF_WORKER_GRP_NAME = '_def_worker'
_DEF_SERVER_GRP_NAME = '_def_server'
_DEF_CLIENT_GRP_NAME = '_def_client'


class DistContext(object):
    r""" Context info for distributed Info in the current process.
      role (DistRole): The role for the current role group.
      rank (int): The current process rank within current role group.
      group_name (str): A unique name for current role group.
      world_size (int): The number of processes in current role group.
      global_world_size (int): The total number of processes in all role groups.
      global_rank (int): The current process rank within all role groups. """
    def __init__(self,
                 role: DistRole,
                 rank: int,
                 group_name: str,
                 world_size: int,
                 global_world_size: int,
                 global_rank: int):
      assert world_size > 0 and rank in range(world_size)
      assert global_world_size > 0 and global_rank in range(global_world_size)
      assert world_size <= global_world_size
      self.role = role
      self.rank = rank
      self.group_name = group_name
      self.world_size = world_size
      self.global_world_size = global_world_size
      self.global_rank = global_rank
    
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = []
        for key, value in self.__dict__.items():
            info.append(f"{key}: {value}")
        info = ", ".join(info)
        return f"{cls}({info})"
    
    def __eq__(self, obj):
        if not isinstance(obj, DistContext):
            return False
        for key, value in self.__dict__.items():
            if value != obj.__dict__[key]:
                return False
        return True
    
    def is_worker(self) -> bool:
        return self.role == DistRole.WORKER
    
    @property
    def worker_name(self) -> str:
        r""" Get worker name of the current process of this context."""
        return f"{self.group_name}-{self.rank}"


_dist_context: DistContext = None
r""" Setup the distributed context within one process.
"""


def get_context() -> DistContext:
    r""" Get distributed context info. """
    return _dist_context


def init_worker_group(world_size: int, rank: int,
                      group_name: Optional[str] = None):
    r""" Initialize one RPC group in worker mode without server/client roles.
        world_size (int): all processe number participating in one distributed
        worker group.
        rank (int): Rank index within one distributed group (it
        should be a number between 0 and ``world_size``-1).
        group_name (str): A unique name of the distributed group """
    
    global _dist_context
    _dist_context = DistContext(
      role=DistRole.WORKER,
      rank=rank,
      group_name=(group_name if group_name is not None
                  else _DEF_WORKER_GRP_NAME),
      world_size=world_size,
      global_world_size=world_size,
      global_rank=rank
    )

