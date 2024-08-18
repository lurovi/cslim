from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from typing import TypeVar

from slim.cellular.NeighborsTopology import NeighborsTopology

T = TypeVar('T')


class NeighborsTopologyFactory(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def class_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        pass
