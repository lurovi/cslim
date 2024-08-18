from collections.abc import MutableSequence
from slim.cellular.NeighborsTopology import NeighborsTopology
from slim.cellular.RowMajorLine import RowMajorLine
from slim.cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorLineFactory(NeighborsTopologyFactory):
    def __init__(self,
                 radius: int = 1
                 ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError(f'Radius must be at least 1, found {radius} instead.')
        self.__radius: int = radius

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorLine(collection=collection, clone=clone, radius=self.__radius)
    
