from collections.abc import MutableSequence
from cslim.cellular.NeighborsTopology import NeighborsTopology
from cslim.cellular.TournamentTopology import TournamentTopology
from cslim.cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class TournamentTopologyFactory(NeighborsTopologyFactory):
    def __init__(self,
                 pressure: int = 3
                 ) -> None:
        super().__init__()
        if pressure < 1:
            raise ValueError(f'Pressure must be at least 1, found {pressure} instead.')
        self.__pressure: int = pressure

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return TournamentTopology(collection=collection, clone=clone, pressure=self.__pressure)
    
