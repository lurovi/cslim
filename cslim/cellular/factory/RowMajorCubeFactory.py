from collections.abc import MutableSequence
from cslim.cellular.NeighborsTopology import NeighborsTopology
from cslim.cellular.RowMajorCube import RowMajorCube
from cslim.cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorCubeFactory(NeighborsTopologyFactory):
    def __init__(self,
                 n_channels: int,
                 n_rows: int,
                 n_cols: int,
                 radius: int = 1
                 ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError(f'Number of channels must be at least 1, found {n_channels} instead.')
        if n_rows < 1:
            raise ValueError(f'Number of rows must be at least 1, found {n_rows} instead.')
        if n_cols < 1:
            raise ValueError(f'Number of columns must be at least 1, found {n_cols} instead.')
        if radius < 1:
            raise ValueError(f'Radius must be at least 1, found {radius} instead.')
        self.__n_channels: int = n_channels
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols
        self.__radius: int = radius

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorCube(collection=collection, n_channels=self.__n_channels, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=clone, radius=self.__radius)
    
