from collections.abc import MutableSequence
from slim.cellular.NeighborsTopology import NeighborsTopology
from slim.cellular.RowMajorMatrix import RowMajorMatrix
from slim.cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorMatrixFactory(NeighborsTopologyFactory):
    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 radius: int = 1
                 ) -> None:
        super().__init__()
        if n_rows < 1:
            raise ValueError(f'Number of rows must be at least 1, found {n_rows} instead.')
        if n_cols < 1:
            raise ValueError(f'Number of columns must be at least 1, found {n_cols} instead.')
        if radius < 1:
            raise ValueError(f'Radius must be at least 1, found {radius} instead.')
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols
        self.__radius: int = radius

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorMatrix(collection=collection, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=clone, radius=self.__radius)
    
