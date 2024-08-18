from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import MutableSequence
from copy import deepcopy

from slim.cellular.NeighborsTopology import NeighborsTopology

T = TypeVar('T')


class RowMajorCube(NeighborsTopology):
    def __init__(self,
                 collection: MutableSequence[T],
                 n_channels: int,
                 n_rows: int,
                 n_cols: int,
                 clone: bool = False,
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
        if len(collection) != self.__n_channels * self.__n_rows * self.__n_cols:
            raise ValueError(f'The length of the collection (found {len(collection)}) must match the product between number of channels ({self.__n_channels}), number of rows ({self.__n_rows}), and number of columns ({self.__n_cols}).')
        self.__collection: MutableSequence[T] = deepcopy(collection) if clone else collection
        self.__size_of_a_single_channel: int = self.n_rows() * self.n_cols()
        self.__radius: int = radius

    def __hash__(self) -> int:
        molt: int = 31
        h: int = 0
        for s in self.__collection:
            h = h * molt + hash(s)
        return h
    
    def __str__(self) -> str:
        return str(self.__collection)
    
    def __repr__(self) -> str:
        return 'RowMajorCube(' + str(self) + ')'
    
    def __eq__(self, value: RowMajorCube) -> bool:
        if self.n_channels() != value.n_channels():
            return False
        if self.n_rows() != value.n_rows():
            return False
        if self.n_cols() != value.n_cols():
            return False
        for l in range(self.n_channels()):
            for i in range(self.n_rows()):
                for j in range(self.n_cols()):
                    if self.get((l, i, j)) != value.get((l, i, j)):
                        return False
        return True
    
    def __len__(self) -> int:
        return self.n_channels() * self.n_rows() * self.n_cols()
    
    def n_channels(self) -> int:
        return self.__n_channels
    
    def n_rows(self) -> int:
        return self.__n_rows
    
    def n_cols(self) -> int:
        return self.__n_cols
    
    def get_whole_collection(self, clone: bool = False) -> MutableSequence[T]:
        return deepcopy(self.__collection) if clone else self.__collection
    
    def get(self, indices: tuple[int, ...], clone: bool = False) -> T:
        if len(indices) != 3:
            raise ValueError(f'The length of indices must be 3, found {len(indices)} instead.')
        l: int = indices[0]
        i: int = indices[1]
        j: int = indices[2]
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        val: T = self.__collection[l * self.__size_of_a_single_channel + i * self.n_cols() + j]
        return deepcopy(val) if clone else val
    
    def set(self, indices: tuple[int, ...], val: T, clone: bool = False) -> T:
        if len(indices) != 3:
            raise ValueError(f'The length of indices must be 3, found {len(indices)} instead.')
        l: int = indices[0]
        i: int = indices[1]
        j: int = indices[2]
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        offset: int = l * self.__size_of_a_single_channel + i * self.n_cols() + j
        old_val: T = self.__collection[offset]
        old_val = deepcopy(old_val) if clone else old_val
        self.__collection[offset] = val
        return old_val

    def size(self) -> int:
        return self.__n_channels * self.__n_rows * self.__n_cols
    
    def shape(self) -> tuple[int, ...]:
        return (self.__n_channels, self.__n_rows, self.__n_cols)

    def neighborhood(self, indices: tuple[int, ...], include_current_point: bool = True, clone: bool = False, distinct_coordinates: bool = False) -> MutableSequence[T]:
        if len(indices) != 3:
            raise ValueError(f'The length of indices must be 3, found {len(indices)} instead.')
        l: int = indices[0]
        i: int = indices[1]
        j: int = indices[2]
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        already_seen_coordinates: set[tuple[int, ...]] = set()
        result: MutableSequence[T] = []
        for ll in range(l - self.__radius, l + self.__radius + 1):
            for ii in range(i - self.__radius, i + self.__radius + 1):
                for jj in range(j - self.__radius, j + self.__radius + 1):
                    if ll == l and ii == i and jj == j:
                        if include_current_point:
                            current_coordinate: tuple[int, ...] = (ll,ii,jj)
                            if not distinct_coordinates or current_coordinate not in already_seen_coordinates: 
                                result.append(self.get(current_coordinate,clone=clone))
                                already_seen_coordinates.add(current_coordinate)
                    else:
                        new_ll: int = ll % self.n_channels()
                        new_ii: int = ii % self.n_rows()
                        new_jj: int = jj % self.n_cols()
                        
                        current_coordinate: tuple[int, ...] = (new_ll,new_ii,new_jj)
                        if not distinct_coordinates or current_coordinate not in already_seen_coordinates: 
                            result.append(self.get(current_coordinate,clone=clone))
                            already_seen_coordinates.add(current_coordinate)
        
        return result

    def get_cube_as_string(self) -> str:
        s: str = '[\n'

        for l in range(self.n_channels()):
            s += '['
            s += '\n'
            for i in range(self.n_rows()):
                s += '['
                s += '\t'
                for j in range(self.n_cols()):
                    s += str(self.get((l, i, j)))
                    s += '\t'
                s += ']\n'
            s += ']\n'
        s += ']\n'
        return s

    def __check_channel_index(self, l: int) -> None:
        if not 0 <= l < self.n_channels():
            raise IndexError(f'Index {l} is out of range with declared number of channels ({self.n_channels()})')
    
    def __check_row_index(self, i: int) -> None:
        if not 0 <= i < self.n_rows():
            raise IndexError(f'Index {i} is out of range with declared number of rows ({self.n_rows()})')
        
    def __check_col_index(self, j: int) -> None:
        if not 0 <= j < self.n_cols():
            raise IndexError(f'Index {j} is out of range with declared number of cols ({self.n_cols()})')
