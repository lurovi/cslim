from __future__ import annotations
import random
from typing import Any, TypeVar
from collections.abc import MutableSequence
from copy import deepcopy

from cslim.cellular.NeighborsTopology import NeighborsTopology

T = TypeVar('T')


class TournamentTopology(NeighborsTopology):
    def __init__(self,
                 collection: MutableSequence[T],
                 clone: bool = False,
                 pressure: int = 3
                 ) -> None:
        super().__init__()
        if pressure < 1:
            raise ValueError(f'Pressure must be at least 1, found {pressure} instead.')
        self.__collection: MutableSequence[T] = deepcopy(collection) if clone else collection
        self.__pressure: int = pressure
        self.__size: int = len(self.__collection)

    def __hash__(self) -> int:
        molt: int = 31
        h: int = 0
        for s in self.__collection:
            h = h * molt + hash(s)
        return h
    
    def __str__(self) -> str:
        return str(self.__collection)
    
    def __repr__(self) -> str:
        return 'TournamentTopology(' + str(self) + ')'
    
    def __eq__(self, value: TournamentTopology) -> bool:
        if self.size() != value.size():
            return False
        for i in range(self.size()):
            if self.get((i,)) != value.get((i,)):
                return False
        return True
    
    def __len__(self) -> int:
        return self.__size

    def get_whole_collection(self, clone: bool = False) -> MutableSequence[T]:
        return deepcopy(self.__collection) if clone else self.__collection
    
    def get(self, indices: tuple[int, ...], clone: bool = False) -> T:
        if len(indices) != 1:
            raise ValueError(f'The length of indices must be 1, found {len(indices)} instead.')
        i: int = indices[0]
        self.__check_index(i)
        val: T = self.__collection[i]
        return deepcopy(val) if clone else val
    
    def set(self, indices: tuple[int, ...], val: T, clone: bool = False) -> T:
        if len(indices) != 1:
            raise ValueError(f'The length of indices must be 1, found {len(indices)} instead.')
        i: int = indices[0]
        self.__check_index(i)
        offset: int = i
        old_val: T = self.__collection[offset]
        old_val = deepcopy(old_val) if clone else old_val
        self.__collection[offset] = val
        return old_val

    def size(self) -> int:
        return self.__size
    
    def shape(self) -> tuple[int, ...]:
        return (self.__size,)

    def neighborhood(self, indices: tuple[int, ...], include_current_point: bool = True, clone: bool = False, distinct_coordinates: bool = False) -> MutableSequence[T]:
        result: MutableSequence[T] = []
        already_seen_coordinates: set[tuple[int, ...]] = set()
        for _ in range(self.__pressure):        
            new_ii: int = int(random.random()*self.size())
            while distinct_coordinates and (new_ii,) in already_seen_coordinates:
                new_ii = int(random.random()*self.size())
            result.append(self.get((new_ii,),clone=clone))
            already_seen_coordinates.add((new_ii,))
        return result

    def get_line_as_string(self) -> str:
        s: str = ''
        s += '['
        s += '\t'
        for i in range(self.size()):
            s += str(self.get((i,)))
            s += '\t'
        s += ']\n'
        return s

    def __check_index(self, i: int) -> None:
        if not 0 <= i < self.size():
            raise IndexError(f'Index {i} is out of range with declared size ({self.size()})')
