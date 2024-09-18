from abc import abstractmethod, ABC
from collections.abc import MutableSequence
from typing import TypeVar

T = TypeVar('T')


class NeighborsTopology(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def class_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def get(self, indices: tuple[int, ...], clone: bool = False) -> T:
        pass

    @abstractmethod
    def set(self, indices: tuple[int, ...], val: T, clone: bool = False) -> T:
        pass

    @abstractmethod
    def neighborhood(self, indices: tuple[int, ...], include_current_point: bool = True, clone: bool = False, distinct_coordinates: bool = False) -> MutableSequence[T]:
        pass
