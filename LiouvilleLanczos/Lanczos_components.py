from abc import ABC,abstractmethod
from typing import Any, Optional

class Inner_product(ABC):
    @abstractmethod
    def __call__(self,A,B,Name:Optional[str]):
        ...

class Liouvillian(ABC):
    @abstractmethod
    def __call__(self,Hamiltonian,Operator):
        ...

class Summation(ABC):
    @abstractmethod
    def __call__(self,*Summands):
        ...