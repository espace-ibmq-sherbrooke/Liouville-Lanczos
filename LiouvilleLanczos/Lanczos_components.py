"""
    Liouville-Lanczos: A library for Many-Body Green's function on quantum and classical computer.
    Copyright (C) 2024  Alexandre Foley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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