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



import pauliarray as pa
from qiskit import QuantumCircuit

# import pauliarray.

Op = pa.PauliOperator


class inner_product():
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimator,mapper:QubitConverter,epsilon:int = 1e-10):
        ...
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp):
        ...

class Liouvillian():

    def __init__(self,eps = 1e-10):
        ...
    def __call__(self,H,A):
        ...
    
class sum():
    def __init__(self,eps):
        ...
    def __call__(self,*X):
        ...