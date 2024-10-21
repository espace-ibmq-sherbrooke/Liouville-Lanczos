# This should be moved out of Liouville-Lanczos. It has more to do with VQE, for 
# which  we do not have a dedicated library as of today.
# should leav with Adapt.py

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

from qiskit.quantum_info import SparsePauliOp,Pauli
from qiskit.opflow import PauliSumOp

VTwoQbitPool = [SparsePauliOp(['ZY']),SparsePauliOp(['YI'])]

def Vqubit_pool_impl(Nqubits):
    """
    'V' pool from http://arxiv.org/abs/1911.10205
    """
    assert(Nqubits>=2)
    if Nqubits == 2:
        return VTwoQbitPool
    else:
        Z = SparsePauliOp(['Z'])
        YI = SparsePauliOp(['Y'+'I'*(Nqubits-1)])
        IY = SparsePauliOp(['IY'+'I'*(Nqubits-2)])
        return [*[Z.tensor(P) for P in Vqubit_pool_impl(Nqubits-1)],YI,IY]
def Vqubit_pool(Nqubits):
    return [PauliSumOp(o) for o in Vqubit_pool_impl(Nqubits)]

def Gqubit_pool_impl(Nqubits):
    """
    'G' pool from http://arxiv.org/abs/1911.10205
    Contain only first neighbhour terms.
    It's adaptive and hardware efficient. Remains to be seen if it's model efficient enough.
    Has a lot of potential to play nice with scaled ECR!
    """
    if Nqubits == 2:
        return VTwoQbitPool
    ZY = VTwoQbitPool[0]
    YI = VTwoQbitPool[1]
    def pad_with_I(op):
        return [ op.tensor(SparsePauliOp('I'*(Nqubits-2))), *[ SparsePauliOp('I'*n).tensor(op.tensor(SparsePauliOp('I'*(Nqubits-2-n)))) for n in range(1,Nqubits-2)],SparsePauliOp('I'*(Nqubits-2)).tensor(op) ]
    IZYIS = pad_with_I(ZY)
    IYIIS = pad_with_I(YI)
    return [*IZYIS,*IYIIS]
def Gqubit_pool(Nqubits):
    return [PauliSumOp(o) for o in Gqubit_pool_impl(Nqubits)]
    

def Connectivity_Gpool(qubit_connectivity):
    """
    Produce an overcomplete Gpool based on the supplied qubit connectivity.
    It should be more hardware efficient to consider all native 2qubit gates
    """
    raise NotImplementedError()
