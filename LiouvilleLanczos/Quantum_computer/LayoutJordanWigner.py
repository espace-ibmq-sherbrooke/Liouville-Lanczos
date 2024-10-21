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



from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp,Pauli
import numpy as np

class JordanWignerlayout(JordanWignerMapper):
    """
    superseded by facilities of runtime v2
    """
    def __init__(self,layout,nqbits):
        self.nq = nqbits
        self.layout = np.array(layout,dtype=np.int64)
        super().__init__()

    def apply_layout(self,op:SparsePauliOp) -> SparsePauliOp:
        """ transform the operator from the logical layout to the physical layout"""
        out = SparsePauliOp('I'*self.nq,0)
        for x in op.primitive:
            LOop = Pauli('I'*self.nq)
            LOop[self.layout] = x.paulis[0] #assume only one pauli per terms... should be ok. SparsePO return a sparsePO with a single element when indexed
            out += SparsePauliOp(LOop,x.coeffs)
        return out
    def map(self, second_q_op: FermionicOp) -> SparsePauliOp:
        SUMOP = super().map(second_q_op)
        return self.apply_layout(SUMOP)