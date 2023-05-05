


from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp,Pauli
import numpy as np

class JordanWignerlayout(JordanWignerMapper): 
    def __init__(self,layout,nqbits):
        self.nq = nqbits
        self.layout = np.array(layout,dtype=np.int64)
        super().__init__()

    def apply_layout(self,op:PauliSumOp) -> PauliSumOp:
        """ transform the operator from the logical layout to the physical layout"""
        out = SparsePauliOp('I'*self.nq,0)
        for x in op.primitive:
            LOop = Pauli('I'*self.nq)
            LOop[self.layout] = x.paulis[0] #assume only one pauli per terms... should be ok. SparsePO return a sparsePO with a single element when indexed
            out += SparsePauliOp(LOop,x.coeffs)
        return out
    def map(self, second_q_op: FermionicOp) -> PauliSumOp:
        SUMOP = super().map(second_q_op)
        return self.apply_layout(SUMOP)