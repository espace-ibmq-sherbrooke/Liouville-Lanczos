"""
Componants for the Liouvillian recursion method on a Quantum computer using Qiskit.
The lanczos algorithm implementation meant to use these components is located in Lanczos.py
"""





from qiskit_nature.second_q import operators as op
from qiskit_nature.second_q.operators import commutators
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit_nature.second_q.mappers import QubitConverter




class inner_product():
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimator,mapper:QubitConverter,epsilon:int = 1e-10):
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        self.mapper = mapper
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp):
        f = commutators.anti_commutator(A,B.adjoint())
        f = f.simplify(self.eps)
        return self.estimator.run(self.state,self.mapper.convert(f)).result().values[0]


def Liouvillian(H,A):
    return commutators.commutator(H,A)