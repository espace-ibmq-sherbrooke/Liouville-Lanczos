



from qiskit_nature.second_q import operators as op
from qiskit_nature.second_q.operators import commutators
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit_nature.second_q.mappers import QubitConverter
import numpy as np

from .QC_lanczos import relative_simplify

from qiskit_research.utils.convenience import add_pauli_twirls

class twirled_inner_product():
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimator,mapper:QubitConverter,twirls = 20,epsilon:int = 1e-10):
        self.estimator = estimator
        self.eps = epsilon
        self.mapper = mapper
        self.twirls = twirls
        self.state = add_pauli_twirls(state,twirls)
        
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp):
        f = commutators.anti_commutator(A,B.adjoint())
        f = relative_simplify(f,self.eps)
        vals = self.estimator.run(self.state,[self.mapper.convert(f)]*self.twirls).result().values
        return np.mean(vals)