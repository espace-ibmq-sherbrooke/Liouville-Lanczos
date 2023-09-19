"""
Componants for the Liouvillian recursion method on a Quantum computer using Qiskit.
The lanczos algorithm implementation meant to use these components is located in Lanczos.py
"""





from qiskit_nature.second_q import operators as op
from qiskit_nature.second_q.operators import commutators
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit_nature.second_q.mappers import QubitConverter
import numpy as np
from typing import Optional

from ..Lanczos_components import Inner_product,Summation
from ..Lanczos_components import Liouvillian as BaseLiouvillian

def relative_simplify(ope:op.SparseLabelOp,eps:float):
    v = max(ope.items(),key = lambda x: np.abs(x[1]))[1]
    return ope.simplify(eps*v)

    

class inner_product(Inner_product):
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimator,mapper:QubitConverter,epsilon:int = 1e-10):
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        self.mapper = mapper
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp,Name:Optional[str]):
        f = commutators.anti_commutator(A,B.adjoint())
        f = relative_simplify(f,self.eps)
        #imaginary contribution are necessarily error.
        try: #Add name to the list of tag for this job.
            if Name is not None:
                opt = self.estimator.options
                opt.environment.job_tags.append(Name)
                self.estimator.options=opt
        except:
            ...
        out = np.real(self.estimator.run(self.state,self.mapper.convert(f)).result().values[0])
        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                opt = self.estimator.options
                opt.environment.job_tags = opt.environment.job_tags[:-1]
                self.estimator.options=opt
        except:
            ...
        return out

class Liouvillian(BaseLiouvillian):

    def __init__(self,eps = 1e-10):
        self.eps = eps
    def __call__(self,H,A):
        comm = commutators.commutator(H,A)
        return relative_simplify(comm,self.eps)
    
class sum(Summation):
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,*X):
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify(A,self.eps)