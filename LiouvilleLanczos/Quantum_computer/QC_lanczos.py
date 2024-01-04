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
    if isinstance(ope, op.FermionicOp):
        ope = ope.normal_order()
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
                tags = self.estimator.options.environment['job_tags']
                tags.append(Name)
                self.estimator.set_options(job_tags=tags)
        except:
            ...
        out = np.real(self.estimator.run(self.state,self.mapper.convert(f)).result().values[0])
        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                tags = self.estimator.options.environment['job_tags']
                tags = tags[:-1] # removes the appended name.
                self.estimator.set_options(job_tags=tags)
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