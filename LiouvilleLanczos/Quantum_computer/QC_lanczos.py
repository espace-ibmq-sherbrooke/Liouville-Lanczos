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
from qiskit.primitives import BaseEstimatorV2
from qiskit_nature.second_q.mappers import QubitMapper
import numpy as np
from typing import Optional

from qiskit.transpiler import PassManager

from qiskit.quantum_info import SparsePauliOp

from ..Lanczos_components import Inner_product as Base_inner_product,Summation as Base_summation
from ..Lanczos_components import Liouvillian as BaseLiouvillian

def relative_simplify_slo(ope:op.SparseLabelOp,eps:float):
    v = max(ope.items(),key = lambda x: np.abs(x[1]))[1]
    if isinstance(ope, op.FermionicOp):
        ope = ope.normal_order()
    return ope.simplify(eps*v)

def relative_simplify_spo(ope:SparsePauliOp,eps:float):
    return ope.simplify(atol=1e-17,rtol=eps)

def separate_imag(op: SparsePauliOp):
    coeffs_real = []
    coeffs_img = []
    for coeff in op.coeffs:
        coeffs_real.append(np.real(coeff))
        coeffs_img.append(np.imag(coeff))
    pauli_op_real = SparsePauliOp(op.paulis, coeffs_real).chop()
    pauli_op_imag = SparsePauliOp(op.paulis, coeffs_img).chop()
    return pauli_op_real, pauli_op_imag




class inner_product_spo(Base_inner_product):
    def __init__(self, state: QuantumCircuit, estimator: BaseEstimatorV2, epsilon: int):
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        
    def __call__(self,A:SparsePauliOp,B:SparsePauliOp,real_result:bool=False,Name:Optional[str]=None):
        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)
        obs_real, obs_imag = separate_imag(f)
        # print(f"A={A}\nB={B}\nf={f}")
        #imaginary contribution are necessarily error.
        try: #Add name to the list of tag for this job.
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags.append(Name)
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        isa_obs_real = obs_real.apply_layout(self.state.layout)
        out = np.real(self.estimator.run([(self.state,isa_obs_real)]).result()[0].data.evs)
        if not real_result:
            isa_obs_imag = obs_imag.apply_layout(self.state.layout)
            out_imag = np.real(self.estimator.run([(self.state,isa_obs_imag)]).result()[0].data.evs)
            out += out_imag * 1j

        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags = tags[:-1] # removes the appended name.
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        return out[0]

class Liouvillian_spo(BaseLiouvillian):

    def __init__(self,eps = 1e-10):
        self.eps = eps
    def __call__(self,H,A):
        comm = H@A-A@H
        return relative_simplify_spo(comm,self.eps)

class sum_spo(Base_summation):
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,*X):
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_spo(A,self.eps)

class inner_product_slo(Base_inner_product):
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimatorV2, mapper:QubitMapper,epsilon:int = 1e-10):
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        self.mapper = mapper
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp,real_result:bool,Name:Optional[str]):
        f = commutators.anti_commutator(A,B.adjoint())
        f = relative_simplify_slo(f,self.eps)
        obs = self.mapper.map(f)
        #separate between real and imaginary observables
        obs_real, obs_imag = separate_imag(obs)
        try: #Add name to the list of tag for this job.
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags.append(Name)
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        #assume incoming circuit is transpiled
        isa_obs_real = obs_real.apply_layout(self.state.layout)
        out = np.real(self.estimator.run([(self.state,isa_obs_real)]).result()[0].data.evs)
        if not real_result:
            isa_obs_imag = obs_imag.apply_layout(self.state.layout)
            out_imag = np.real(self.estimator.run([(self.state,isa_obs_imag)]).result()[0].data.evs)
        out +=  out_imag * 1j
        
        try: #remove the name from the list of tags of the upcoming jobs
            if Name is not None:
                tags = self.estimator.options.environment.job_tags
                tags = tags[:-1] # removes the appended name.
                self.estimator.options.update(job_tags = tags)
        except:
            ...
        return out[0]

class Liouvillian_slo(BaseLiouvillian):

    def __init__(self,eps = 1e-10):
        self.eps = eps
    def __call__(self,H,A):
        comm = commutators.commutator(H,A)
        return relative_simplify_slo(comm,self.eps)
    
class sum_slo(Base_summation):
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,*X):
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_slo(A,self.eps)