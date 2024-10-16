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


def relative_simplify_spo(ope:SparsePauliOp,eps:float):
    """
    relative simplify truncates terms with a relative participation smaller than eps.
    Qiskit quantum info sparse pauli operator based implementation.
    performs much better than slo.
    """
    return ope.simplify(atol=1e-17,rtol=eps)

def separate_imag(op: SparsePauliOp):
    """
    split a non hermiation operator in two hermitian operator.
    op = A+iB
    input: op
    returns: A,B
    """
    coeffs_real = []
    coeffs_img = []
    for coeff in op.coeffs:
        coeffs_real.append(np.real(coeff))
        coeffs_img.append(np.imag(coeff))
    pauli_op_real = SparsePauliOp(op.paulis, coeffs_real).chop()
    pauli_op_imag = SparsePauliOp(op.paulis, coeffs_img).chop()
    return pauli_op_real, pauli_op_imag




class inner_product_spo(Base_inner_product):
    """
    Operator Inner product implementation for sparse pauli operators.
    Compute $\\bra{\\psi} \\rho \\{A,B^\\dagger \\} \\ket{\\psi}$
    where $\\ket{\\psi}$ is a quantum state specified by a quantum circuit.
    $A$ and $B$ are quantum operators specified by sparse pauli operators.
    Uses ibm runtime estimatorV2 interface to submit tasks to quantum computers.
    """
    def __init__(self, state: QuantumCircuit, estimator: BaseEstimatorV2, epsilon: int):
        """
        Constructor for the inner product.
        requires
        - state: a quantum circuit that produce a desired state.
        - estimator: an EstimatorV2 estimator to submit jobs to a quantum computer.
        - epsilon: target accuracy, smaller relative values are truncated. 
        """
        self.state = state
        self.estimator = estimator
        self.eps = epsilon
        
    def __call__(self,A:SparsePauliOp,B:SparsePauliOp,real_result:bool=False,Name:Optional[str]=None):
        """
        compute the innerproduct between that A and B operator.
        set real_result to True to bypass computation of an imaginary part in the result.
        Set a Name to easily identify the associated jobs on IBM quantum.
        """
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
    """
    Sparse pauli operator based implementation of the Liouvillian.
    """
    def __init__(self,eps = 1e-10):
        """
        initialise with a relative precision. operator with smaller coefficients 
        are truncated.
        """
        self.eps = eps
    def __call__(self,H,A):
        """
        Compute the result of the Liouvillian for system with Hamiltonian H on 
        operator A.
        """
        comm = H@A-A@H
        return relative_simplify_spo(comm,self.eps)

class sum_spo(Base_summation):
    """
    Sum up to 3 sparse pauli operator
    """
    def __init__(self,eps):
        """
        target relative precision of the results.
        """
        self.eps = eps
    def __call__(self,*X):
        """
        perform the sum
        """
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_spo(A,self.eps)

def relative_simplify_slo(ope:op.SparseLabelOp,eps:float):
    """
    relative simplify truncates terms with a relative participation smaller than eps.
    qiskit nature's sparse label op based implementation of relative simplify.
    Sparse label ops are very slow. Avoid.
    """
    v = max(ope.items(),key = lambda x: np.abs(x[1]))[1]
    if isinstance(ope, op.FermionicOp):
        ope = ope.normal_order()
    return ope.simplify(eps*v)

class inner_product_slo(Base_inner_product):
    """
    don't use this.
    """
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
    """
    dont' use this
    """
    def __init__(self,eps = 1e-10):
        self.eps = eps
    def __call__(self,H,A):
        comm = commutators.commutator(H,A)
        return relative_simplify_slo(comm,self.eps)
    
class sum_slo(Base_summation):
    """
    don't use this.
    """
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,*X):
        if len(X) > 2:
            A =  X[0]+X[1]+X[2]
        else:
            A = X[0]+X[1]
        return relative_simplify_slo(A,self.eps)