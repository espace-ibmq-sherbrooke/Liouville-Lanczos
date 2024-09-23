# This should be moved out of Liouville-Lanczos. It has more to do with VQE, for 
# which  we do not have a dedicated library as of today.
# should leav with adapt_pool.py

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
#%%
import logging

from qiskit import QuantumCircuit,QiskitError
from typing import List,Optional
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import AdaptVQE,VQE,AdaptVQEResult
from qiskit.algorithms.minimum_eigensolvers.adapt_vqe import TerminationCriterion,estimate_observables,OperatorBase
from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.algorithms.list_or_dict import ListOrDict

from qiskit.quantum_info import SparsePauliOp,Pauli
from qiskit.opflow import PauliSumOp

logger = logging.getLogger(__name__)

##DEBUG TODO
import matplotlib.pyplot as plt
##DEBUG

class Adapt(AdaptVQE):
    """
    More modular than qiskit base implementation, also does away with some silly stuff.
    Customization points: 
        - _compute_averages: compute the averages fo the commutators for the determination of the next layer.
        - _compute_commutators: compute the relevent commutators of the pool operators with the optimized operator.
                                Called once at the start of compute_minimum_eigenvalue
        - _select_next_layer: call on _compute_averages to compute the average values with the current ansatz of the
                              relevent commutators to select the next layer to append to the ansatz
    I'm trying to work with what i'm given, so this modularity isn't perfect; there's some unpaid technical debt.
    _select_next_layer must be custimized with care, it sadly cannot assume a single responsability without a more significant changes.
                              
    The "silly stuff" is the cyclicity test. Look at _check_cyclicity for more details.
    """
    def __init__(self, solver: VQE, *, threshold: float = 0.00001, max_iterations: int | None = None) -> None:
        super().__init__(solver, threshold=threshold, max_iterations=max_iterations)

    @staticmethod
    def _check_cyclicity(indices: list[int]) -> bool:
        """
        Whenever this function return true adapt is interupted...
        The default implementation interupt as soon as an operator is used twice.
        Reusing the same operator multiple time, so long as it's not twice in a 
        row is perfectly valid. In effect, it induce evolution related to the 
        lie algebra of the pool.
        """
        if len(indices)>=2:
            return indices[-1] == indices[-2]
        else:
            return False
    def _compute_averages(
        self,
        theta: list[float],
        commutators: OperatorBase,
    ) -> list[tuple[complex, complex]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        # The excitations operators are applied later as exp(i*theta*excitation).
        # For this commutator, we need to explicitly pull in the imaginary phase.
        res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, theta)
        return res
        
    def _select_next_layer(self,theta,operator,prev_op_indices,iteration,commutators):
        logger.debug("Computing gradients")
        cur_grad= self._compute_averages(theta, commutators)
        # pick maximum gradient
        selected_operator_index, max_grad = max(
            enumerate(cur_grad), key=lambda item: np.abs(item[1][0])
        )
        # store maximum gradient's index for cycle detection
        prev_op_indices.append(selected_operator_index)
        
        interupt_iteration,TerminationCriterion = self._select_next_layer_logging(max_grad,selected_operator_index,iteration,prev_op_indices)
        
        self._excitation_list.append(self._excitation_pool[selected_operator_index])
        theta.append(0.0)
        return interupt_iteration,TerminationCriterion
        
    def _compute_commutators(self,operator):
        commutators = [1j * (operator @ exc - exc @ operator).reduce() for exc in self._excitation_pool]
        return commutators
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> AdaptVQEResult:
        """Computes the minimum eigenvalue.

        Args:
            operator: Operator whose minimum eigenvalue we want to find.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            TypeError: If an ansatz other than :class:`~.EvolvedOperatorAnsatz` is provided.
            QiskitError: If all evaluated gradients lie below the convergence threshold in the first
                iteration of the algorithm.

        Returns:
            An :class:`~.AdaptVQEResult` which is a :class:`~.VQEResult` but also but also
            includes runtime information about the AdaptVQE algorithm like the number of iterations,
            termination criterion, and the final maximum gradient.
        """
        if not isinstance(self.solver.ansatz, EvolvedOperatorAnsatz):
            raise TypeError("The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type.")

        # Overwrite the solver's ansatz with the initial state
        self._tmp_ansatz = self.solver.ansatz
        self._excitation_pool = self._tmp_ansatz.operators
        self.solver.ansatz = self._tmp_ansatz.initial_state

        commutators = self._compute_commutators(operator)
        prev_op_indices: list[int] = []
        theta: list[float] = []
        max_grad: tuple[float, Optional[PauliSumOp]] = (0.0, None)
        self._excitation_list = []
        history: list[float] = []
        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients
            do_break,termination_criterion = self._select_next_layer(theta,operator,prev_op_indices,iteration,commutators)
            if do_break:
                break
            # run VQE on current Ansatz
            self._tmp_ansatz.operators = self._excitation_list
            self.solver.ansatz = self._tmp_ansatz
            self.solver.initial_point = theta
            ##Debug TODO
            # theta2 = theta.copy()
            # T = np.linspace(0,2*np.pi,100)
            # Eigen = []
            # for t in T:
            #     theta2[-1] = t
            #     Eigen.append(self.solver.estimator.run(self.solver.ansatz,operator,theta2).result().values[0])
            # plt.plot(T,Eigen)
            print("WHY ", self.solver.estimator.run(self.solver.ansatz,operator,theta).result().values[0] )
            # print("selected operator {}, angle {}, energy: {}".format(self.solver.ansatz.operators,theta[-1],Eigen))
            ##debug end
            raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
            history.append(raw_vqe_result.eigenvalue)
            logger.info("Current eigenvalue: %s", str(raw_vqe_result.eigenvalue))
        else:
            # reached maximum number of iterations
            termination_criterion = TerminationCriterion.MAXIMUM
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.termination_criterion = termination_criterion
        result.eigenvalue_history = history

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            aux_values = estimate_observables(
                self.solver.estimator, self.solver.ansatz, aux_operators, result.optimal_point
            )
            result.aux_operators_evaluated = aux_values

        logger.info("The final energy is: %s", str(result.eigenvalue))
        self.solver.ansatz.operators = self._excitation_pool
        return result
    def _select_next_layer_logging(self,max_grad,selected_operator_index,iteration,prev_op_indices):
        logger.info(
            "Found maximum gradient %s at index %s",
            str(np.abs(max_grad[0])),
            str(selected_operator_index),
        )
        # log gradients
        if np.abs(max_grad[0]) < self.threshold:
            if iteration == 1:
                raise QiskitError(
                    "All gradients (maximum: {}) have been evaluated to lie below the convergence threshold "
                    "during the first iteration of the algorithm. Try to either tighten the "
                    "convergence threshold or pick a different ansatz.".format(max_grad[0])
                )
            logger.info(
                "AdaptVQE terminated successfully with a final maximum gradient: %s",
                str(np.abs(max_grad[0])),
            )
            termination_criterion = TerminationCriterion.CONVERGED
            return True,termination_criterion
        # check indices of picked gradients for cycles
        if self._check_cyclicity(prev_op_indices):
            logger.info("Alternating sequence found. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
            termination_criterion = TerminationCriterion.CYCLICITY
            return True,termination_criterion
        # add new excitation to self._ansatz
        logger.info(
            "Adding new operator to the ansatz: %s", str(self._excitation_pool[selected_operator_index])
        )
        return False,None


class qubitAdapt(Adapt):
    """
    qubit adapt, exploit the simplicity of the pool to make better layer selection
    """
    def __init__(self, solver: VQE, *, threshold: float = 0.00001, max_iterations: int | None = None) -> None:
        super().__init__(solver, threshold=threshold, max_iterations=max_iterations)
        for operator in solver.ansatz.operators:
            assert isinstance(operator,PauliSumOp), "all the operators in the pool must be PauliSumOp"
            assert len(operator)==1, "qubit adapt requires that all the pool operators are single pauli string"

    def _compute_commutators(self,operator):
        """
        compute both the first and second derivatives of each operator
        """
        self._hessian_ready = False # the full hessian is only compute if we fail to get out of a local minima with its diagonnal
        #We initialise this value to false here, because this function is called early in compute_minimum_eigenvalue, and it's topical
        commutators = [1j * (operator @ exc - exc @ operator).reduce() for exc in self._excitation_pool]
        double_commutator = [ (-2*exc@exc@operator + 2*exc@operator@exc).reduce() for exc in self._excitation_pool]
        return commutators,double_commutator
    
    def _compute_averages(
        self,
        theta: list[float],
        commutators: OperatorBase,
    ) -> list[tuple[complex, complex]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        # The excitations operators are applied later as exp(i*theta*excitation).
        # For this commutator, we need to explicitly pull in the imaginary phase.
        len_grad = len(commutators[0])
        res = estimate_observables(self.solver.estimator, self.solver.ansatz, [*commutators[0],*commutators[1]], theta)
        return res[:len_grad],res[len_grad:]

    def _compute_hessian_operators(self,operator):
        """
        Preliminary test show that computing the full hessian is not a good strategy to get us out of a pool local minima
        $$ Hessian_{ij} = \langle P_i H P_j + P_j H P_i - P_i P_j H - P_j P_i H \rangle $$
        where $\{P_i\}$ are the pool operator, and the operator being evolved is H.
        """
        default = PauliSumOp(SparsePauliOp("I"*(self._excitation_pool[0].num_qubits)))
        ps = len(self._excitation_pool)
        out = {}
        for i,exc1 in enumerate(self._excitation_pool):
            for j,exc2 in enumerate(self._excitation_pool[i:]):
                out[(i,j+i)] = exc1@operator@exc2 + exc2@operator@exc1 - exc1@exc2@operator - exc2@exc1@operator
        self._hessian_op = out

    def compute_hessian_aver(self,theta):
        """
        could skip identities...
        """
        return estimate_observables(self.solver.estimator,self.solver.ansatz,self._hessian_op,theta)
    
    def _select_next_layer(self,theta,operator,prev_op_indices,iteration,commutators):
        """
        because we know that the pool is composed of single Pauli strings, we know
        that each of them (when alone varying) induce an energy variation of the form E(t) = Acos(2t+a)+B
        Using the first and second derivative, we compute A and a, select the next operator
        based on the maximal possible gain when optimising only this operator.
        """
        logger.debug("Computing gradients")
        first_derivs, second_derivs = self._compute_averages(theta, commutators)
        first_derivs,mtdt1 = list(zip(*first_derivs))
        second_derivs,mtdt2 = list(zip(*second_derivs))
        first_derivs = np.array(first_derivs,dtype=np.float64)
        second_derivs = np.array(second_derivs,dtype=np.float64)
        a = np.arctan2(-2*first_derivs,-second_derivs)#signe première dérivé
        # A = -0.25*second_derivs/np.cos(a)#fails if we're at an inflexion point...
        # A = -0.5*first_derivs/np.sin(a)#fails if we're at an extrema...
        cosa = np.cos(a)
        # sina = np.sin(a)
        A = np.sqrt( (first_derivs*0.5)**2 + (second_derivs*0.25)**2)
        thetas = -0.5*(np.pi-a)#signe du temps
        Opt_point = np.cos(-2*thetas+a) #signe du temps
        delta = A*(-1 - cosa)
        # pick highest gain operator
        selected_operator_index, max_grad = min(
            enumerate(delta), key=lambda item: (item[1])
        )
        
        prev_op_indices.append(selected_operator_index)
        
        interupt_iteration,TerminationCriterion = self._select_next_layer_logging([max_grad],selected_operator_index,iteration,prev_op_indices)
        
        self._excitation_list.append(self._excitation_pool[selected_operator_index])
        theta.append(thetas[selected_operator_index]) #There's a bug 
        return interupt_iteration,TerminationCriterion
# %%
