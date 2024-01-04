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
import logging
from typing import Sequence,Callable

from qiskit import QuantumCircuit

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.circuit.library import UCC

logger = logging.getLogger(__name__)


def generate_fermionic_pool(excitation_type: int, system_size, *bogus_crap):
    out = []
    n = [*range(2 * system_size)]

    def generate_tuples(partial_ex, n):
        out = []
        for part in partial_ex:
            for i in n[part[-1] + 1 :]:
                out.append([*part, i])
        return out

    partial = [[i] for i in n]
    for i in range(excitation_type - 1):
        partial = generate_tuples(partial, n)
    i = 0
    for a in partial:
        for b in partial[i:]:
            out.append((tuple(a), tuple(b)))
        i += 1
    return out


def generate_sd(num_spatial_orbitals, num_particles, *more_crap):
    return [
        *generate_fermionic_pool(1, num_spatial_orbitals),
        *generate_fermionic_pool(2, num_spatial_orbitals),
    ]


class Hubbard_EvolvingAnsatz(UCC):
    def __init__(
        self,
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        excitations: str
        | int
        | list[int]
        | Callable[
            [int, tuple[int, int]],
            list[tuple[tuple[int, ...], tuple[int, ...]]],
        ]
        = generate_sd,
        qubit_converter: QubitConverter | None = None,
        *,
        alpha_spin: bool = True,
        beta_spin: bool = True,
        max_spin_excitation: int | None = None,
        generalized: bool = False,
        preserve_spin: bool = True,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
    ):
        super().__init__(
            num_spatial_orbitals,
            num_particles,
            excitations,
            qubit_converter,
            alpha_spin= alpha_spin,
            beta_spin = beta_spin,
            max_spin_excitation=max_spin_excitation,
            generalized=generalized,
            preserve_spin=preserve_spin,
            reps=reps,
            initial_state=initial_state,
        )

    def _check_excitation_list(self, excitations: Sequence) -> None:
        """Checks the format of the given excitation operators.

        The following conditions are checked:
        - the list of excitations consists of pairs of tuples
        - each pair of excitation indices has the same length

        Args:
            excitations: the list of excitations

        Raises:
            QiskitNatureError: if format of excitations is invalid
        """
        logger.debug("Checking excitation list...")

        error_message = "{error} in the following UCC excitation: {excitation}"

        for excitation in excitations:
            if len(excitation) != 2:
                raise QiskitNatureError(
                    error_message.format(
                        error="Invalid number of tuples", excitation=excitation
                    )
                    + "; Two tuples are expected, e.g. ((0, 1, 4), (2, 3, 6))"
                )

            if len(excitation[0]) != len(excitation[1]):
                raise QiskitNatureError(
                    error_message.format(
                        error="Different number of occupied and virtual indices",
                        excitation=excitation,
                    )
                )
