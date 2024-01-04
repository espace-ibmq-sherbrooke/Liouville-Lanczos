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
from math import pi

import numpy as np
import rustworkx as rx
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    HyperCubicLattice,
    Lattice,
    LatticeDrawStyle,
    LineLattice,
    SquareLattice,
    TriangularLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel

def Line_Hubbard_BaseProblem(t:float,mu:float,U:float,N:int,boundary_condition = BoundaryCondition.OPEN):
    num_nodes = N
    
    line_lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
    fhm = FermiHubbardModel(
    line_lattice.uniform_parameters(
        uniform_interaction=t,
        uniform_onsite_potential=-mu,
    ),
    onsite_interaction=U,
    )
    return fhm
def Line_Hubbard(t:float,mu:float,U:float,N:int,boundary_condition = BoundaryCondition.OPEN):
    return Line_Hubbard_BaseProblem(t,mu,U,N,boundary_condition).second_q_op()

