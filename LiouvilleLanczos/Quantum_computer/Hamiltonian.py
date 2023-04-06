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

