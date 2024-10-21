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
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, ParameterVector

import numpy as np
from qiskit.circuit.library.standard_gates import (
    IGate,
    XGate,
    YGate,
    ZGate,
    RXGate,
    RYGate,
    RZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    SwapGate,
    CXGate,
    CYGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CHGate,
)

# from qiskit_research.mzm_generation.utils import transpile_circuit
# from qiskit_research.utils.pulse_scaling import 
# from qiskit_research.utils.pulse_scaling import cr_scaling_passes
ecr = QuantumCircuit(2)
ecr.ecr(0,1)
def flat(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    return TwoLocal(nqbits,['rx','ry'],ecr,'pairwise',reps,name="[rx,ry],ecr,sca,reps={}".format(reps),parameter_prefix=parameter_prefix)
def SCA(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    return TwoLocal(nqbits,['rx','ry'],ecr,'sca',reps,name="[rx,ry],ecr,sca,reps={}".format(reps),parameter_prefix=parameter_prefix)


def two_rots_layer(ham,layers,reps=1,parameter_prefix:str =  "θ"):
    nqbits = ham.num_spin_orbitals
    L = np.sum([len(l) for l in layers])*2
    par = ParameterVector(parameter_prefix,4*L*(reps+1))
    circ = QuantumCircuit(nqbits)
    par_count = 0
    for i in range(reps):
        for layer in layers:
            for (qbit0,qbit1) in layer:
                circ.rx(par[par_count],qbit0)
                circ.ry(par[par_count+1],qbit0)
                par_count += 2
                circ.rx(par[par_count],qbit1)
                circ.ry(par[par_count+1],qbit1)
                par_count += 2
                circ.compose(ecr,[qbit0,qbit1],inplace=True)
    for qbit in range(nqbits):
        circ.rx(par[par_count],qbit)
        circ.ry(par[par_count+1],qbit)
        par_count += 2
    return circ

def two_rots_line(ham,reps=1,parameter_prefix:str =  "θ"):
    nqbits = ham.num_spin_orbitals
    layers = [(i,i+1) for i in range(nqbits-1)]
    return two_rots_layer(ham,[layers],reps,parameter_prefix)

def two_rots_flat(ham,reps=1, parameter_prefix: str = "θ"):
    nqbits = ham.num_spin_orbitals
    even = [ (i,i+1) for i in range(0,nqbits-1,2)]
    odd = [ (i,i+1) for i in range(1,nqbits-1,2)]
    return two_rots_layer(ham,[even,odd],reps,parameter_prefix) 
# %%

class S():
    def __init__(self,n:int):
        self.num_spin_orbitals = n
T = S(4)
#%%
# two_rots_flat(T,1).draw()
# #%%
# two_rots_line(T,1).draw()
# # %%
