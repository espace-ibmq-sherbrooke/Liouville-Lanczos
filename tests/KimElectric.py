# %%
import importlib
#importlib.reload(module)

import qiskit
import numpy as np
from qiskit import qpy
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.providers.ibmq import AccountProvider
from qiskit.primitives import Estimator as exactEstimator
from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator,AerProvider
from qiskit_ibm_runtime import (
    Session,
    Sampler,
    QiskitRuntimeService,
    Options,
    Estimator,
)
from qiskit_ibm_runtime.options import (
    ExecutionOptions,
    EnvironmentOptions,
    ResilienceOptions,
    TranspilationOptions,
    
)
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    Optimize1qGates,
    Optimize1qGatesSimpleCommutation,
    Optimize1qGatesDecomposition,
    RemoveBarriers,
    LayoutTransformation,
    RZXCalibrationBuilder,
    EchoRZXWeylDecomposition,
    NoiseAdaptiveLayout,
    ApplyLayout,
    RZXCalibrationBuilderNoEcho,
)
from qiskit.transpiler.passmanager import PassManager


from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.operators import FermionicOp
from LiouvilleLanczos.Quantum_computer.Hamiltonian import (
    Line_Hubbard,
    BoundaryCondition,
    Line_Hubbard_BaseProblem,
)
from LiouvilleLanczos.Quantum_computer import Mapping
from qiskit_research.utils.convenience import scale_cr_pulses,add_pauli_twirls,add_dynamical_decoupling

qubit_converter = QubitConverter(JordanWignerMapper())
doubble_occup = FermionicOp(
    {
        "+_0 +_4 -_4 -_0": 1,
        "+_1 +_5 -_5 -_1": 1,
        "+_2 +_6 -_6 -_2": 1,
        "+_3 +_7 -_7 -_3": 1,
    },
    num_spin_orbitals=8,
)
first_hop = FermionicOp(
    {
        "+_0 -_1": 1,
        "+_1 -_0": 1,
        "+_1 -_2": 1,
        "+_2 -_1": 1,
        "+_2 -_3": 1,
        "+_3 -_2": 1,
        "+_4 -_5": 1,
        "+_5 -_4": 1,
        "+_5 -_6": 1,
        "+_6 -_5": 1,
        "+_6 -_7": 1,
        "+_7 -_6": 1,
    },
    num_spin_orbitals=8,
)
Number_op = FermionicOp(
    {
    '+_0 -_0':1,
    '+_1 -_1':1,
    '+_2 -_2':1,
    '+_3 -_3':1,
    '+_4 -_4':1,
    '+_5 -_5':1,
    '+_6 -_6':1,
    '+_7 -_7':1,
    },
    num_spin_orbitals=8
    )
t = -1
U = 4
mu = U/2
lineH = qubit_converter.map(Line_Hubbard(t,mu,U,4))
Hubbard_FOP = t*first_hop-mu*Number_op+U*doubble_occup
Hubbard_FOP
HAM = qubit_converter.map(Hubbard_FOP)
Hubbard_matrix = HAM.to_matrix()
E,S = np.linalg.eigh(Hubbard_matrix)
EE,SS = np.linalg.eigh(lineH.to_matrix())
print(E[0],EE[0])
# %%
service_algolab = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-qida/iq-quantum/algolab"
)
#%%
from typing import Optional
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
# %%
with open("FH1x4_ground_state_1.qpy", "rb") as handle:
    HS1 = qpy.load(handle)[0]
with open("FH1x4_ground_state_2.qpy", "rb") as handle:
    HS2 = qpy.load(handle)[0]
with open("FH1x4_ground_state_3_initSlater.qpy", "rb") as handle:
    SS1 = qpy.load(handle)[0]
with open("FH1x4_ground_state_4_initSlater.qpy", "rb") as handle:
    SS2 = qpy.load(handle)[0]

SS1 = RemoveBarriers()(SS1)
HS1 = RemoveBarriers()(HS1)
HS2 = RemoveBarriers()(HS2)
SS2 = RemoveBarriers()(SS2)
eesti = exactEstimator()


#%%
"""
Vérification que les circuit de Camille sont exprimé dans la base du hamiltonien
que j'ai construit.
"""
print(eesti.run(HS1,HAM).result().values[0])
print(eesti.run(HS2,HAM).result().values[0])
print(eesti.run(SS1,HAM).result().values[0])
E_SS1 = eesti.run(SS1,HAM).result().values[0]
print(eesti.run(SS2,HAM).result().values[0])


# %%
from qiskit.circuit.library import standard_gates as sg

C = QuantumCircuit(3)

C.x(0)

A = sg.ECRGate

# %%
Sherbrooke = backend = service_algolab.backend("ibm_sherbrooke")
mtl = backend = service_algolab.backend("ibmq_kolkata")
# %%
conf = Sherbrooke.configuration()
# %%
conf
# %%
pm_2 = generate_preset_pass_manager(optimization_level=2, backend=mtl)
OnequbitPass = PassManager(
    [
        Optimize1qGatesSimpleCommutation(),
        Optimize1qGatesDecomposition(),
        Optimize1qGates(),
    ]
)
# %%
# SS1_l = pm.layout.run()
# %%
SS1_1q = OnequbitPass.run(SS1)
# %%
"""Check that those transpilation passes didn't do anything dumb"""
print(eesti.run(SS1_1q,HAM).result().values[0])
#%%

SS1_1q_m = SS1_1q.copy()
SS1_1q_m.measure_all()
scored_m,m_def = Mapping.optimize_init_layout(SS1_1q_m,mtl,10,level=3,seed=1123123)#level 2 give wrong results?
m_def.remove_final_measurements()

options = Options()
# options.transpilation.initial_layout = scored_m
with Session(backend=Sherbrooke) as session:
    options.environment.job_tags = ["resil0"]
    options.resilience_level=0
    mtl_estimator = Estimator(session=session,options=options)
    mtl_estim_result_resil0 = mtl_estimator.run(SS1_1q,HAM)
    options.environment.job_tags = ["resil1"]
    options.resilience_level=1
    mtl_estimator = Estimator(session=session,options=options)
    mtl_estim_result_resil1 = mtl_estimator.run(SS1_1q,HAM)
    # options.environment.job_tags = ["resil2"]
    # options.resilience_level=2
    # q_estimator = Estimator(session=session,options=options)
    # mtl_estim_result_resil2 = q_estimator.run(m_def,HAM)
    # options.environment.job_tags = ["resil3"]
    # options.resilience_level=3
    # q_estimator = Estimator(session=session,options=options)
    # mtl_estim_result_resil2 = q_estimator.run(m_def,HAM)

# %%
num_tries = 10
SS1_1q_m = SS1_1q.copy()
SS1_1q_m.measure_all()
scored_layout,deflated_circuit = Mapping.optimize_init_layout(SS1_1q_m,mtl,num_tries,level=3,seed=1123123)#level 2 give wrong results?
deflated_circuit.remove_final_measurements()
#%%
layout_methods = ('trivial', 'dense', 'noise_adaptive', 'sabre')
routing_methods = ('basic', 'lookahead', 'stochastic', 'sabre', 'none')
deflated_circuit.remove_final_measurements()
deflated_circuit.measure_all()
ss1_tr = transpile(deflated_circuit,mtl,initial_layout=list(scored_layout))
#%%

SS1_cr = scale_cr_pulses(ss1_tr,mtl)
# %%
print(eesti.run(deflated_circuit,HAM).result().values)
# %%
inspected_permutation = [3,5,6,7,4,0,2,1]
unt_def = QuantumCircuit(8).compose(deflated_circuit,inspected_permutation)
print(eesti.run(unt_def,HAM).result().values)

#%%
# from itertools import permutations
# for p in permutations(inspected_permutation):
#     unt_def = QuantumCircuit(8).compose(deflated_circuit,p)
#     E = (eesti.run(unt_def,HAM).result().values[0])
#     if np.abs(E-E_SS1) < 1e-6:
#         print(E)
#         print(p)
#         # break
#%%
brute_force_permutation_A = (3, 5, 6, 7, 2, 0, 1, 4)
brute_force_permutation_B = (7, 1, 2, 3, 6, 4, 5, 0)
brute_force_permutation_C = (4, 2, 1, 0, 5, 7, 6, 3)
brute_force_permutation_D = (0, 6, 5, 4, 1, 3, 2, 7)
unt_def = QuantumCircuit(8).compose(deflated_circuit,brute_force_permutation_C)
print(eesti.run(unt_def,HAM).result().values)

#%%
zqr = QuantumRegister(8,"z")
T = QuantumCircuit(zqr).compose(SS1_1q)
T.barrier(range(8))
for i in range(8):
    T.rz(0.1*i,i)
T.draw('mpl')

#%%
importlib.reload(Mapping)

#%%
SS1_1q_m = SS1_1q.copy()
SS1_1q_m.measure_all()
scored_m,m_def = Mapping.optimize_init_layout(SS1_1q_m,mtl,num_tries,level=3,seed=1123123)#level 2 give wrong results?
m_def.remove_final_measurements()
#%%
scored_T,T_def,best_T = Mapping.optimize_init_layout(T,mtl,num_tries,level=3,seed=1123123)
#%%
scored_il,il_def,best_il = Mapping.optimize_init_layout(T,mtl,num_tries,level=2,seed=1123123,initial_layout=range(8))

# %%
from qiskit.visualization import plot_circuit_layout
from qiskit.visualization import plot_coupling_map
# %%
from qiskit.circuit.random import random_circuit

rc = random_circuit(8,5,4,True)
rc.draw('mpl')
#%%
rc_t = transpile(rc,mtl)
rc_t.draw('mpl',idle_wires=False)
# %%
template_basis = ['cx','rz','sx','p','rx','rzx']
template_normalisation = PassManager([Optimize1qGatesDecomposition(template_basis)])
SS1_tr2 = template_normalisation.run(ss1_tr)
# %%
