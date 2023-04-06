# %%
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

)
from qiskit.transpiler.passmanager import PassManager


from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.operators import FermionicOp
from LiouvilleLanczos.Quantum_computer.Hamiltonian import (
    Line_Hubbard,
    BoundaryCondition,
    Line_Hubbard_BaseProblem,
)
from LiouvilleLanczos.Quantum_computer.Mapping import optimize_init_layout
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
mtl = backend = service_algolab.backend("ibmq_montreal")
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
options = Options()
options.environment.job_tags = ["wPtwirls"]
# options.resilience_level=2
# with Session(backend=mtl) as session:
mtl_estimator = Estimator(session=mtl,options=options)
mtl_estim_result = mtl_estimator.run(SS1_1q,HAM)

# %%
SS1_mtl = transpile(SS1_1q,mtl)
#%%
# SS1_cr = scale_cr_pulses(SS1_mtl,mtl)
#%%
# with open("SS1_MTL_CRscaled.qpy",'wb') as handle:
#     qpy.dump(SS1_cr,handle)
# %%
NPT = 15
pt_SS1 = add_pauli_twirls(SS1_1q,NPT)
pttest_result = eesti.run(pt_SS1,[HAM]*NPT).result()
# %%
num_tries = 10
scored_layout,deflated_circuit,best_circuit = optimize_init_layout(SS1_1q,mtl,num_tries,level=3,seed=1123123)
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
# %%
scored_T,T_def,best_T = optimize_init_layout(T,mtl,num_tries,level=2,seed=1123123)
scored_il,il_def,best_il = optimize_init_layout(T,mtl,num_tries,level=2,seed=1123123,initial_layout=range(8))

# %%
