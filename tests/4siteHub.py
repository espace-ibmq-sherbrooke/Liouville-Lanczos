#%%
import qiskit
import numpy as np
from qiskit import qpy
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.primitives import Estimator as exactEstimator
from qiskit.transpiler.passes import (
    RemoveBarriers,
)
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_slo,inner_product_slo,sum_slo

from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum

from LiouvilleLanczos.Green import CF_Green

import numpy as np

from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.operators import FermionicOp

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
C0u= FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=8,
)
C1u= FermionicOp(
    {
        "+_1": 1,
    },
    num_spin_orbitals=8,
)
C2u= FermionicOp(
    {
        "+_2": 1,
    },
    num_spin_orbitals=8,
)
C3u= FermionicOp(
    {
        "+_3": 1,
    },
    num_spin_orbitals=8,
)
C0u_mat = C0u.to_matrix().toarray()
C1u_mat = C1u.to_matrix().toarray()
C2u_mat = C2u.to_matrix().toarray()
C3u_mat = C3u.to_matrix().toarray()
t = -1
U = 4
mu = U/2
Hubbard_FOP = t*first_hop-mu*Number_op+U*doubble_occup
Hubbard_FOP
HAM = qubit_converter.map(Hubbard_FOP)
Hubbard_matrix = HAM.to_matrix()
E,S = np.linalg.eigh(Hubbard_matrix)
print("Exact", E[0])
# %%
with open("FH1x4_ground_state_3_initSlater.qpy", "rb") as handle:
    SS1 = qpy.load(handle)[0]

SS1 = RemoveBarriers()(SS1)
estimator = exactEstimator()


#%%
"""
Vérification que les circuit de Camille sont exprimé dans la base du hamiltonien
que j'ai construit.
"""
print(estimator.run(SS1,HAM).result().values[0])
E_SS1 = estimator.run(SS1,HAM).result().values[0]
print("circuit", E_SS1)
#%%
eps = 1e-5
estimator_lanczos = Lanczos(inner_product_slo(SS1,estimator,qubit_converter,eps),Liouvillian_slo(eps),sum_slo(eps))
matrix_lanczos = Lanczos(MatrixState_inner_product(S[:,0]),Matrix_Liouvillian(),Matrix_sum())
#%%
a_mat,b_mat = matrix_lanczos(Hubbard_matrix,C1u_mat,10)
# %%
# a,b = estimator_lanczos(Hubbard_FOP,C1u,10,1e-3) # paulisumop are horrible!
a,b = ([-0.007300000000000002,
  -0.0021167460995202003,
  0.025516176493191413,
  -0.02142163554568002,
  0.04498748405915269,
  -0.07348957330101491,
  0.10544643498103433,
  -0.15238005914670316,
  0.13862509883041355,
  -0.032808915239906894,
  -0.06536050843940663],
 [1.0,
  2.4495095611162663,
  2.677239515331171,
  4.135861768914755,
  3.8508410455659994,
  6.280536373686746,
  3.410701122452174,
  5.801058773078206,
  5.24194358483827,
  5.057670229846699,
  5.006364123403905])
b = b[1:-1]# conventionnal difference in the computation.
a = a[:-1]
print("Obtained using Jerome's implentation that relies on an old version of PauliArray. 40000 shots oer averages")

# %%
green = CF_Green(a,b)
green_ed = CF_Green(a_mat,b_mat)
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
plt.plot(w,np.imag(green_ed(w)))
plt.plot(w,np.imag(green(w)))
# plt.savefig("hubu4mu4.pdf")
# %%
