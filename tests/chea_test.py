# %%
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2  # TwoLocal, ZZFeatureMap, etc
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from numbers import Number
import numpy as np
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal,polar
from LiouvilleLanczos.Quantum_computer.VQE_stuff.ansatz import ControllableHEA

def Lanczos_valh(H, psi0, eps, iter=200):
    psi_n = psi0 / np.sqrt(psi0 @ psi0)
    psi_nm = 0
    alpha_s = []
    beta_s = []
    alpha = 0
    beta = 0
    for i in range(iter):
        psi_nm = H @ psi_n - beta * psi_nm
        alpha = psi_n @ psi_nm
        alpha_s.append(alpha)
        psi_nm -= alpha * psi_n
        beta = np.sqrt(psi_nm @ psi_nm)
        psi_n, psi_nm = psi_nm / beta, psi_n
        Evals, Evecs = eigh_tridiagonal(alpha_s, beta_s)
        # print(Evals,beta)
        if abs(beta * Evecs[-1, -1]) < eps:
            break
        beta_s.append(beta)
    return Evals[0], beta * Evecs[-1, -1]


# %%
backend = FakeQuebec()
target = backend.target


# %%
def Heisenberg(J, n, ent_map):
    if isinstance(J, Number):
        J = np.ones(len(ent_map)) * J
    else:
        assert len(J) == len(ent_map)
    String = "I" * n
    H = SparsePauliOp("I" * n, 0)
    for j, c in zip(J, ent_map):
        c = np.sort(c)
        XX = String[: c[0]] + "X" + String[c[0] + 1 : c[1]] + "X" + String[c[1] + 1 :]
        H += SparsePauliOp(XX, j)
        YY = String[: c[0]] + "Y" + String[c[0] + 1 : c[1]] + "Y" + String[c[1] + 1 :]
        H += SparsePauliOp(YY, j)
        ZZ = String[: c[0]] + "Z" + String[c[0] + 1 : c[1]] + "Z" + String[c[1] + 1 :]
        H += SparsePauliOp(ZZ, j)
    return H.chop()


# %%


from qiskit.visualization import plot_gate_map

# need to manually add edge qubits
plot_gate_map(backend, figsize=(7, 7), font_size=40, font_color="black")
# %% Nick's code to produce the ansatz for an entire Qcomp.
cm = target.build_coupling_map()
deg3_qubits = [
    idx for idx, row in enumerate(cm.distance_matrix) if list(row).count(1) == 3
]
edge_qubits = [0, 2, 6, 10, 18, 32, 37, 51, 56, 70, 75, 89, 94, 108, 116, 120, 124]
full_layering = [
    [edge for edge in cm.get_edges() if d3q in edge]
    for d3q in deg3_qubits + edge_qubits
]
ent_map = sum(
    [[layer[idx] for layer in full_layering if idx < len(layer)] for idx in range(3)],
    [],
)
# create ansatz

ansatz_HEA = EfficientSU2(
    target.num_qubits, entanglement=ent_map, reps=3, skip_final_rotation_layer=True
)
print(f"Circuit has {ansatz_HEA.num_parameters} parameters")
ansatz_HEA.decompose().draw(fold=-1)

ansatz_CHEA = ControllableHEA(target.num_qubits, ent_map, reps = 2)
print(f"Circuit has {ansatz_CHEA.num_parameters} parameters")
ansatz_CHEA.decompose().draw(fold=-1)
# %%
# The subset of qubits i want to use.
qubit_subset = [0, 1, 2, 3, 4, 14, 15, 18, 19, 20, 21, 22]
remap = {q: i for q, i in zip(qubit_subset, range(len(qubit_subset)))}
# filtering the entanglement map such that only the desired qubits are present.
f_ent_map = []
for a, b in ent_map:
    if a in qubit_subset and b in qubit_subset:
        f_ent_map.append((remap[a], remap[b]))
H = Heisenberg(1, 12, f_ent_map)
ansatz = ControllableHEA(
    len(qubit_subset), ent_map=f_ent_map, reps=2
)
#%%
ansatz.add_control([(0, 1, 2, 3, 4, 9, 10), (5,6,7,8,11)])

# %%
estimator = Estimator()
spsa = SPSA(maxiter=600)
lbfgs = L_BFGS_B()
E_dict = {}
E_dict_c_hea = {}
for rep in range(1, 5):
    print(f"reps: {rep}")
    ansatz = ControllableHEA(
        len(qubit_subset), ent_map=f_ent_map, reps=rep
    )
    print(f"Circuit has {ansatz.num_parameters} parameters")
    # ansatz.draw(fold=-1)
    vqe = VQE(estimator, ansatz, spsa)
    sol = vqe.compute_minimum_eigenvalue(H)
    print("spsa Energy ", sol.optimal_value)
    vqe = VQE(estimator, ansatz, lbfgs, initial_point=sol.optimal_point)
    sol = vqe.compute_minimum_eigenvalue(H)
    E_dict_c_hea[rep] = sol.optimal_value
    print("bfgs energy ", sol.optimal_value)
# %%
