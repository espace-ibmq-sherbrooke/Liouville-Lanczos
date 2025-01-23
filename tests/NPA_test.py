# %%
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate  # TwoLocal, ZZFeatureMap, etc
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from numbers import Number
from qiskit.synthesis import LieTrotter
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator, StatevectorSampler, Estimator
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal,polar
from LiouvilleLanczos.Quantum_computer.VQE_stuff.ansatz import ControllableHEA, Real_NP_ansatz

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

# %%
estimator = Estimator()
spsa = SPSA(maxiter=600)
lbfgs = L_BFGS_B()
E_dict_ESU2 = {}
E_dict_NPA = {}
param_dict_ESU2 = {}
param_dict_NPA = {}
reduced_ent_map = [(1, 0), (3,4), (5,7), (8,9), (10,11)]

for rep in range(1, 3, 2):
    print(f"reps: {rep}")
    ansatz_NP = QuantumCircuit(12)
    ansatz_NP.x(0)
    ansatz_NP.x(2)
    ansatz_NP.x(4)
    ansatz_NP.x(7)
    ansatz_NP.x(9)
    ansatz_NP.x(11)
    ansatz_NP.append(Real_NP_ansatz(12, rep, reduced_ent_map, 'theta'), range(12))
    vqe = VQE(estimator, ansatz_NP, spsa)
    sol = vqe.compute_minimum_eigenvalue(H)
    print("spsa Energy ", sol.optimal_value)
    vqe = VQE(estimator, ansatz_NP, lbfgs, initial_point=sol.optimal_point)
    sol = vqe.compute_minimum_eigenvalue(H)
    E_dict_NPA[rep] = sol.optimal_value
    param_dict_NPA[rep] = sol.optimal_parameters
    print("bfgs energy ", sol.optimal_value)
# %%
#Test on real QC
def prep_psi_0(qc: QuantumCircuit):
    qc.cx(12, 9)
    #
    qc.cx(9,10)
    #
    qc.cx(10,11)
    qc.cx(9,8)
    #
    qc.cx(11,6)
    qc.cx(8,7)
    #
    qc.cx(7,5)
    qc.cx(6,4)
    #
    qc.cx(4,3)
    qc.cx(5,0)
    #
    qc.cx(3,2)
    #
    qc.cx(2,3)
    qc.cx(4,6)
    qc.cx(11,10)
    qc.cx(9,8)
    qc.cx(7,5)
    #ansatz_NP = Real_NP_ansatz(12, 1, f_ent_map, 'theta' + ctrl_state)
    #qc.append(ansatz_NP, range(12))
    return qc

def prep_psi_0_dag(qc: QuantumCircuit):
    for i in [0,2,4,7,9,11]:
        qc.cx(12,i,ctrl_state=0)
    return qc

#%%
#NO ANSATZ
#basis state has energy -12
estim = StatevectorEstimator()
real_H = []
imag_H = []
real_F = []
imag_F = []
real_obs_H = SparsePauliOp('X', 1) ^ H
imag_obs_H = SparsePauliOp('Y', 1) ^ H
real_obs_I = SparsePauliOp('X', 1) ^ SparsePauliOp('I'*12, 1)
imag_obs_I = SparsePauliOp('Y', 1) ^ SparsePauliOp('I'*12, 1)
for i in range(10):
    qc = QuantumCircuit(13)
    qc.h(12)
    qc = prep_psi_0(qc)
    synth = LieTrotter(reps = 1)
    delta_t = np.pi/40
    time_evol = PauliEvolutionGate(H, delta_t, synthesis = synth)
    for j in range(i):
        qc.append(time_evol, range(12))
    qc = prep_psi_0_dag(qc)
    res = estim.run([(qc, [real_obs_H, real_obs_I, imag_obs_H, imag_obs_I])])
    real_H.append(res.result()[0].data.evs[0])
    imag_H.append(res.result()[0].data.evs[2])
    real_F.append(res.result()[0].data.evs[1])
    imag_F.append(res.result()[0].data.evs[3])

def make_H_S_from_vec(real, imag):
    assert len(real) == len(imag)
    n = len(real)
    H_tilde = np.zeros([n,n], dtype=complex)
    for i in range(n):
        for k in range(n):
            if k - i >= 0:
                H_tilde[i,k] = real[k-i] + 1j*imag[k-i]
            else:
                H_tilde[i,k] = real[np.abs(k-i)] - 1j*imag[np.abs(k-i)]
    return H_tilde

H_tilde = np.array(make_H_S_from_vec(real_H, imag_H))
S_tilde = np.array(make_H_S_from_vec(real_F, imag_F))

def truncation(threshold, S):
    eigvals, eigvecs = np.linalg.eig(S)
    idx = eigvals.argsort()[::-1]   
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    D = len(eigvals)
    truncated_eigvals = []
    truncated_eigvecs = []
    #make truncated matrix whose rows are eigenvectors of S with eigval above threshold
    for i in range(D):
        if eigvals[i] >= threshold:
            truncated_eigvals.append(eigvals[i])
            truncated_eigvecs.append(eigvecs[i])
    truncated_eigvals = np.array(truncated_eigvals)
    truncated_eigvecs = np.array(truncated_eigvecs).T    
    return truncated_eigvecs
#%%
#solve diagonalization problem, find dependence on dimension of krylov subspace. 
def E_vs_D(H_tilde, S_tilde, threshold_slope):
    GSEs = []
    for d in range(1, len(S_tilde)+1):
        epsilon = threshold_slope*d
        H_temp = H_tilde[0:d, 0:d]
        S_temp = S_tilde[0:d, 0:d]
        V_eps = truncation(epsilon, S_temp)
        #solve GEVP
        A = V_eps.conj().T @ H_temp @ V_eps
        B = V_eps.conj().T @ S_temp @ V_eps
        print(A.shape[0])
        E, c = scipy.linalg.eig(a = A, b = B)
        GSEs.append(sorted(E)[0])
    return GSEs
GSEs = E_vs_D(H_tilde, S_tilde, 1e-8)
plt.plot(GSEs)
#%%
#start conjugating observables by circuit elements

def cx_by_0_spo(n_qubits: int, ctrl: int, target: int):
    #CNOT = 1/2(II + IZ + XI - XZ) where Z is on ctrl, X is on target
    x_on_ctrl = ['I']*n_qubits
    x_on_ctrl[n_qubits - 1 - ctrl] = 'X'
    x_on_ctrl = ''.join(x_on_ctrl)
    x_on_ctrl = SparsePauliOp(x_on_ctrl)

    cx = cx_spo(n_qubits, ctrl, target)
    return x_on_ctrl & cx & x_on_ctrl

n_qubits = 13
circuit_gates = []
params = list(param_dict_NPA[1].values())
for target in [0,2,4,7,9,11]:
    circuit_gates.append(cx_by_0_spo(n_qubits, 12, target))
# (5,0,8), (8,7,10), (10,9,11),(2,1,4),(4,6,6),(11,6,7),(3,2,9)
conj_real_H = real_obs_H
conj_imag_H = imag_obs_H
conj_real_I = real_obs_I
conj_imag_I = imag_obs_I
for gate in circuit_gates[::-1]:
    conj_real_H = (gate & conj_real_H & gate).simplify()
    conj_imag_H = (gate & conj_imag_H & gate).simplify()
    conj_real_I = (gate & conj_real_I & gate).simplify()
    conj_imag_I = (gate & conj_imag_I & gate).simplify()
#%%
#sanity check
circuits = []
real_H = []
imag_H = []
real_F = []
imag_F = []
real_obs_H = SparsePauliOp('X', 1) ^ H
imag_obs_H = SparsePauliOp('Y', 1) ^ H
real_obs_I = SparsePauliOp('X', 1) ^ SparsePauliOp('I'*12, 1)
imag_obs_I = SparsePauliOp('Y', 1) ^ SparsePauliOp('I'*12, 1)
for i in range(10):
    qc = QuantumCircuit(13)
    qc.h(12)
    qc = prep_psi_0(qc)
    synth = LieTrotter(reps = 1)
    delta_t = np.pi/40
    time_evol = PauliEvolutionGate(H, delta_t, synthesis = synth)
    for j in range(i):
        qc.append(time_evol, range(12))
    circuits.append(qc)
    res = estim.run([(qc, [conj_real_H, conj_real_I, conj_imag_H, conj_imag_I])])
    real_H.append(res.result()[0].data.evs[0])
    imag_H.append(res.result()[0].data.evs[2])
    real_F.append(res.result()[0].data.evs[1])
    imag_F.append(res.result()[0].data.evs[3])
#%%
H_tilde = np.array(make_H_S_from_vec(real_H, imag_H))
S_tilde = np.array(make_H_S_from_vec(real_F, imag_F))
GSEs = E_vs_D(H_tilde, S_tilde, 1e-8)
plt.plot(GSEs)
#consistent
#%%
#run on quebec
from qiskit_ibm_runtime import (
    Session,
    Sampler,
    QiskitRuntimeService,
    Options,
    Estimator,
    EstimatorV2
)
from qiskit_ibm_runtime.options import (
    EnvironmentOptions,
    EstimatorOptions
)
service_algolab = QiskitRuntimeService(
    channel="ibm_quantum",
    instance = 'ibm-q-qida/iq-quantum/algolab'
)
torino = service_algolab.backend('ibm_torino')
#quebec =  service_algolab.backend("ibm_quebec")
# %%
backends = {'ibm_torino': torino}
bkd = torino.name
#backends = {'ibm_quebec':quebec}
#bkd = quebec.name
#%%
results = []
with Session(backend=backends[bkd]) as session:
    estim_options = EstimatorOptions()
    backend = backends[session.backend()]
    estim_options.resilience_level = 2
    estim_options.default_shots=10000 #shot noise 10000 -> ~0.01, 100000 -> ~0.003
    estim_options.environment.job_tags = []
    estim_options.dynamical_decoupling.enable = True
    estim_options.dynamical_decoupling.sequence_type = 'XpXm'
    estim = EstimatorV2(mode = session ,options=estim_options)
    eps = 1e-5
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    pubs = []
    for i in range(0,len(circuits)):
        isa_circuit = pm.run(circuits[i])
        isa_real_H = conj_real_H.apply_layout(isa_circuit.layout)
        isa_imag_H = conj_imag_H.apply_layout(isa_circuit.layout)
        isa_real_I = conj_real_I.apply_layout(isa_circuit.layout)
        isa_imag_I = conj_imag_I.apply_layout(isa_circuit.layout)
        pub = (isa_circuit, [isa_real_H, isa_imag_H, isa_real_I, isa_imag_I])
        estim_job = estim.run([pub])
        results.append(estim_job.result())
#%%
evs = [result[0].data.evs for result in results]
real_H_qc, imag_H_qc, real_F_qc, imag_F_qc = [ev[0] for ev in evs], [ev[1] for ev in evs], [ev[2] for ev in evs], [ev[3] for ev in evs]
H_tilde_qc = np.array(make_H_S_from_vec(real_H_qc, imag_H_qc))
S_tilde_qc = np.array(make_H_S_from_vec(real_F_qc, imag_F_qc))
GSEs_qc = E_vs_D(H_tilde_qc, S_tilde_qc, 1e-5)
plt.plot(GSEs_qc)
#%%
#try q-ctrl
from qiskit_ibm_catalog import QiskitFunctionsCatalog
catalog = QiskitFunctionsCatalog()
perf_mgm = catalog.load('q-ctrl/performance-management')
#%%
pubs_list = []
obs = [conj_real_H, conj_imag_H, conj_real_I, conj_imag_I]
#separate pubs
for i in range(5):
    pubs_list.append([])
    for j in range(5):
        pubs_list[i].append((circuits[20*i + 4*j], obs))
#%%
for i in range(5):
    qctrl_res = perf_mgm.run(
        primitive="estimator",
        pubs = pubs_list[i],
        instance = 'pinq-quebec-hub/iq-quantum/algolab',
        backend_name = 'ibm_quebec',
    )
    print(qctrl_res.job_id)
#%%
real_H_qctrl = []
imag_H_qctrl = []
real_F_qctrl = []
imag_F_qctrl = []
for i in range(5):
    job = perf_mgm.get_jobs()[4-i]
    for j in range(4):
        evs = job.result()[j].data.evs
        real_H_qctrl.append(evs[0])
        imag_H_qctrl.append(evs[1])
        real_F_qctrl.append(evs[2])
        imag_F_qctrl.append(evs[3])
# %%
import matplotlib.pyplot as plt
plt.plot(real_H)
plt.plot(real_H_qc)
plt.xlabel('$\Delta t (\pi/40)$')
plt.legend(['simulation', 'ibm_torino'])
plt.title('real H projection elements')
plt.show()
# %%
plt.plot(real_F)
plt.plot(real_F_qc)
plt.xlabel('$\Delta t (\pi/40)$')
plt.legend(['simulation', 'ibm_torino'])
plt.title('Krylov basis overlap')
plt.show()
#%%
plt.plot(GSEs)
plt.plot(GSEs_qc)
plt.xlabel('D')
plt.legend(['simulation', 'ibm_torino'])
plt.title('Ground state energy')
plt.show()
# %%


#do not look at this code, horrible
def cx_spo(n_qubits: int, ctrl: int, target: int):
    cx0 = ['I']*n_qubits
    cx1 = ['I']*n_qubits
    cx2 = ['I']*n_qubits
    cx3 = ['I']*n_qubits

    cx1[n_qubits - 1 - ctrl] = 'Z'
    cx2[n_qubits - 1 - target] = 'X'
    cx3[n_qubits - 1 - ctrl] = 'Z'
    cx3[n_qubits - 1 - target] = 'X'

    cx = SparsePauliOp([''.join(cx0), ''.join(cx1), ''.join(cx2), ''.join(cx3)], [1/2,1/2,1/2,-1/2])
    return cx

def ccx_spo(n_qubits, ctrl1, ctrl2, target):
    ccx_3q = QuantumCircuit(3)
    ccx_3q.ccx(0,1,2)
    ccx_3q = SparsePauliOp.from_operator(Operator(ccx_3q))
    paulis = []
    for pauli in ccx_3q.paulis:
        n_pauli = ['I']*n_qubits
        n_pauli[n_qubits - 1 - ctrl1] = str(pauli[0])
        n_pauli[n_qubits - 1 - ctrl2] = str(pauli[1])
        n_pauli[n_qubits - 1 - target] = str(pauli[2])
        paulis.append(''.join(n_pauli))
    return SparsePauliOp(paulis, ccx_3q.coeffs)

def ccry_spo(n_qubits, ctrl1, ctrl2, target, theta):
    cry = QuantumCircuit(2)
    cry.cry(theta,0,1)
    ccry_3q = QuantumCircuit(3)
    ccry_3q = ccry_3q.compose(cry.control(1), [0,1,2])
    ccry_3q = SparsePauliOp.from_operator(Operator(ccry_3q))
    paulis = []
    for pauli in ccry_3q.paulis:
        n_pauli = ['I']*n_qubits
        n_pauli[n_qubits - 1 - ctrl1] = str(pauli[0])
        n_pauli[n_qubits - 1 - ctrl2] = str(pauli[1])
        n_pauli[n_qubits - 1 - target] = str(pauli[2])
        paulis.append(''.join(n_pauli))
    return SparsePauliOp(paulis, ccry_3q.coeffs)

def ctrl_NPA_component_spo(n_qubits: int, ctrl: int, q0: int, q1: int, theta):
    #npa component goes like cx(q0, q1) cry(q1, q0) cx(q0, q1)
    ccx = ccx_spo(n_qubits, ctrl, q0, q1)
    ccry = ccry_spo(n_qubits, ctrl, q1, q0, theta)
    
    return (ccx & ccry & ccx).simplify()