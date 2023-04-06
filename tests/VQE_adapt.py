#%%

#%%
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.minimum_eigensolvers import AdaptVQE
from qiskit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import SlaterDeterminant,BogoliubovTransform
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit.primitives import Estimator,Sampler
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer import AerSimulator
from qiskit import transpile
from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition,Line_Hubbard_BaseProblem
from LiouvilleLanczos.Quantum_computer.Adapt import Adapt,qubitAdapt
from LiouvilleLanczos.Quantum_computer.adapt_pools import Gqubit_pool,Vqubit_pool
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper,QubitConverter
from scipy.integrate import RK23,RK45
import matplotlib.pyplot as plt
import seaborn as sns  

sns.set_theme(style="darkgrid")
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5}) # To enhance the grid's aesthetics
#%%
# Search for better states using SPSA algorithm
from qiskit.algorithms.optimizers import SPSA,COBYLA,POWELL,CG,L_BFGS_B,UMDA,SNOBFIT,IMFIL,SciPyOptimizer
from scipy.optimize import minimize,newton_krylov


spsa = SPSA(3000)
cobyla = COBYLA(200)
powell = POWELL(200)
cg = CG(500)
cg_quick = CG(30,)
bfgs = L_BFGS_B(20)
umda = UMDA(1000)
imfil = IMFIL(2000)
#%%
# Set a starting point for reproduceability

# Create an object to store intermediate results
from dataclasses import dataclass
@dataclass
class VQELog:
    values: list
    parameters: list
    def __init__(self) -> None:
        self.reset()
    def update(self, count, parameters, mean, _metadata):
        self.values.append(mean)
        self.parameters.append(parameters)
        print(f"Running circuit {count}", end="\r", flush=True)
    def reset(self):
        self.values = []
        self.parameters = []
    def __call__(self,count,parameters,mean,_extra):
        return self.update(count,parameters,mean,_extra)

log = VQELog()

Kperspin = -1*np.array([
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0],
    ])
Eps,Sps = np.linalg.eigh(Kperspin)
#%%
Sp = np.zeros((8,8))
Sp[:4,:4] = Sp[4:,4:] = Sps
Sp2 = Sp[[0,4,1,5,2,6,3,7],:]
Sp = Sp2[:,[0,4,1,5,2,6,3,7]]
#%%
Ep = np.sort([*Eps,*Eps])

# Ep,Sp = np.linalg.eigh(Kper)
S = Sp
print(Sp)
#%% Slat2 + Slat3 is best (there's sometime a sign difference in the definitions). [0134] - [0125]
qubit_converter = QubitConverter(JordanWignerMapper())
Bogtrans = BogoliubovTransform(Sp.T,qubit_converter=qubit_converter)
Bog01 = QuantumCircuit(8)
Bog01.x(0)
Bog01.x(1)
Bog01.h(2)
Bog01.cx(2,3)
Bog01.cx(3,4)
Bog01.cx(4,5)
Bog01.x(2)
Bog01.z(2)
Bog01.x(5)
Bog01.compose(Bogtrans,inplace=True)

pBog = Bog01.copy()
pBog.ry(0.1,5)
pBog.ry(3,7)

Hn = QuantumCircuit(8)
for i in range(8):
    Hn.h(i)

#%%
U = 4
Ham_baseProblem = Line_Hubbard_BaseProblem(-1,U/2,U,4,boundary_condition = BoundaryCondition.PERIODIC)
Ham_Umu = Line_Hubbard(0,U/2,U,4,boundary_condition = BoundaryCondition.PERIODIC)
Ham_U = Line_Hubbard(0,0,U,4,boundary_condition = BoundaryCondition.PERIODIC)
Ham_Kmu = Line_Hubbard(-1,U/2,0,4,boundary_condition = BoundaryCondition.PERIODIC)
Ham_K = Line_Hubbard(-1,0,0,4,boundary_condition = BoundaryCondition.PERIODIC)
Ham = Ham_baseProblem.second_q_op()
HHam = qubit_converter.convert(Ham)
Ham_Umu = qubit_converter.convert(Ham_Umu)
Ham_U = qubit_converter.convert(Ham_U)
Ham_Kmu = qubit_converter.convert(Ham_Kmu)
Ham_K = qubit_converter.convert(Ham_K)
Exact,Evec = np.linalg.eigh(Ham.to_matrix().toarray())
print(Exact[0])
#%%
def to_pool(hamil):
    pool = [*hamil]
    pool = [ PauliSumOp(p.primitive/(p.primitive.coeffs[0]) ) for p in pool]
    ID = PauliSumOp(SparsePauliOp('I'*pool[0].num_qubits))
    pool = [p for p in pool if p != ID]
    return pool
#%% stuff for adapt
estimator = Estimator()
Gpool = Gqubit_pool(8)
Gpool2 = []
for i,a in enumerate(Gpool[:-1]):
    for b in Gpool[i+1:]:
        op = (a@b) 
        p = op.primitive
        p.coeffs = 1
        Gpool2.append(PauliSumOp(p))
log.reset()
#%%
Ucircuit = EvolvedOperatorAnsatz(Ham_U,initial_state=QuantumCircuit(8),reps=1)
Ucircuit.decompose(reps=4).draw('mpl')
#%%
Vpool = Vqubit_pool(8)
Ham_pool = to_pool(HHam)
Pool = [Ham_U, Ham_K]
reps = 1
VHA_ansatz = EvolvedOperatorAnsatz(Pool,initial_state=Bog01,reps=reps)
VHA_vqe = VQE(
    estimator,
    VHA_ansatz,
    cobyla,
    initial_point=[1.5]*len(Pool)*reps,
    callback=log,
)
VHA_result1 = VHA_vqe.compute_minimum_eigenvalue(HHam)
print("VHA energy: ", VHA_result1.eigenvalue)
#%%
reps = 2
VHA_ansatz = EvolvedOperatorAnsatz(Pool,initial_state=Bog01,reps=reps)
VHA_vqe = VQE(
    estimator,
    VHA_ansatz,
    cobyla,
    initial_point=list(VHA_result1.optimal_point)+[0.1]*len(Pool)*(reps-1),
    callback=log,
)
VHA_result2 = VHA_vqe.compute_minimum_eigenvalue(HHam)
print("VHA energy: ", VHA_result2.eigenvalue)
#%%
reps =3
VHA_ansatz = EvolvedOperatorAnsatz(Pool,initial_state=Bog01,reps=reps)
VHA_vqe = VQE(
    estimator,
    VHA_ansatz,
    cobyla,
    initial_point=list(VHA_result2.optimal_point)+[0.1]*len(Pool)*(reps-2),
    callback=log,
)
VHA_result3 = VHA_vqe.compute_minimum_eigenvalue(HHam)
print("VHA energy: ", VHA_result3.eigenvalue)

#%%
# Qubit_ansatz = EvolvedOperatorAnsatz([Ham_U,Ham_K],initial_state=Bog01,reps=1).bind_parameters(VHA_result.optimal_point[0:1])
ansatz = EvolvedOperatorAnsatz([*Ham_pool],initial_state=Bog01,reps=1)
initial_point = np.zeros(ansatz.num_parameters)
jobA = estimator.run(ansatz,HHam,initial_point)
jobC = estimator.run(Hn,HHam)
# jobCU = estimator.run(Hn,Ham_U)
# jobCKmu = estimator.run(Hn,Ham_Kmu)
jobB = estimator.run(Bog01,HHam)
jobB2 = estimator.run(Bog01,HHam@HHam)
jobK = estimator.run(Bog01,Ham_Kmu)
jobK2 = estimator.run(Bog01,Ham_Kmu@Ham_Kmu)
jobU = estimator.run(Bog01,Ham_Umu)
jobU2 = estimator.run(Bog01,Ham_U@Ham_U@HHam - Ham_U@HHam@Ham_U)
jobCX = estimator.run(Bog01,HHam@Ham_Umu - Ham_Umu@HHam)
print("ansatz initial energy ",jobA.result())
print("2slaters",jobB.result())
print("|+>^n total",jobC.result())
# print("|+>^n interaction",jobCU.result())
# print("|+>^n Kin+chem",jobCKmu.result())
print(jobB2.result())
print(jobCX.result())
print("<K^2>-<K>^2: ", jobK2.result().values[0] - jobK.result().values[0]**2)
print("<U^2>-<U>^2: ", jobU2.result().values[0] - jobU.result().values[0]**2)
#%%

#%%
local_vqe = VQE(
    estimator,
    ansatz,
    powell,
    initial_point=initial_point,
    callback=log,
)
# print(ansatz.initial_state)

# %%
# instance adapt_VQE
adapt_vqe = qubitAdapt(local_vqe,max_iterations=10)
adapt_vqe.threshold = 1e-6  # critère de convergence.
# %% Try to run the damn thing
log.reset()
VQE_result = adapt_vqe.compute_minimum_eigenvalue(HHam)

print(VQE_result.eigenvalue)
plt.plot(log.values)
#%%
opt_ansatz = VQE_result.optimal_circuit.assign_parameters(VQE_result.optimal_parameters)
opt_ansatz.decompose().draw('mpl')
# %%
sim = AerSimulator(method='statevector')
tansatz = transpile(opt_ansatz,sim)
tansatz.save_statevector('statevector')
result = sim.run(tansatz).result()
ansatzvec = result.data(0)['statevector'].reshape(2**8)
print("ansatz ", np.abs(Evec[:,0]@ansatzvec))
# %%

Test = VQE_result.optimal_circuit

theta = np.linspace(0,2*np.pi,400)
E = np.zeros_like(theta)
for i,t in enumerate(theta):
    Period_job = estimator.run(Test.assign_parameters([t]),HHam)
    E[i] = Period_job.result().values[0]
#%%
plt.plot(theta,np.cos(2*theta+3*np.pi/2)/2-4)
plt.plot(theta,E)


# %%
