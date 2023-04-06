#%%

#%%
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.algorithms import TimeEvolutionProblem
from qiskit_nature.algorithms import AdaptVQE
from qiskit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import SlaterDeterminant,BogoliubovTransform


from qiskit.primitives import Estimator,Sampler
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer import AerSimulator
from qiskit import transpile
from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition
from LiouvilleLanczos.Quantum_computer.ansatz import SCA,flat,two_rots_flat,two_rots_line
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
from qiskit.algorithms.optimizers import SPSA,COBYLA,POWELL,CG,L_BFGS_B,UMDA,SNOBFIT,IMFIL
from scipy.optimize import minimize,newton_krylov


spsa = SPSA(8000)
cobyla = COBYLA(200)
powell = POWELL(2000)
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
    def update(self, count, parameters, mean, _metadata):
        self.values.append(mean)
        self.parameters.append(parameters)
        print(f"Running circuit {count}", end="\r", flush=True)
    def reset(self):
        self.values = []
        self.parameters = []

log = VQELog([],[])

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

H_circ = QuantumCircuit(8)
for i in range(8):
    H_circ.h(i)

Slater0 = SlaterDeterminant(S.T[list(range(4))])
Slater1 = SlaterDeterminant(S.T[[0,1,4,5]])
Slater2 = SlaterDeterminant(S.T[[0,1,3,4]])
Slater3 = SlaterDeterminant(S.T[[0,1,2,5]])
Slater4 = SlaterDeterminant(S.T[[0,1,3,5]])
Bog0 = QuantumCircuit(8)
Bog0.x(0)
Bog0.x(1)
Bog0.x(2)
Bog0.x(3)
Bog0 = Bog0.compose(Bogtrans)
Bog1 = QuantumCircuit(8)
Bog1.x(0)
Bog1.x(1)
Bog1.x(4)
Bog1.x(5)
Bog1 = Bog1.compose(Bogtrans)


#%%
U = 4
Ham = Line_Hubbard(-1,U/2,U,4,boundary_condition=BoundaryCondition.PERIODIC)
K = Line_Hubbard(-1,1e-4/2,1e-4,4,boundary_condition=BoundaryCondition.PERIODIC)
#K is the kinetic hamiltonian with terms to lift the particle number and wavevector degeneracy
# The ground state of this hamiltonian has such large overlap with the exact ground state at U=4,
# with such small perturbation, only degenerate perturbation theory can explain it.
# The perturbations are particle number conserving, and there are only four four-particles states.
N = Line_Hubbard(0,-1,0,4,boundary_condition=BoundaryCondition.PERIODIC)
V = Line_Hubbard(0,0,U,4,boundary_condition=BoundaryCondition.PERIODIC)
sca = SCA(Ham,3)
ansatz = sca.compose(Bogtrans)
# ansatz = H_circ.compose(sca)
# ansatz.compose(two_rots_flat(Ham,1,parameter_prefix='a'),inplace=True)
# ansatz = Slater.compose(ansatz)
ansatz.parameter_bounds = [(0,4*np.pi) for param in ansatz.parameters]
HHam = qubit_converter.convert(Ham)
K = qubit_converter.convert(K).to_matrix()
N = qubit_converter.convert(N).to_matrix()
V = qubit_converter.convert(V).to_matrix()
shots = 200000



#%%
Exact,Evec = np.linalg.eigh(Ham.to_matrix().toarray())
EK,Kvec = np.linalg.eigh(K)
print(Exact[0])
#%%
initial_point = np.zeros( ansatz.num_parameters)
#%%
esti = Estimator()
job = esti.run(Bog01,HHam) 
print("Bogoliubov",job.result())
job = esti.run(Slater0,HHam) 
print("slater0",job.result())
job = esti.run(Slater1,HHam) 
print("slater1",job.result())
job = esti.run(ansatz,HHam,initial_point) 
print("ansatz",job.result())

#%%
log.reset()
vqe = VQE(esti,ansatz,spsa,callback=log.update,initial_point=initial_point)
VQE_result = vqe.compute_minimum_eigenvalue(HHam)
print(VQE_result.eigenvalue)
plt.plot(log.values)
#%% VARQITE from last VQE
vqite = VarQITE(ansatz,VQE_result.optimal_parameters,estimator=esti,num_timesteps=20,ode_solver=RK23)
vqite_result = vqite.evolve(TimeEvolutionProblem(HHam,4.0,aux_operators=[HHam]))
E = [ x[0][0] for x in vqite_result.observables]
plt.plot(E)
print(vqite_result.aux_ops_evaluated)
#%%

#%% VARQITE from last VARQITE
vqite = VarQITE(ansatz,vqite_result.parameter_values[-1],estimator=esti,num_timesteps=20,ode_solver=RK23)
vqite_result = vqite.evolve(TimeEvolutionProblem(HHam,4.0,aux_operators=[HHam]))
E = [ x[0][0] for x in vqite_result.observables]
plt.plot(E)
print(vqite_result.aux_ops_evaluated)
# %% VQE from last VARQITE
log.reset() 
vqe = VQE(esti,ansatz,powell,callback=log.update,initial_point=vqite_result.parameter_values[-1])
VQE_result = vqe.compute_minimum_eigenvalue(HHam)
print(VQE_result.eigenvalue)
plt.plot(log.values)
# %% VQE from last VQE
VQE_E = 5
while np.abs(VQE_E-Exact[0]) > 1e-2:
    new_log = VQELog([],[])
    best_E_ind = np.argmin(log.values)
    vqe = VQE(esti,ansatz,powell,callback=new_log.update,initial_point=log.parameters[best_E_ind])
    VQE_result = vqe.compute_minimum_eigenvalue(HHam)
    log = new_log
    VQE_E = VQE_result.eigenvalue
    print(VQE_result.eigenvalue)
plt.plot(log.values)

#%%
def filter_param_bounds(params,bounds):
    out = []
    for param,bound in zip(params,bounds):
        delta = bound[1]-bound[0]
        param-=bound[0]
        out.append(param%delta+bound[0])
    return out

new_log = VQELog([],[])
best_E_ind = np.argmin(log.values)
vqe = VQE(esti,ansatz,cg,callback=new_log.update,initial_point=filter_param_bounds(log.parameters[best_E_ind],ansatz.parameter_bounds))
VQE_result = vqe.compute_minimum_eigenvalue(HHam)
log = new_log
VQE_E = VQE_result.eigenvalue
print(VQE_result.eigenvalue)
plt.plot(log.values)
 # %%
from qiskit_nature.second_q.operators import FermionicOp
c_0p = FermionicOp({"+_0":1},4)
c_1p = FermionicOp({"+_1":1},4)
c_2p = FermionicOp({"+_2":1},4)
c_3p = FermionicOp({"+_3":1},4)
#%%
from LiouvilleLanczos.Lanczos import Lanczos,inner_product
#%%
f_0 = c_0p
state = ansatz.assign_parameters(vqite_result.parameter_values[-1])
#%%
in_prod = inner_product(state,esti,qubit_converter,1e-5)
lanczos = Lanczos(in_prod)
a,b = lanczos(Ham,f_0,6)
# %%

jp = esti.run(state,qubit_converter.convert(c_0p@Ham@c_0p.adjoint()))
jm = esti.run(state,qubit_converter.convert(c_0p.adjoint()@Ham@c_0p))

# %%
jn = esti.run(state_read, qubit_converter.convert(c_3p@c_3p.adjoint()))
print(jn.result())
# %%
from qiskit import qpy
with open('optimal_2sites.qpy','wb') as fd:
    qpy.dump(state,fd)
# %%
from qiskit import qpy
with open('optimal_2sites.qpy','rb') as fd:
    state_read = qpy.load(fd)[0]


#%%

sim = AerSimulator(method='statevector')
tBog = transpile(Bog01,sim)
tBog.save_statevector('statevector')
result = sim.run(tBog).result()
Bogvec = result.data(0)['statevector'].reshape(2**8)
tslat0 = transpile(Slater0,sim)
tslat0.save_statevector('statevector')
result = sim.run(tslat0).result()
slat0vec = result.data(0)['statevector'].reshape(2**8)
tslat1 = transpile(Slater1,sim)
tslat1.save_statevector('statevector')
result = sim.run(tslat1).result()
slat1vec = result.data(0)['statevector'].reshape(2**8)
tslat2 = transpile(Slater2,sim)
tslat2.save_statevector('statevector')
result = sim.run(tslat2).result()
slat2vec = result.data(0)['statevector'].reshape(2**8)
tslat3 = transpile(Slater3,sim)
tslat3.save_statevector('statevector')
result = sim.run(tslat3).result()
slat3vec = result.data(0)['statevector'].reshape(2**8)
tslat4 = transpile(Slater4,sim)
tslat4.save_statevector('statevector')
result = sim.run(tslat4).result()
slat4vec = result.data(0)['statevector'].reshape(2**8)
#%%
tansatz = transpile(ansatz.assign_parameters(VQE_result.optimal_parameters),sim)
tansatz.save_statevector('statevector')
result = sim.run(tansatz).result()
ansatzvec = result.data(0)['statevector'].reshape(2**8)
# %%
print("bog ", np.abs(Evec[:,0]@Bogvec))
print("slat0 ", np.abs(Evec[:,0]@slat0vec))
print("slat1 ", np.abs(Evec[:,0]@slat1vec))
print("slat2 ", np.abs(Evec[:,0]@slat2vec))
print("slat3 ", np.abs(Evec[:,0]@slat3vec))
print("slat1+slat0 ", np.abs(Evec[:,0]@(slat1vec+slat0vec)/np.sqrt(2) ))
print("slat1-slat0 ", np.abs(Evec[:,0]@(slat1vec-slat0vec)/np.sqrt(2) ))
print("slat2+slat3 ", np.abs(Evec[:,0]@(slat2vec+slat3vec)/np.sqrt(2) ))
print("slat2-slat3 ", np.abs(Evec[:,0]@(slat2vec-slat3vec)/np.sqrt(2) ))
#%%
print("ansatz ", np.abs(Evec[:,0]@ansatzvec))

# %%
slats_fids = []
t = np.linspace(0,2*np.pi,100)
for tt in t:
    slats_fids.append(np.abs(Evec[:,0]@(np.cos(tt)*slat1vec+np.sin(tt)*slat0vec)))
plt.plot(t/(np.pi),slats_fids)
# %%
MH = HHam.to_matrix()
for i in range(16):
    print("overlap ",np.abs(Evec[:,0]@Kvec[:,i]))
    print("energy: ", Kvec[:,i]@MH@Kvec[:,i])
    print("n: ", Kvec[:,i]@N@Kvec[:,i])
    print("interaction: ", Kvec[:,i]@V@Kvec[:,i])
    print("")
# %%

print("slat0 ", (Evec[:,0]@slat0vec))
print("slat1 ", (Evec[:,0]@slat1vec))
print("slat2 ", (Evec[:,0]@slat2vec))
print("slat3 ", (Evec[:,0]@slat3vec))
print("slat4 ", (Evec[:,0]@slat4vec))

# %%
