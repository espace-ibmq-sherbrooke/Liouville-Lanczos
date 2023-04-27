#%%
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian,inner_product,sum

from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition

from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import Matrix_inner_product,Matrix_Liouvillian,Matrix_sum
from LiouvilleLanczos.Quantum_computer.Mapping import find_best_layout
from LiouvilleLanczos.Green import CF_Green

from qiskit import qpy

from qiskit.primitives import Estimator as pEstimator

from qiskit_nature.second_q.mappers import JordanWignerMapper,QubitConverter
from qiskit_nature.second_q.operators import FermionicOp
from qiskit import QuantumCircuit
import numpy as np
from LiouvilleLanczos.Quantum_computer.err_mitig_inprod import twirled_inner_product 
from qiskit_research.utils.convenience import add_pauli_twirls

#%%

U = 4
Ham = Line_Hubbard(-1,U/2,U,2,boundary_condition=BoundaryCondition.OPEN)
C0 = FermionicOp(
    {
        "+_0": 1,
    },
    num_spin_orbitals=4,
)
C0_mat = C0.to_matrix().toarray()
C0101 = FermionicOp(
    {
        "+_0 +_1": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C0110 = FermionicOp(
    {
        "+_0 +_3": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C1001 = FermionicOp(
    {
        "+_1 +_2": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
C1010 = FermionicOp(
    {
        "+_2 +_3": 1,
    },
    num_spin_orbitals=4,
).to_matrix().toarray()
Hmat = Ham.to_matrix().toarray()
estimator = pEstimator()

qubit_converter = QubitConverter(JordanWignerMapper())
HHam = qubit_converter.convert(Ham)
void_state = np.zeros(2**4)
os2 = 1/np.sqrt(2)
void_state[0] = 1
States = [C0101@void_state,C0110@void_state,C1001@void_state,C1010@void_state]
Pn2s0 = np.concatenate([s.reshape(1,2**4) for s in States])
Hn2s0 = Pn2s0@Hmat@Pn2s0.T
Statesp0 = [ (States[1]-States[2])*os2,(States[0]+States[3])*os2]
Pn2s0p0 = np.concatenate([s.reshape(1,2**4) for s in Statesp0])
Hn2s0p0 = Pn2s0p0@Hmat@Pn2s0p0.T
E,S = np.linalg.eigh(Hmat)
Ep0, Sp0 = np.linalg.eigh(Hn2s0p0)
GS_mat = S[:,0]
print(E[0])
t = np.arctan2(Sp0[1,0].real,Sp0[0,0].real)
bt = 0.7854074074074073
GS_analytical = QuantumCircuit(4)
GS_analytical.h(0)
GS_analytical.x(1)
GS_analytical.cx(0,1)
GS_analytical.ry(bt,2)
GS_analytical.x(3)
GS_analytical.cx(2,3)
GS_analytical.cx(1,3)
GS_analytical.cx(1,2)
GS_analytical.cz(1,2)
GS_analytical.swap(1,2)
print(estimator.run(GS_analytical,HHam).result().values)
#%%
matrix_lanczos = Lanczos(Matrix_inner_product(GS_mat),Matrix_Liouvillian(),Matrix_sum())
a_mat,b_mat = matrix_lanczos(Hmat,C0_mat,10)

#%%

# ansatz = SCA(Ham,2)
# powell = POWELL(200,2000)
# vqe = VQE(estimator,ansatz,powell)
# vqer = vqe.compute_minimum_eigenvalue(HHam)

# varqite = VarQITE(ansatz,vqer.optimal_parameters,estimator=estimator,num_timesteps=20,ode_solver=RK23)
# vqite_result = varqite.evolve(TimeEvolutionProblem(HHam,8.0,aux_operators=[HHam]))
# print(vqite_result.aux_ops_evaluated[-1])
# print(E[0])
# #%%
# with open("optimal_2sites.qpy",'wb') as qpyfile:
#     qpy.dump(vqer.optimal_circuit.assign_parameters(vqite_result.parameter_values[-1]),qpyfile)
#%%
with open("optimal_2sites.qpy",'rb') as qpyfile:
    GS = qpy.load(qpyfile)[0]
#%%
print(estimator.run(GS,HHam).result().values)
print(estimator.run(GS_analytical,HHam).result().values)
#%%
eps = 1e-6
lanczos = Lanczos(inner_product(GS_analytical,estimator,qubit_converter,eps),Liouvillian(eps),sum(eps))
#%%
a,b = lanczos(Ham,C0,10,1e-3)

# %%
green = CF_Green(a,b)
green_ed = CF_Green(a_mat,b_mat)

#%%
#On constate que ça marhe très bien en simulation
import matplotlib.pyplot as plt
w = np.linspace(-5.5,5.5,1000)-1e-1j
plt.plot(w,np.imag(green(w)))
plt.savefig("hubu4mu2.pdf")
# plt.plot(w,np.imag(green_ed(w)))
#%%
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

service_algolab = QiskitRuntimeService(
    channel="ibm_quantum", instance="ibm-q-qida/iq-quantum/algolab"
)
#%%
Sher = service_algolab.backend("ibm_sherbrooke")
wash = service_algolab.backend("ibm_washington")
cair =  service_algolab.backend("ibm_cairo")
kolk =  service_algolab.backend("ibmq_kolkata")
hanoi =  service_algolab.backend("ibm_hanoi")
auck =  service_algolab.backend("ibm_auckland")
#%%
backends = {"ibmq_kolkata":kolk,"ibm_sherbrooke":Sher,"ibm_washington":wash}
#%% compute eigenenergy with QC
options = Options()
options.optimization_level = 3
options.transpilation.approximation_degree=1.0
Ntwirl = 20
with Session(backend=kolk) as session:
    backend = backends[session.backend()]
    init_layout,GS_opt = find_best_layout(GS_analytical,backend,10)
    circuit = GS_opt
    # options.resilience_level = 1
    # options.environment.job_tags = ["resil1_E"]
    # estim = Estimator(session=session,options=options)
    # job1 = estim.run(GS_analytical,HHam)
    options.resilience_level = 2
    # options.resilience.extrapolator = "QuadraticExtrapolator"
    options.resilience.noise_factors = (1,2,3,4,5)#seems pretty good with linear extrap
    options.environment.job_tags = ["resil2_E"]
    options.transpilation.initial_layout = init_layout
    estim = Estimator(session=session,options=options)
    job2 = estim.run(GS_opt,HHam)
    options.environment.job_tags = ["resil2_twirls_E"]
    estim = Estimator(session=session,options=options)
    job2 = estim.run(add_pauli_twirls(GS_opt,num_twirled_circuits=Ntwirl),[HHam]*Ntwirl)
    # options.resilience_level = 3
    # options.environment.job_tags = ["resil3_E"]
    # estim = Estimator(session=session,options=options)
    # job2 = estim.run(GS_analytical,HHam)
# %%
job = service_algolab.job('ch4iteqccl2b15p5uvo0')
# %%
from numpy.polynomial import Polynomial
def rzne(results,degree=1):
    """
    Does the pauli twirl average before the ZNE.
    With degree one fit, it doesn't matter wether we do ZNE before or after averaging the pauli twirls
    """
    noise_factors = results.metadata[0]['zne']['noise_amplification']['noise_factors']
    zne_values = np.array([ rm['zne']['noise_amplification']['values'] for rm in results.metadata])
    mean_val = np.mean(zne_values,axis=0)
    f = Polynomial.fit(noise_factors,mean_val,degree)
    return f(0)

# %%
