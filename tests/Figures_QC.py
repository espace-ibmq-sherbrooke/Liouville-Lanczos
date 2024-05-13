
#%%
from LiouvilleLanczos.Quantum_computer.Hamiltonian import Line_Hubbard,BoundaryCondition
from importlib import reload

import LiouvilleLanczos.Lanczos
from LiouvilleLanczos.Lanczos import Lanczos
from LiouvilleLanczos.matrix_impl import MatrixState_inner_product,Matrix_Liouvillian,Matrix_sum,DensityMatrix_inner_product,Hamiltonian_inner_product,Matrix_Hamiltonian
from LiouvilleLanczos.Green import CF_Green,Green_matrix,PolyLehmann_Green,PolyCF_Green
from LiouvilleLanczos.Quantum_computer.QC_lanczos import Liouvillian_slo as QCLiou,inner_product_slo as QCip,sum_slo as QCsum
Fliou = QCLiou(1e-15)
MaLiou = Matrix_Liouvillian()

reload(LiouvilleLanczos.Lanczos)
Lanczos = LiouvilleLanczos.Lanczos.Lanczos

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
sns.set_palette('colorblind')

#%%two site hubbard model GS circuit
from qiskit import QuantumCircuit
bt = 0.7854074074074073
GS_analytical = QuantumCircuit(4)
GS_analytical.h(0)
GS_analytical.x(1)
GS_analytical.cx(0,1)
GS_analytical.ry(bt,2)
GS_analytical.x(3)
GS_analytical.cx(2,3)
GS_analytical.swap(1,2)
GS_analytical.cx(2,3)
GS_analytical.cx(2,1)
GS_analytical.cz(2,1)
GS_analytical.draw('mpl',filename="2sitecirc.pdf")

#%%
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
mapper = JordanWignerMapper()

import numpy as np

def fermi_dirac(w,T):
    return 0.5*(1-np.tanh(0.5*w/T))

def fermi_dirac_T0(w):
    return w<=0.0

def fermi_dirac_dT(w,T):
    return -w*fermi_dirac_dw(w,T)/T

def fermi_dirac_dw(w,T):
    return -0.25/(T*np.cosh(0.5*w/T))

def dmat(beta,E,S):
    rhop = np.exp(-beta*(E-E[0]))
    rhop /= np.sum(rhop)
    rhop = S@np.diag(rhop)@S.T
    return rhop
#%% Produit la plaquette 1x2
U = 4
N_site = 2
N_orb = 2*N_site
mu2 = -0.0
mu = U/2 + mu2 
t = -1
Boundary = BoundaryCondition.OPEN
Ham_sq = Line_Hubbard(t,mu,U,N_site,boundary_condition=Boundary)
K_sq = Line_Hubbard(t,0,0,N_site,boundary_condition=Boundary)
Mu_sq = Line_Hubbard(0,U/2+mu2,0,N_site,boundary_condition=Boundary)
N_sq = Line_Hubbard(0,-1,0,N_site,boundary_condition=Boundary)
V_sq = Line_Hubbard(0,0,U,N_site,boundary_condition=Boundary)
mu_0 = np.array([[-U/2-mu2,0,0,0],[0,-U/2-mu2,0,0],[0,0 ,-U/2-mu2,0],[0,0,0,-U/2-mu2]])
K_0 = np.array([[0,t,0,t],[t,0,t,0],[0,t ,0,t],[t,0,t,0]])
H_0 = mu_0+K_0

def FermionicOp2Matrix(fo):
    return mapper.map(fo).to_matrix()

C0 = FermionicOp(
    {
        "-_0": 1,
    },
    num_spin_orbitals=N_orb,
)
C0_mat = np.real(FermionicOp2Matrix(C0))
C1 = FermionicOp(
    {
        "-_1": 1,
    },
    num_spin_orbitals=N_orb,
)
C1_mat = np.real(FermionicOp2Matrix(C1))
C2 = FermionicOp(
    {
        "-_2": 1,
    },
    num_spin_orbitals=N_orb,
)
C2_mat = np.real(FermionicOp2Matrix(C2))
C3 = FermionicOp(
    {
        "-_3": 1,
    },
    num_spin_orbitals=N_orb,
)
C3_mat = np.real(FermionicOp2Matrix(C3))
#Convertie en matrice Numpy
Ham = np.real(FermionicOp2Matrix(Ham_sq))
K = np.real(FermionicOp2Matrix(K_sq))
N = np.real(FermionicOp2Matrix(N_sq))
Mu = np.real(FermionicOp2Matrix(Mu_sq))
V = np.real(FermionicOp2Matrix(V_sq))
#Diagonalise
E,S = np.linalg.eigh(Ham)

GS_mat = S[:,0]

#% Grille de fréquences
w = np.linspace(-(U+2),U+2,1000)
z = w-0.2j

# I have improperly named the output python files containing the data.
# They consequently cannot be imported as python modules...
# I copy-pasted the data with a comment identifying the source file.
#%%Greenfct, T=0
#10k shots per average
T0_matrix_lanczos = Lanczos(MatrixState_inner_product(GS_mat),Matrix_Liouvillian(),Matrix_sum())
# T0_hamilt_less_lanczos = Lanczos(Hamiltonian_inner_product(),Matrix_Hamiltonian(-E[0],1),Matrix_sum())
# T0_hamilt_great_lanczos = Lanczos(Hamiltonian_inner_product(),Matrix_Hamiltonian(-E[0],-1),Matrix_sum())
a_ex,b_ex,mu_ex = T0_matrix_lanczos.polynomial_hybrid(Ham,C0_mat,[C2_mat],40)
CFT0_Green00 = CF_Green(a_ex,b_ex)
T0_Green00 = CFT0_Green00.to_Lehmann()
T0_Green01 = PolyLehmann_Green(a_ex,b_ex,[m[0] for m in mu_ex],T0_Green00)
Green_ex = Green_matrix([T0_Green00,T0_Green01],2,[(0,0,0,1),(1,1,0,1),(0,1,1,1),(1,0,1,1)])
# from QC_hubbard2site_data_ibm_sherbrooke_2023-09-29 13:24:58.510333
a_sher10k = [-0.012855074664641862, 0.03693252689956106, -0.030814378775743514, 0.02523439342063026, -0.09497966099663938, 0.2537174920481988, -0.6139226415827819, 0.8390481143723716, -0.7758101855605963, 0.487877930658288]
b_sher10k=[1.0, 2.235898800294271, 1.7889090598867616, 3.0447106818186516, 0.587272629379638, 1.9740407084114706, 1.9940529084000225, 1.4139747491089758, 1.1191978052701703, 3.566367021755596]
mu_sher10k=[[0.0], [0.4472474335011889], [-0.007973019150648334], [-0.8434935283847338], [0.033785106913615144], [-0.2205605238242081], [0.0024844925863531098], [-0.01426004917184498], [-0.07922164607969007], [-0.38861721931914855]]
QC_Green00_sher10k = CF_Green(a_sher10k,b_sher10k)
#from QC_hubbard2site_data_ibm_quebec_2023-09-29 12:50:48.789461
a_queb10k = [-0.0012538775510204082, 0.07529528986087661, -0.08399313683090669, 0.04235659057670431, -0.358743821334164, 0.40636999616834535, -0.05758852218916925, 0.8730063795760724, -0.11906942573871526, 1.8725807820011524]
b_queb10k=[1.0, 2.2360968059079176, 1.7902002186357513, 2.9686756037281556, 0.3499550865492233, 5.071369918991414, 9.823146398778864, 10.186979839226943, 4.622199623840157, 5.566462829018226]
mu_queb10k=[[0.0], [0.4472078298926652], [-0.029943872106810106], [-0.8550024508244485], [0.21471111111387958], [-0.4890632322564047], [-0.14313724978823075], [0.2834586308674055], [0.13129707174192679], [-0.4819825224462918]]
QC_Green00_queb10k = CF_Green(a_queb10k,b_queb10k)
#from V2_QC_hubbard2site_data_ibm_quebec_15_15_58
a_queb10k_V2 = np.real([(0.09271128926636638+0j), (-0.31698532540512103-0.003572352402397284j), (0.3933722857171073+0.01524312739576108j), (-0.35909771202053486-0.018803127527701264j), (0.4015336586638849+0.002393488087792875j), (-0.450666440361444-0.02716183433318591j), (0.42077387142958994-0.01639072954264988j), (-0.5582855047097648+0.01126677989657012j), (0.7472543479313848+0.0331584135127503j), (-0.48566813242169493-0.003928619712815829j)])
b_queb10k_V2 = np.real([(1+0j), (2.2366377695204447+0j), (1.8018416380255118+0j), (2.88665503100614+0j), (1.1549048149380159+0j), (2.3727221045560603+0j), (1.752944543077286+0j), (1.6960178260076464+0j), (1.6340630064406159+0j), (3.4686318608901883+0j)])
mu_queb10k_V2 = np.real([[0j], [(0.44709966612716595+0j)], [(0.08497958847615386-0.0008864250512884365j)], [(-0.7343580244173532-0.09275837645457413j)], [(-0.23460683270922217-0.005635299595888764j)], [(-0.21634510025977816+0.12386885928949454j)], [(0.1274670441645987+0.02670037370751219j)], [(0.42334050846763116-0.055700157471329705j)], [(0.22732341097921183+0.009076626305405722j)], [(0.03706418355096688+0.04025860751383242j)]])
QC_Green00_queb10k_V2 = CF_Green(a_queb10k_V2,b_queb10k_V2)
#%% quantum simulation, testing for the sign on the moments.
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
GS_circ = QuantumCircuit(4)
GS_circ.prepare_state(GS_mat)
estim = Estimator()
Qsim_lanczos = Lanczos(QCip(GS_circ,estim,mapper,1e-15),QCLiou(1e-15),QCsum(1e-15))
print(estim.run(GS_circ,mapper.map(Ham_sq)).result())
#a_sim,b_sim,mu_sim = Qsim_lanczos.polynomial_hybrid(Ham_sq,C0,[C2],5)
#Does not reproduce.

# %%
w = np.linspace(-6,6,1000)
z = w-0.1j
plt.plot(w,np.imag(T0_Green00(z)),label="exact")
plt.plot(w,np.imag(QC_Green00_queb10k(z)),label="ibm_quebec")
plt.plot(w,np.imag(QC_Green00_sher10k(z)),label="ibm_sherbrooke")
plt.plot(w,np.imag(QC_Green00_queb10k_V2(z)),label="ibm_quebec_V2")
plt.legend()
plt.xlabel("frequency ($\omega$)")
plt.ylabel("Spectral Weight")
plt.savefig("2siteDOSes.pdf")
# %%
MU = np.array([[-mu,0],[0,-mu]])
H_0 = np.array([[-mu,t],[t,-mu]])
N = len(w)
#For some reason there's a minus sign on all the quantum computed polynomial moments
# Very likely a bug. Does not happen with quantum simulation. Liouville-Lanczos' implementation is not dependent on
# wether we're dealing with a real QC or not. This hints at something fishy in qiskit... 
#Or its a bug that was present in Liouville-Lanczos when I performed the quantum computation.
#(e.g. bad sign on the Liouvillian. this would affect the alpha coefficient in the same way, but their value is pure noise for that problem.)
# Because real quantum hardware is (on the surface) involved, it's a bit expensive to test.
QC_Green01_queb1 = PolyCF_Green([],[],[-m[0] for k,m in enumerate(mu_queb10k)],QC_Green00_queb10k)
Green_q = Green_matrix([QC_Green00_queb10k.to_Lehmann(),QC_Green01_queb1.to_Lehmann()],2,[(0,0,0,1),(1,1,0,1),(0,1,1,1),(1,0,1,1)])
QC_Green01_sherx = PolyCF_Green([],[],[-m[0] for m in mu_sher10k],QC_Green00_sher10k)
Green_s = Green_matrix([QC_Green00_sher10k.to_Lehmann(),QC_Green01_sherx.to_Lehmann()],2,[(0,0,0,1),(1,1,0,1),(0,1,1,1),(1,0,1,1)])
#%% Compute GS energy with Galitsky-Migdal's formula
def fermi_dirac_T0(w):
    return w<=0.0
Kq = Green_q.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),H_0)#Kin+mu energy for one spin
muq = Green_q.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),MU)#Kin+mu energy for one spin
wG = Green_q.integrate_scalarfreq(lambda x: x*fermi_dirac_T0(x),np.eye(2))
E0 = (wG+Kq)#/2 missing because spin degeneracy
print("queb",E0)
Kq = Green_s.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),H_0)#Kin+mu energy for one spin
muq = Green_s.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),MU)#Kin+mu energy for one spin
wG = Green_s.integrate_scalarfreq(lambda x: x*fermi_dirac_T0(x),np.eye(2))
E0 = (wG+Kq)#/2 missing because spin degeneracy
print("sher",E0)
Kq = Green_ex.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),H_0)#Kin+mu energy for one spin
muq = Green_ex.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),MU)#Kin+mu energy for one spin
wG = Green_ex.integrate_scalarfreq(lambda x: x*fermi_dirac_T0(x),np.eye(2))
E0 = (wG+Kq)#/2 missing because spin degeneracy
print("exact",E0)
# %%
w = np.linspace(-6,6,1000)
z = w-0.1j
plt.plot(w,np.imag(T0_Green01(z)),label="exact")
plt.plot(w,np.imag(QC_Green01_queb1(z)),label="ibm_quebec")
plt.plot(w,np.imag(QC_Green01_sherx(z)),label="ibm_sherbrooke")
plt.legend()
plt.xlabel("frequency ($\omega$)")
plt.ylabel("Spectral Weight")
# plt.savefig("2siteDOSes.pdf")
plt.savefig("offDiagDOSes.pdf")
# %% 100k shots per averages
#from QC_hubbard2site_data_ibm_sherbrooke_2023-10-02 17:35:26.341417
a_sher100k = [-0.03251154758899268, 0.0710224328647117, -0.06457048201008102, 0.04465875313550449, 0.059131326795173605, -0.1382079686204271, 0.02281358881486545]
b_sher100k=[1.0, 2.236290696407671, 1.7888580866001078, 3.0700707879805216, 0.501457355697284, 3.837236663081367, 4.023550604945933]
mu_sher100k=[[0.0], [0.4471690561546307], [-0.010564759845977304], [-0.9220340482812656], [0.05529268146509505], [-0.16294435107894356], [-0.012141263144521058]]
QC_Green00_sher100k = CF_Green(a_sher100k,b_sher100k)
QC_Green01_sher100k= PolyCF_Green([],[],[-m[0] for m in mu_sher100k],QC_Green00_sher100k)
Green_s100k = Green_matrix([QC_Green00_sher100k.to_Lehmann(),QC_Green01_sher100k.to_Lehmann()],2,[(0,0,0,1),(1,1,0,1),(0,1,1,1),(1,0,1,1)])
# from QC_hubbard2site_data_ibm_quebec_2023-10-02 14:25:53.314127
a_queb100k = [0.01680300626304802, -0.014641915450875433, -0.028083376055043213, 0.052400740075163585, -0.17630320760789917, -0.06151792472675477, 0.13619615697076926, -0.2691556513498396, -0.007937486028779861, 0.1219565551340035]
b_queb100k=[1.0, 2.2360240652779497, 1.7888700609790429, 2.9883010436974797, 0.6831521365423429, 2.4153042772614723, 2.1908392878130787, 1.7037660561845271, 2.207563321606449, 2.3707140205694954]
mu_queb100k=[[0.0], [0.4472223781168002], [-0.004934793745336071], [-0.7233784667312323], [0.07202162481750417], [-0.43446495057811507], [-0.08184152518753443], [0.432423517277843], [-0.018322227876538408], [-0.12703786829727318]]
QC_Green00_queb100k = CF_Green(a_queb100k,b_queb100k)
QC_Green01_queb100k= PolyCF_Green([],[],[-m[0] for m in mu_queb100k],QC_Green00_queb100k)
Green_q100k = Green_matrix([QC_Green00_queb100k.to_Lehmann(),QC_Green01_queb100k.to_Lehmann()],2,[(0,0,0,1),(1,1,0,1),(0,1,1,1),(1,0,1,1)])
Kq = Green_q100k.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),H_0)#Kin+mu energy for one spin
muq = Green_q100k.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),MU)#Kin+mu energy for one spin
wG = Green_q100k.integrate_scalarfreq(lambda x: x*fermi_dirac_T0(x),np.eye(2))
E0 = (wG+Kq)#/2 missing because spin degeneracy
print(E0)
Kq = Green_s100k.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),H_0)#Kin+mu energy for one spin
muq = Green_s100k.integrate_scalarfreq(lambda x: fermi_dirac_T0(x),MU)#Kin+mu energy for one spin
wG = Green_s100k.integrate_scalarfreq(lambda x: x*fermi_dirac_T0(x),np.eye(2))
E0 = (wG+Kq)#/2 missing because spin degeneracy
print(E0)
#%%
def plotij(i,j):
    plt.plot(w,np.imag(Green_ex(z))[i,j,:],label="exact")
    plt.plot(w,np.imag(Green_q100k(z))[i,j,:],label="ibm_quebec")
    plt.plot(w,np.imag(Green_s100k(z))[i,j,:],label="ibm_sherbrooke")
    plt.legend()
plotij(0,0)
plotij(0,1)
# %%
plt.plot(b_queb10k[:5])
plt.plot(b_sher10k[:5])
plt.plot(b_ex)
# %%
plt.plot(b_queb100k[:5])
plt.plot(b_sher100k[:5])
plt.plot(b_ex)

# %%
plt.semilogy(np.abs(a_queb10k))
plt.semilogy(np.abs(a_sher10k))
plt.gca().set_ylim(1e-3,1)
# %%
plt.semilogy(np.abs(a_queb100k))
plt.semilogy(np.abs(a_sher100k))
plt.gca().set_ylim(1e-3,1)

# %%
