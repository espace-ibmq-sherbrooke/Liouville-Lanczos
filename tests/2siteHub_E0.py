#%%
from qiskit.primitives import Estimator as pEstimator
from qiskit.primitives import backend_estimator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit import QuantumCircuit
import numpy as np
from qiskit import transpile
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel

#%%
#Construction du hamiltonien de Hubbard à 2 sites avec les outils de qiskit
N=2 #nommbre de site
t=-1
U = 4
mu=U/2
boundary_condition = BoundaryCondition.OPEN

line_lattice = LineLattice(num_nodes=N, boundary_condition=boundary_condition)
Ham = FermiHubbardModel(
line_lattice.uniform_parameters(
    uniform_interaction=t,
    uniform_onsite_potential=-mu,
),
onsite_interaction=U,
).second_q_op()

#%% calcul de la solution exacte numériquement
mapper = JordanWignerMapper()
qubit_jw_op = mapper.map(Ham)
Hmat = qubit_jw_op.to_matrix()
E,S = np.linalg.eigh(Hmat)
GS_mat = S[:,0]
print(f" calcul par diagonnalisation exacte {E[0]}")
#%%Circuit pour l'état fondamental, construit par inspection
# à partir de la solution analytique.
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
GS_analytical.draw('mpl')
#%% calcul de l'énergie total à partir d'un simulateur 
# d'ordinateur quantique
estimator = pEstimator() #prépare un objet estimator, celui ci est basé sur un simulateur
#La fonction d'un estimator est d'évalué la valeur moyenne d'un observable avec un circuit de préparation d'état.
HHam = qubit_jw_op #La hamiltonien d'électron est convertie en hamiltonien de qubit
job = estimator.run(GS_analytical,HHam) #créé une job. avec un vrai QC, ça nous placerai dans une file d'attente
result = job.result() # obtient le résultat. bloque l'instance python jusqu'à ce que le résultat soit disponible
values = result.values # les valeur moyenne qui ont été mesuré. result contient toute sorte d'autre métadonnées
print(f"E0 simulation QC: {values[0]}")
# %%
