#%%
import numpy as np


#experiments with 1d Hubbard model
#%% Choquette's statement about degenracy is only relevent with periodic boundary
U = 8
mu = 4
t = -1

L = 4
K = np.array([
    [0,1,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,1,0],
              ])
M = np.eye(2*L)
#%%
E,S = np.linalg.eigh(K)

from qiskit_nature.second_q.circuit.library import SlaterDeterminant,BogoliubovTransform
Slater = SlaterDeterminant(S.T[list(range(4))])
Slater.draw('mpl')
# %%

Kper = np.array([
    [0,1,0,1,0,0,0,0],
    [1,0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0,0],
    [1,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,1],
    [0,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,1],
    [0,0,0,0,1,0,1,0],
              ])

#alternating up-down...

Kper = np.array([
    [0,0,1,0,0,0,1,0],
    [0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0],
    [0,1,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,0],
    [0,0,0,1,0,0,0,1],
    [1,0,0,0,1,0,0,0],
    [0,1,0,0,0,1,0,0],
    ])

Ep,Sp = np.linalg.eigh(Kper)

print(Ep)

circuit = BogoliubovTransform(Sp)
circuit.draw('mpl')
# %%
