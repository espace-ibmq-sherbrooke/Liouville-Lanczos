import numpy as np
class Matrix_Liouvillian():
    def __call__(self,H,f):
        return H@f-f@H

class Matrix_Hamiltonian():
    def __init__(self, E0,sign):
        self.E0 = E0
        self.sign = sign
    def __call__(self,H,f):
        return self.sign*(H-self.E0*np.eye(H.shape[0]))@f

class Matrix_sum():
    def __call__(self,*X):
        X = np.array(X)
        return np.add.reduce(X,axis=0)

class DensityMatrix_inner_product():
    def __init__(self,DensityMatrix):
        self.DM =DensityMatrix
    def __call__(self, A,B) -> float:
        Bd = np.conj(B.T)
        P = A@Bd
        T = Bd@A
        rP = self.DM@P
        rT = self.DM@T
        out = np.trace(rP)+np.trace(rT)
       #debug
        I = np.trace(  rP + rT )
        II = np.trace(  self.DM@(P + T) )
        assert I-out < 1e-4
        assert II-out < 1e-4
        # print(I-out)
       #!debug 
        return I
    
class Hamiltonian_inner_product():
    def __call__(self,A,B):
        return  np.conj(A.T)@B
    
class MatrixState_inner_product():
    def __init__(self,state):
        self.state =state
    def __call__(self, A,B) -> float:
        Bd = np.conj(B.T)
        # f = A@Bd + Bd@A
        sa = A@self.state
        sa = Bd@sa
        sb = Bd@self.state
        sb = A@sb
        stated = np.conj(self.state)
        return stated@(sa+sb)