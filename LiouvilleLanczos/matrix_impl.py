"""
    Liouville-Lanczos: A library for Many-Body Green's function on quantum and classical computer.
    Copyright (C) 2024  Alexandre Foley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from .Lanczos_components import Inner_product,Liouvillian,Summation
class Matrix_Liouvillian(Liouvillian):
    def __call__(self,H,f):
        return H@f-f@H

class Matrix_Hamiltonian(Liouvillian):
    def __init__(self, E0,sign):
        self.E0 = E0
        self.sign = sign
    def __call__(self,H,f):
        return self.sign*(H-self.E0*np.eye(H.shape[0]))@f

class Matrix_sum(Summation):
    def __call__(self,*X):
        X = np.array(X)
        return np.add.reduce(X,axis=0)

class DensityMatrix_inner_product(Summation):
    def __init__(self,DensityMatrix):
        self.DM =DensityMatrix
    def __call__(self, A,B,*args,**kwargs) -> float:
        Bd = np.conj(B.T)
        P = A@Bd
        T = Bd@A
        rP = self.DM@P
        rT = self.DM@T
        O = np.trace(rP)+np.trace(rT)
       #debug
       # for some reason, the different possible order 
       # of evaluation don't behave in the same way. 
       # They give drastically different results when close to Eigenspace depletion.
        # I = np.trace(  rP + rT )
        II = np.trace(  self.DM@(P + T) ) #most reliably precise so far.
        III = np.einsum('ij,ji',self.DM,P+T) #in principle slightly faster than 2, untested
        # IIII = np.einsum('ij,jk,ki',self.DM,A,Bd) + np.einsum('ij,jk,ki',self.DM,Bd,A)
        out = II
        print(f"DMIP instability 3:{III}, 2:{II}, 0:{O}")
        # assert abs(I-out) < 1e-10
        # assert abs( II-out) < 1e-10
        # assert abs( III-out) < 1e-10
        # assert abs( IIII-out) < 1e-10
        # assert abs( O-out) < 1e-10
       #!debug 
        return out
    
class Hamiltonian_inner_product(Inner_product):
    def __call__(self,A,B):
        return  np.conj(A.T)@B
    
class MatrixState_inner_product(Inner_product):
    def __init__(self,state):
        self.state =state
    def __call__(self, A,B,*args,**kwargs) -> float:
        Bd = np.conj(B.T)
        # f = A@Bd + Bd@A
        sa = A@self.state
        sa = Bd@sa
        sb = Bd@self.state
        sb = A@sb
        stated = np.conj(self.state)
        return stated@(sa+sb)