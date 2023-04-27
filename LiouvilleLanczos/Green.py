#%%
from abc import ABC,abstractmethod
import numpy as np
from numpy import polynomial as poly

class Green_function_base(ABC):
    @abstractmethod
    def __call__(self,freq):
        ...


class Tcheb_Green(Green_function_base):
    
    def Gn(w,n):
        m = (-2*(w<0)+1)
        s = np.sqrt((w-0.00000000000001j)**2-1)
        return m*((w-m*s)**(n))/s
    def __init__(self,mu,scale,shift):
        self.mu = np.array(mu)
        self.scale = scale
        self.shift = shift
    def __call__(self,w):
        sw = w*self.scale-self.shift
        M = self.scale * np.array( [np.transpose(Tcheb_Green.Gn(sw,n)) for n,m in enumerate(self.mu) ])
        return M@self.mu
        

class CF_Green(Green_function_base):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        #add checks on relative length of a and b

    def __call__(self,freq,termination_function = 0):
        #might be more efficient to convert to lehman's representation.
        #doesn't allow for termination function.
        out = freq - self.a[-1] - termination_function
        for ai,bi in zip(self.a[-2::-1],self.b[-1::-1]):
            out = freq - ai - bi**2/out
        return 1/out

    def to_Lehmann(self):
        M = np.diag(self.a) + np.diag(self.b,1)+np.diag(self.b,-1)
        E,U = np.linalg.eigh(M)
        return Lehmann_Green(E,U[:,0]) 

class Lehmann_Green(Green_function_base):
    def __init__(self, E,U):
        self.E = np.array(E)
        self.E = self.E.reshape(1,*E.shape)
        self.U =  np.array(U)
        self.U2 = np.einsum('ij...,mj...',self.U,self.U.conj())

    def __call__(self, freq):
        return np.sum(self.U2/(freq-self.E),axis=-1)
    
# %%
