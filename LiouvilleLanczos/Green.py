#%%
from abc import ABC,abstractmethod
import numpy as np


class Green_function_base(ABC):
    @abstractmethod
    def __call__(self,freq):
        ...


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
        return out


class Lehmann_Green(Green_function_base):
    def __init__(self, E,U):
        self.E = np.array(E)
        self.E = self.E.reshape(1,*E.shape)
        self.U =  np.array(U)
        self.U2 = np.einsum('ij...,mj...',self.U,self.U.conj())

    def __call__(self, freq):
        return np.sum(self.U2/(freq-self.E),axis=-1)
    
# %%
