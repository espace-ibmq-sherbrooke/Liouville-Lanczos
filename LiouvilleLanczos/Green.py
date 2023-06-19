#%%
from abc import ABC,abstractmethod
import numpy as np
from numpy import polynomial as poly
from typing import Optional
from scipy.sparse import csr_array

class Green_function_base(ABC):
    @abstractmethod
    def __call__(self,freq):
        ...


class Tcheb1st_Green(Green_function_base):
    def Gn(w,n):
        s = 1j*np.sqrt(1-(w-0.00000000000001j)**2)
        return -((w+s)**(n))/s
    def __init__(self,mu,scale,shift):
        self.mu = np.array(mu)
        self.scale = scale
        self.shift = shift
    def __call__(self,w):
        sw = w*self.scale-self.shift
        M = self.scale * np.array( [np.transpose(Tcheb1st_Green.Gn(sw,n)) for n,m in enumerate(self.mu) ])
        return M@self.mu
    
class Tcheb2nd_Green(Green_function_base):
    def Gn(w,n):
        s = 1j*np.sqrt(1-(w-0.00000000000001j)**2)
        return ((w+s)**(n+1))
    def __init__(self,mu,scale,shift):
        self.mu = np.array(mu)
        self.scale = scale
        self.shift = shift
    def __call__(self,w):
        sw = w*self.scale-self.shift
        M = self.scale * np.array( [np.transpose(Tcheb2nd_Green.Gn(sw,n)) for n,m in enumerate(self.mu) ])
        return M@self.mu
        

    

class CF_Green(Green_function_base):
    def __init__(self,a,b):
        self._a = a
        self._b = b
        #add checks on relative length of a and b

    @property
    def alpha(self):
       return self._a
    @property
    def beta(self):
        return self._b

    def __call__(self,freq,termination_function = 0):
        #might be more efficient to convert to lehman's representation.
        #doesn't allow for termination function.
        out = freq - self._a[-1] - termination_function
        for ai,bi in zip(self._a[-2::-1],self._b[-1:0:-1]):
            out = freq - ai - bi**2/out
        return self._b[0]**2/out

    def to_Lehmann(self):
        M = np.diag(self._a) + np.diag(self._b[1:],1)+np.diag(self._b[1:],-1)
        E,U = np.linalg.eigh(M)
        return Lehmann_Green(E,self._b[0]*U[0,:]) 

class Poly_Green_Base(Green_function_base):
    def __init__(self,moments) -> None:
        super().__init__()
        self._mu = moments
    @property
    def complex_weight(self):
        return self._Gd
    
    @property
    def moments(self):
        return self._mu
    
    @property
    @abstractmethod
    def alpha(self):
        ...

    @property
    @abstractmethod
    def beta(self):
        ...

    @property
    @abstractmethod
    def _Gd(self):
        ...

    def primary_0(self):
        return 1
    
    def secondary_1(self):
        -1/self.beta[1]

    def next_poly(self,freq,n,poly_n,poly_nm1):
        return self.beta[n+1]**-1*( (freq-self.alpha[n])*poly_n - self.beta[n]*poly_nm1)
    
    def primary_poly(self,w):
        Ln = self.primary_0()
        Lnm1 = 0 
        out = self.moments[0]*Ln
        n=1
        for mu in self.moments[1:]:
            Ln,Lnm1 = self.next_poly(w,n,Ln,Lnm1),Ln
            out+=Ln*mu
            n+=1
            
    def __call__(self,freq):
        Gf = self._Gd(freq)
        Ln = self.primary_0()
        Lnm1 = 0 
        out = self._mu[0]*Gf*Ln
        Qn = self.secondary_1()
        Qnm1 = 0
        Ln,Lnm1 = self.next_poly(freq,n,Ln,Lnm1),Ln
        out += self.moments[1]*(Qn+Ln*Gf)
        for i,mu_n in enumerate(self.moments[2:]):
            n=i+2
            Ln,Lnm1 = self.next_poly(freq,n,Ln,Lnm1),Ln
            Qn,Qnm1 = self.next_poly(freq,n+1,Qn,Qnm1),Qn
            out += mu_n(Qn+Ln*Gf)
        return out
    
class PolyCF_Green(Poly_Green_Base):
    def __init__(self,a,b,moments,CF_Green:Optional[CF_Green]=None) -> None:
        super().__init__(moments)
        if CF_Green is None:
            self._Gd = CF_Green(a,b)
        else:
            self._Gd = CF_Green
    @classmethod
    def from_shared_CF(cls,moments,CF_green:CF_Green):
        return PolyCF_Green(None,None,moments,CF_Green)
    @property
    def complex_weight(self):
        return self._Gd
    @property
    def alpha(self):
        return self.Gd.alpha
    @property
    def beta(self):
        return self.Gd.beta
    def to_Lehmann(self):
        return PolyLehmann_Green(self.alpha,self.beta,self.moments)

class PolyLehmann_Green(Poly_Green_Base):
    def __init__(self,a,b,moments) -> None:
        super().__init__(moments)
        self.__a=a
        self.__b=b
        self._Gd = CF_Green(a,b).to_Lehmann()
    @property
    def complex_weight(self):
        return self._Gd
    @property
    def alpha(self):
        return self.__a
    @property
    def beta(self):
        return self.__b
    
    def integrate(self,frequency_weights_function):
        mod_fw = lambda w: self.primary_poly(w)*frequency_weights_function(w)
        return self._Gd.integrate(mod_fw)

class Lehmann_Green(Green_function_base):
    def __init__(self,E,U):
        self._E = np.array(E)
        self._E = self._E.reshape(1,*E.shape)
        self._U =  np.array(U)
        if len(U.shape)>1:
            self._U2 = np.einsum('ij...,mj...',self._U,self._U.conj())
        else:
            self._U2 = np.abs(self._U)**2

    def __add__(self,other:"Lehmann_Green"):
        return Lehmann_Green(np.concat([self._E,other._E]),np.concat(self._U,other._U))

    @property
    def Poles(self):
        return self._E
    @property
    def Weights(self):
        return self._U

    def __call__(self, freq):
        if isinstance(freq,np.ndarray):
            return np.sum(self._U2/(np.expand_dims(freq,-1)-self._E),axis=-1)
        else:
            return np.sum(self._U2/(freq-self._E),axis=-1)
    
    def integrate(self,frequency_weights_function):
        return np.sum(self._U2*frequency_weights_function(self._E))
    
class Green_matrix(Green_function_base):
    def __init__(self,Green_list,matrix_size,position_maps):
        self.Green_list = Green_list
        self.size = matrix_size
        self.position_maps = self.prepare_position_maps(position_maps)

    def prepare_position_maps(self,raw_position_map):
        #convert matrix positions in the raw map.
        def convert_matpos(mat_pos):
            r,c,k = mat_pos
            assert r < self.size
            assert c < self.size
            assert k < len(self.Green_list)
            return r+c*self.size,k
        R,C,D = np.zeros(len(raw_position_map),dtype=np.int64),np.zeros(len(raw_position_map),dtype=np.int64),np.zeros(len(raw_position_map),dtype=np.complex128)
        for i,(pos,coeff) in enumerate(raw_position_map):
            R[i],C[i] = convert_matpos(pos)
            D[i] = coeff
        out = csr_array(D,(R,C) ,shape=(self.size**2,len(self.Green_list)))
        return out

    def __call__(self, freq):
        """évalue les élément de la fonction de Green et les combines selon le contenue de position_maps"""
        GL = np.zeros(len(self.Green_list),dtype=np.complex128)
        for i,g in enumerate(self.Green_list):
            GL[i] = g(freq)
        out = self.position_maps@GL
        return out.reshape((self.size,self.size)) 
# %%
