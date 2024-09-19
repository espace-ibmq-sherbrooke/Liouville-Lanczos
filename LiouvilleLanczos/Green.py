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

#%%
from abc import ABC,abstractmethod
import numpy as np
from numpy import polynomial as poly
from numpy.typing import NDArray
from typing import Optional,Callable
from scipy.sparse import csr_array
# from multimethod import multimethod

class Callable_Green_function_base(ABC):
    """
    Trait for Green's function that can be evaluated at any complex frequencies
    """
    @abstractmethod
    def __call__(self,freq:complex):
        ...

class integrable_Green_function_base(Callable_Green_function_base):
    """
    Trait for Green's functions that implement an efficient "integrate" method
    """
    @abstractmethod
    def integrate(self,frequency_weights_function:Callable):
        ...

class Tcheb1st_Green(Callable_Green_function_base):
    """
    Tchebyshev polynomial of the first kind reprensention of the
    Green's function
    """
    def Gn(w,n):
        """
        Stieltjes transform of T_n(w)/sqrt(1-x^2) where T_n is the nth 
        Tchebyshev polynomial of the first kind.
        """
        tinytyni_j = 1e-14j#Tiny tyni imaginary part to make the complex root give predictible results on the real line.
        s = 1j*np.sqrt(1-(w-tinytyni_j)**2)
        return -((w+s)**(n))/s
    def __init__(self,mu,scale,shift):
        """
        Construct the Green function as a function of frequency as stieltjes 
        transform weighted of linear combination of Tchebyshev polynomials of the
        first kind.
        The Techbyshev moments mu are the coefficient before each of the 
        Tchebyshev polynomials.
        
        Tchebyshev polynomials are conventionnaly defined on the [-1,1] domain.
        the scale argument defines the scaling factor to bring the polynomial to
        the desired bandwidth.
        the shift argument serves to center the domain at the right position.
        The imaginary part of the Green function is only non-zero on that specific
        part of the real line. 
        """
        self.mu = np.array(mu)
        self.scale = scale
        self.shift = shift
    def __call__(self,w):
        """
        evaluate the Green function at the (complex or real) frequency w.
        """
        sw = w*self.scale-self.shift
        M = self.scale * np.array( [np.transpose(Tcheb1st_Green.Gn(sw,n)) for n,m in enumerate(self.mu) ])
        return M@self.mu
    
class Tcheb2nd_Green(Callable_Green_function_base):
    """
    Tchebyshev polynomial of the second kind reprensentation of the
    Green's function
    """
    def Gn(w,n):
        """
        Stieltjes transform of U_n(w)/sqrt(1-x^2) where U_n is the nth 
        Tchebyshev polynomial of the second kind.
        """
        tinytyni_j = 1e-14j#Tiny tyni imaginary part to make the complex root give predictible results on the real line.
        s = 1j*np.sqrt(1-(w-tinytyni_j)**2)
        return ((w+s)**(n+1))
    def __init__(self,mu,scale,shift):
        """
        Construct the Green function as a function of frequency as stieltjes 
        transform weighted of linear combination of Tchebyshev polynomials of the
        second kind.
        The Techbyshev moments mu are the coefficient before each of the 
        Tchebyshev polynomials.
        
        Tchebyshev polynomials are conventionnaly defined on the [-1,1] domain.
        the scale argument defines the scaling factor to bring the polynomial to
        the desired bandwidth.
        the shift argument serves to center the domain at the right position.
        The imaginary part of the Green function is only non-zero on that specific
        part of the real line. 
        """
        self.mu = np.array(mu)
        self.scale = scale
        self.shift = shift
    def __call__(self,w):
        sw = w*self.scale-self.shift
        M = self.scale * np.array( [np.transpose(Tcheb2nd_Green.Gn(sw,n)) for n,m in enumerate(self.mu) ])
        return M@self.mu
        

    

class CF_Green(Callable_Green_function_base):
    """
    Continued fraction reprensation Green's function.
    If you are not interested in using special termination functions,
    The Lehmann represenation is computationnaly more efficient.
    A conversion method is apart of this class.
    """
    def __init__(self,a,b):
        """
        construct a continued fraction representation of the Green function, 
        from the a,b coefficient produced by Lanczos algorithm.
        The square of the b[0] coefficient is a scaling factor on the 
        Green's function.
        """
        self._a = a
        self._b = b
        assert len(a)==len(b), "the two list of coefficient should have the same size!"
        #add checks on relative length of a and b

    @property
    def alpha(self):
       return self._a
    @property
    def beta(self):
        return self._b

    def __call__(self,freq,termination_function = 0):
        """
        Compute the Green function as a continued fraction, begining by the bottom
        floor of the continued fraction.
        A termination function can be supplied, which will be inserted in the 
        last floor. A suitably chosen termination function extrapolate the continued
        fraction to infite number of floor limit.
        """
        #might be more efficient to convert to lehman's representation.
        #doesn't allow for termination function.
        out = freq - self._a[-1] - termination_function
        for ai,bi in zip(self._a[-2::-1],self._b[-1:0:-1]):
            out = freq - ai - bi**2/out
        return self._b[0]**2/out

    def to_Lehmann(self):
        """
        Convert the continued fraction representation to the Lehman representation
        of the Green function. This representation is more efficient from a computationnal
        standpoint, but injecting a termination
        """
        M = np.diag(self._a) + np.diag(self._b[1:],1)+np.diag(self._b[1:],-1)
        E,U = np.linalg.eigh(M)
        return Lehmann_Green(E,self._b[0]*U[0,:]) 

class Poly_Green_Base(Callable_Green_function_base):
    """
    Generic base for polynomial specified by the three term recursion coefficents.
    """
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
        """Order 0 polynomial of the first kind"""
        return 1
    
    def secondary_1(self):
        """order 0 polynomial of the second kind, labeled 1, as it is produced 
        by the stieltjes transform of the order 1 polynomial of the first kind 
        mulitplied with the weight function """
        return -1/self.beta[1]

    def next_poly(self,freq,n,poly_n,poly_nm1):
        """
        Compute the next polynomial of either kind at a specific frequency using 
        the previous two polynomials of the same kind.

        freq is the frequency at which the polynomial is evaluated.

        n is the order of poly_n. It is used to fetch the correct coefficient of
        the three term recursion.

        poly_n is the polynomial of degree n evaluated at freq.
        poly_nm1 is the polynomial of degree n-1 evalueted at freq.

        return the polynomial of degree n+1 at freq.
        """
        return self.beta[n+1]**-1*( (freq-self.alpha[n])*poly_n - self.beta[n]*poly_nm1)
    
    def eval_poly(self,w,kind,order):
        """
        Evaluate the primary or secondary polynomial of a given order at specified 
        frequency grid w. This implementation is not efficient if all polynomial 
        up to order n are needed. TODO: make this possibility efficient.
        
        w: frequency (grid)
        kind: the kind of polynomial to compute, 
              0 or "primary" for the first kind
              1 or "secondary" for the second kind
        order: order of the desired polynomial.

        """
        assert(kind == 0 or kind == 1 or kind == "primary" or kind == "secondary")
        if kind != 0:
            #reduce the number of possibilities
            kind = (kind == "secondary")*1
        if kind == 0:
            n = 0
            Ln = self.primary_0()
            Lnm1 = 0 
        if kind == 1:
            Lnm1 = 0
            if order == 0:
                #the secondary polynomial with label 0 is conventionnally 0.
                return Lnm1
            n = 1
            Ln = self.secondary_1()
        while n<order:
            n+=1
            Ln,Lnm1 = self.next_poly(w,n,Ln,Lnm1),Ln
        return Ln


    def secondary_poly_sum(self,w):
        """
        Sum the secondary polynomials at frequency w, each multiplied by the corresponding
        moment.
        """
        Qn = self.secondary_1()
        Qnm1 = 0 
        out = self.moments[1]*Qn
        n=1
        for mu in self.moments[2:]:
            Qn,Qnm1 = self.next_poly(w,n,Qn,Qnm1),Qn
            out+=Qn*mu
            n+=1
        return out

    def primary_poly_sum(self,w):
        """
        Sum the primary polynomials at frequency w, each multiplied by the corresponding
        moment.
        """
        Ln = self.primary_0()
        Lnm1 = 0 
        out = self.moments[0]*Ln
        n=0
        for mu in self.moments[1:]:
            Ln,Lnm1 = self.next_poly(w,n,Ln,Lnm1),Ln
            out+=Ln*mu
            n+=1
        return out
            
    def __call__(self,freq):
        """
        evaluate de Green function at frequency freq
        """
        Gf = self._Gd(freq)
        L = self.primary_poly_sum(freq)
        Q = self.secondary_poly_sum(freq)
        return Q+L*Gf
    
class PolyCF_Green(Poly_Green_Base):
    """
    Polynomial representation of the Green function with a continued fraction represented
    weighing function.

    This class implement a conversion to PolyLehmann_Green, which offer more efficient 
    frequency integration routines.
    """
    def __init__(self,moments,CF_Green:CF_Green) -> None:
        """
        Construct a polynomial decomposed Green function from a set of moments and
        a continued fraction represented weighting function.

        moments: the coefficient of each polynomials
        CF_Green: the weighting function in a continued fraction representation. 
                  provides the recursion coefficient as well.
        """
        super().__init__(moments)
        self.__Gd = CF_Green
    @classmethod
    def from_shared_CF(cls,moments,CF_green:CF_Green):
        return PolyCF_Green(None,None,moments,CF_Green)
    @property
    def complex_weight(self):
        return self._Gd
    @property
    def _Gd(self):
        return self.__Gd
    @property
    def alpha(self):
        return self._Gd.alpha
    @property
    def beta(self):
        return self._Gd.beta
    def to_Lehmann(self):
        return PolyLehmann_Green(self.alpha,self.beta,self.moments)
    def call(self,freq,termination_function):
        """
        evaluate de Green function at frequency freq, with a termination function in the continued fraction.

        if a termination function is not desired, use the parent class provided __call__ operator
        """
        Gf = self._Gd(freq,termination_function)
        L = self.primary_poly_sum(freq)
        Q = self.secondary_poly_sum(freq)
        return Q+L*Gf


class Lehmann_Green(integrable_Green_function_base):
    """
    Lehmann reprensentation for the Green's function. Provided efficient frequency
    integration routines.
    The weight can be matrices.
    """
    def __init__(self,E,U):
        """
        Construct the Lehmann reprensentation from the set of poles positions E and 
        their complex weight U.
        """
        self._E = np.array(E)
        self._E = self._E.reshape(*E.shape)
        self._U =  np.array(U)
        self._U2 = Lehmann_Green.produce_U2(self._U)
    
    @staticmethod
    def produce_U2(U):
        if len(U.shape)>1:
            U2 = np.einsum('ij...,mj...',U,U.conj())
        else:
            U2 = np.abs(U)**2
        # U2 = U2.reshape((*U2.shape,1))
        return U2
        
    def __add__(self,other:"Lehmann_Green"):
        return Lehmann_Green(np.concat([self._E,other._E]),np.concat(self._U,other._U))

    @property
    def Poles(self):
        """Set of poles"""
        return self._E
    @property
    def Weights(self):
        """complex weights"""
        return self._U

    def __call__(self, freq):
        """
        Evaluate the Green function at frequency freq, accepts a set of frequency 
        in the form of a numpy array
        """
        if isinstance(freq,np.ndarray):
            fshape = freq.shape
            efreqshape = (*fshape,*[1 for i in range(len(self._U2.shape))])
            efreq = freq.reshape(efreqshape)
            return np.sum(self._U2/(efreq-self._E),axis=-1)
        else:
            return np.sum(self._U2/(np.expand_dims(freq,-1)-self._E),axis=-1)
    
    def integrate(self,frequency_weights_function):
        """
        Integrate over frequencies the product of the Green function with an analytical
        function (analytical at the poles of the Green's function).
        """
        return np.sum(self._U2*frequency_weights_function(self._E),axis=-1)

class PolyLehmann_Green(Poly_Green_Base,integrable_Green_function_base):
    """
    Polynomial representation of the Green's function with the weighing function represented as 
    a sum of simple poles (Lehmann's representation)
    """
    def __init__(self,a,b,moments,Lehmann_green:Optional[Lehmann_Green]=None) -> None:
        """
        construct the Green function from the recursion coefficients and polynomial
        moments. Optionnally accept a pre-constructed Lehmann represented Weight 
        function.
        If the Weigh function is not supplied it will be constructed from the recursion coefficients.
        """
        super().__init__(moments=moments)
        if Lehmann_green is None:
            self.__Gd = CF_Green(a,b).to_Lehmann()
        else:
            self.__Gd = Lehmann_green
        self.__a=a
        self.__b=b
    @staticmethod
    def from_Lehmann_Green(a,b,moments,Lehmann_green:Lehmann_Green):
        return PolyLehmann_Green(a,b,moments,Lehmann_green)
    @property
    def _Gd(self):
        return self.__Gd
    @property
    def complex_weight(self):
        """
        Weight function
        """
        return self._Gd
    @property
    def alpha(self):
        """
        alpha coefficients of the three terms recursion
        """
        return self.__a
    @property
    def beta(self):
        """
        beta coefficients of the three terms recursion
        """
        return self.__b
    
    def integrate(self,frequency_weights_function):
        """
        Integrate over frequencies the product of the Green function with an analytical
        function (analytical at the poles of the Green's function).
        """
        mod_fw = lambda w: self.primary_poly_sum(w)*frequency_weights_function(w)
        return self._Gd.integrate(mod_fw)
    
class Green_matrix(integrable_Green_function_base):
    """
    Matrix Green function, from irreducible elements. Provides fast frequency integration
    routines if all irreducible elements provide integration routines.
    """
    def __init__(self,Green_list,matrix_size,position_maps):
        """
        Construct a matrix Green function from irreducible elements and a mapping 
        to matrix index.

        Green_list: list of Green function, the irreducible elements.
        matrix_size: size of the matrix Green function
        position_map: how to add the irreducible elements to produce each of the 
        reducible matrix elements.

        The position map should take the form of a list of tuple, each tuple should
        have the format (i,j,g,w)
        where (i,j) are matrix position, g is the index of an irreducible Green 
        function, and w is a scalar weight to apply to the Green function
        """
        self.Green_list = Green_list
        self.size = matrix_size
        self.position_maps = self.prepare_position_maps(position_maps)

    def prepare_position_maps(self,raw_position_map):
        '''
        convert matrix positions in the user supplied map to a format more suitable
        to computations.
        '''
        def convert_matpos(mat_pos):
            r,c,k = mat_pos
            assert r < self.size
            assert c < self.size
            assert k < len(self.Green_list)
            return r+c*self.size,k
        R,C,D = np.zeros(len(raw_position_map),dtype=np.int64),np.zeros(len(raw_position_map),dtype=np.int64),np.zeros(len(raw_position_map),dtype=np.complex128)
        for i,(il,jl,kl,coeff) in enumerate(raw_position_map):
            R[i],C[i] = convert_matpos((il,jl,kl))
            D[i] = coeff
        out = csr_array((D,(R,C)) ,shape=(self.size**2,len(self.Green_list)))
        return out

    def __call__(self, freq):
        """
        Evaluate the matrix valued Green function at frequency freq.

        the frequency can be a numpy array of frequencies
        """
        Lfreq = [1]
        if type(freq) is np.ndarray:
            Lfreq = freq.shape
        GL = np.zeros((len(self.Green_list),*Lfreq),dtype=np.complex128)
        for i,g in enumerate(self.Green_list):
            GL[i,:] = g(freq)
        out = self.position_maps@GL
        return out.reshape((self.size,self.size,*Lfreq))
    
    def __Check_integrable(self):
        """
        Verify the all the irreducible Green function provide an integration routine.
        """
        for i,g in enumerate(self.Green_list):
            if not issubclass(type(g),integrable_Green_function_base):
               try:
                    self.Green_list[i] = g.to_Lehmann()
               except:
                    raise TypeError(f"Green function element {i} representation is not Lehmann and cannot be converted")

    # @multimethod
    def integrate_scalarfreq(self,scalar_frequency_weights_function:Callable,matrix_frequency_constant:NDArray):
        """
        integrate G(w)*f(w) where G in the matrix Green function and f(w) is an 
        analytical scalar function.
        """
        self.__Check_integrable()
        GI =  np.array([g.integrate(scalar_frequency_weights_function) for i,g in enumerate(self.Green_list)])
        fm = matrix_frequency_constant.flatten()
        return fm@self.position_maps@GI

    
    # @multimethod
    def integrate(self,frequency_weights_function:Callable):
        """
        integrate G(w).F(w) where G is the Green function and F(w) is a matrix valued
        analytical function.
        """
        self.__Check_integrable()
        #Give ok result only for diagonnal functions
        pm_size = self.position_maps.shape[0]
        def integrand(w,i):
            fwf = frequency_weights_function(w)
            fwfl =  np.prod( fwf.shape[:-2],dtype=np.int64)
            fwf = fwf.reshape((fwfl,pm_size))
            return (fwf@self.position_maps[:,[i]] ).squeeze()
        return np.sum([g.integrate(lambda w: integrand(w,i)) for i,g in enumerate(self.Green_list) ])
# %%
