from quimb.tensor import MatrixProductOperator as MPO
from quimb.tensor import MatrixProductState as MPS
from .MPO_compress import MPO_compressing_sum

from ..Lanczos_components import Inner_product,Summation
from ..Lanczos_components import Liouvillian as BaseLiouvillian

class Liouvillian(BaseLiouvillian):
    def __init__(self,eps) -> None:
        self.eps = eps

    def __call__(self,H:MPO,O:MPO):
        H.site_tag_id = O.site_tag_id
        #une implémentation variationnel du commutateur pourrait donné un speedup significatif
        HO = O.apply(H)
        OH = -1*H.apply(O)
        OH.upper_ind_id = HO.upper_ind_id
        H.lower_ind_id = HO.lower_ind_id
        return MPO_compressing_sum([HO,OH],self.eps,self.eps*0.5)
    
class inner_product(Inner_product):
    def __init__(self,ket:MPS):
        self.ket = ket
    def __call__(self,A:MPO,B:MPO,*args,**kwargs):
        """The Liouvillian inner product for the computation
        of zero temperature Fermionic Green's function: <{A,B^\dagger}>_{psi}"""
        BH = B.H
        bra = self.ket.H
        A = A.copy()#so we don't modify the input args.
        bra.site_ind_id = 'z{}'
        A.upper_ind_id = 'x{}'
        A.lower_ind_id = self.ket.site_ind_id
        BH.upper_ind_id = A.upper_ind_id
        BH.lower_ind_id = bra.site_ind_id
        BA = (bra|BH|A|self.ket).contract()
        BH.upper_ind_id = self.ket.site_ind_id
        BH.lower_ind_id = 'y{}'
        A.lower_ind_id = 'y{}'
        A.upper_ind_id = bra.site_ind_id
        AB = (bra|A|BH|self.ket).contract()
        return AB+BA
    
class Compressing_operator_sum(Summation):
    def __init__(self,eps) -> None:
        self.eps = eps
    def __call__(self,*ops):
        return MPO_compressing_sum(ops,self.eps,0.5*self.eps)