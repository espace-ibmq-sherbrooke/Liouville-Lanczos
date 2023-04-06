from quimb.tensor import MatrixProductOperator as MPO
from quimb.tensor import MatrixProductState as MPS

def Liouvillian(H:MPO,O:MPO,eps:float):
    H.site_tag_id = O.site_tag_id
    #une implémentation variationnel du commutateur pourrait donné un speedup significatif
    HO = O.apply(H)
    OH = -1*H.apply(O)
    OH.upper_ind_id = HO.upper_ind_id
    OH.lower_ind_id = HO.lower_ind_id
    return MPO_compressing_sum([HO,OH],eps,eps*0.5)

def inner_product(A:MPO,B:MPO,ket:MPS):
    """The Liouvillian inner product for the computation
    of Fermionic Green's function: <{A,B^\dagger}>_{psi}"""
    BH = B.H
    bra = ket.H
    A = A.copy()#so we don't modify the input args.
    bra.site_ind_id = 'z{}'
    A.upper_ind_id = 'x{}'
    A.lower_ind_id = ket.site_ind_id
    BH.upper_ind_id = A.upper_ind_id
    BH.lower_ind_id = bra.site_ind_id
    BA = (bra|BH|A|ket).contract()
    BH.upper_ind_id = ket.site_ind_id
    BH.lower_ind_id = 'y{}'
    A.lower_ind_id = 'y{}'
    A.upper_ind_id = bra.site_ind_id
    AB = (bra|A|BH|ket).contract()
    return AB+BA
