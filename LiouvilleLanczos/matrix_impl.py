import numpy as np
class Matrix_Liouvillian():
    def __call__(self,H,f):
        return H@f-f@H

class Matrix_sum():
    def __call__(self,*X):
        X = np.array(X)
        return np.add.reduce(X,axis=0)

class Matrix_inner_product():
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