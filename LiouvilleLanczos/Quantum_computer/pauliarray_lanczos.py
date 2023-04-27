


import pauliarray as pa
from qiskit import QuantumCircuit

# import pauliarray.

Op = pa.PauliOperator


class inner_product():
    
    def __init__(self,state:QuantumCircuit,estimator:BaseEstimator,mapper:QubitConverter,epsilon:int = 1e-10):
        ...
    
    def __call__(self,A:op.SparseLabelOp,B:op.SparseLabelOp):
        ...

class Liouvillian():

    def __init__(self,eps = 1e-10):
        ...
    def __call__(self,H,A):
        ...
    
class sum():
    def __init__(self,eps):
        ...
    def __call__(self,*X):
        ...