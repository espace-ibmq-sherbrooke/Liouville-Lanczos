import numpy as np
import quimb.tensor as qtn
from quimb.tensor import MatrixProductOperator as MPO
from quimb.tensor import MatrixProductState as MPS

c_up = np.zeros((4,4))
c_up[0,1] = 1
c_up[2,3] = 1
c_dn = np.zeros((4,4))
c_dn[0,2] = 1
c_dn[1,3] = -1
F = np.eye(4)
F[1,1] = -1
F[2,2] = -1
id = np.eye(4)
n_up = c_up.T@c_up
n_dn = c_dn.T@c_dn

def MPO_C(local_c,F,s,N):
    tens = []
    assert(F.shape == local_c.shape)
    i = 0
    while i<s:
        tens.append(np.array([[F]]))
        i+=1
    tens.append(np.array([[local_c]]))
    i+=1
    while i < N:
        tens.append(np.array([[np.eye(F.shape[0])]]))
        i+=1
    tens[0] = tens[0][:,0,:,:]
    tens[-1] = tens[-1][0,:,:,:]
    return MPO(tens)
    

def Hubbard_1D(t,mu,U,N):
    tens = np.zeros((6,6,4,4))
    tens[0,0,:,:] = id
    tens[5,5,:,:] = id
    tens[5,1,:,:] = (F@c_up)
    tens[5,2,:,:] = (c_up.T@F)
    tens[5,3,:,:] = (F@c_dn)
    tens[5,4,:,:] = (c_dn.T@F)
    tens[1,0,:,:] = t*(c_up.T)
    tens[2,0,:,:] = t*(c_up)
    tens[3,0,:,:] = t*(c_dn.T)
    tens[4,0,:,:] = t*(c_dn)
    tens[5,0,:,:] = U*(n_up@n_dn) - mu*(n_up+n_dn)
    tensors = [tens for i in range(N)]
    tensors[0] = tensors[0][5,:,:,:]
    if len(tensors) > 1:
        tensors[-1] = tensors[-1][:,0,:,:]
    else:
        tensors[0] = tensors[0][0,:,:]
    return qtn.MatrixProductOperator(tensors)


def Anderson_star(el,tl,U,mu,er,tr):
    tensors = []
    for e,t in zip(el,tl):
        tens = np.zeros((6,6,4,4))
        tens[0,0,:,:] = id
        tens[1,1,:,:] = F
        tens[2,2,:,:] = F
        tens[3,3,:,:] = F
        tens[4,4,:,:] = F
        tens[5,5,:,:] = id
        tens[5,0,:,:] = e*(n_up+n_dn)
        tens[5,1,:,:] = t*(F@c_up)
        tens[5,2,:,:] = t*(c_up.T@F)
        tens[5,3,:,:] = t*(F@c_dn)
        tens[5,4,:,:] = t*(c_dn.T@F)
        tensors.append(tens)
    tens = np.zeros((6,6,4,4))
    tens[0,0,:,:] = id
    tens[5,5,:,:] = id
    tens[5,1,:,:] = (F@c_up)
    tens[5,2,:,:] = (c_up.T@F)
    tens[5,3,:,:] = (F@c_dn)
    tens[5,4,:,:] = (c_dn.T@F)
    tens[1,0,:,:] = (c_up.T)
    tens[2,0,:,:] = (c_up)
    tens[3,0,:,:] = (c_dn.T)
    tens[4,0,:,:] = (c_dn)
    tens[5,0,:,:] = U*(n_up@n_dn) - mu*(n_up+n_dn)
    tensors.append(tens)
    for e,t in zip(er,tr):
        tens = np.zeros((6,6,4,4))
        tens[0,0,:,:] = id
        tens[1,1,:,:] = F
        tens[2,2,:,:] = F
        tens[3,3,:,:] = F
        tens[4,4,:,:] = F
        tens[5,5,:,:] = id
        tens[5,0,:,:] = e*(n_up+n_dn)
        tens[1,0,:,:] = t*(c_up.T)
        tens[2,0,:,:] = t*(c_up)
        tens[3,0,:,:] = t*(c_dn.T)
        tens[4,0,:,:] = t*(c_dn)
        tensors.append(tens)
    tensors[0] = tensors[0][5,:,:,:]
    if len(tensors) > 1:
        tensors[-1] = tensors[-1][:,0,:,:]
    else:
        tensors[0] = tensors[0][0,:,:]
    return qtn.MatrixProductOperator(tensors)