#%%
from LiouvilleLanczos.Tensor_networks import TN_lanczos as tnl
from LiouvilleLanczos.Tensor_networks import Operators as ops
from LiouvilleLanczos.Lanczos import Lanczos
import quimb.tensor as qtn
%load_ext snakeviz
#%%
el = [-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0]
er = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
tl = [0.5]*len(el)
tr = [0.5]*len(er)
U=8
mu=4

H = ops.Anderson_star(el,tl,U,mu,er,tr)
HH = ops.Hubbard_1D(1,U/2,U,20)
C_0 = ops.MPO_C(ops.c_up,ops.F,4,8)
N = ops.MPO_C(ops.c_up.T@ops.c_up,ops.id,4,8)
#%%
dmrg = qtn.DMRG(HH,bond_dims=1000,cutoffs=1e-12)
print(dmrg.solve(1e-16,max_sweeps=1000))
psi = dmrg.state
#%%
eps=1e-6
liou = tnl.Liouvillian(eps)
inpo = tnl.inner_product(psi)
#%%
# %%
#%%
lanczos = Lanczos(inpo,liou,tnl.Compressing_operator_sum(eps))
#%%
a,b = lanczos(H,C_0,20)
# %%
import cProfile
pr = cProfile.Profile()
pr.enable()
a,b = lanczos(H,C_0,20)
pr.disable()
# %%
import pstats, io
from pstats import SortKey
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
pr.dump_stats("stat.txt")
# %%
