

import quimb.tensor as tn



class environement_container():
    def __init__(self):
        self.env = []
    def __getitem__(self,i):
        return self.env[i+1]
    def __setitem__(self,i,newvalue):
        self.env[i+1]=newvalue
    

def update_left_env(state:tn.MatrixProductState,MPO:tn.MatrixProductOperator,env,i):
    """
    ┌──────┐     ┌────────┐        ┌──────┐   
    │      │     │    *   │        │      │   
    │      │  f  │   s    │        │      │  
    │      ├─────┤    i   ├─── a   │      ├─── a
    │      │     └────┬───┘        │      │   
    │      │          │ e          │      │   
    │      │     ┌────┴───┐        │      │   
    │  g   │     │        │       =│  g   │    
    │ E    │  g  │   h    │        │ E    │  
    │  i-1 ├─────┤    i   ├─── b   │  i   ├─── b
    │      │     └────┬───┘        │      │   
    │      │          │ d          │      │   
    │      │   h ┌────┴───┐        │      │   
    │      ├─────┤   s    ├─── c   │      ├─── c
    │      │     │    i   │        │      │   
    └──────┘     └────────┘        └──────┘   
    """
    Eim1 = env[i-1]
    statei = state[i]
    hi = MPO[i]
    stateic = statei.conj()
    h = list(set(stateic.inds).intersection(Eim1.inds()))[0]
    f = h+'AAA'
    stateic.reindex({h:f,})
    env[i] = (stateic&statei&hi&Eim1).contract()


def update_right_env(state,MPO,env,oc):
    ...
def initialise_env(state:tn.MatrixProductState,MPO:tn.MatrixProductOperator,oc):
    output = environement_container()
    output.env = [tn.Tensor()]*state.L
    for i in range(oc):
        update_left_env(state,MPO,output,i)
    for i in range(state.L,oc,-1):
        update_right_env(state,MPO,output,i)
    return output
def dmrg_update(state,MPO,env,oc):
    """
    state:MPS
    env: [env_gauche,env_droit]
    oc : site à mettre à jour
    return énergie aprés la mis à jour
    """
    #1 obtient la matrice local :H
    #détermine le réseaux à contracter pour obtenir H
    H = Res.contract()
    #2 calcul vap
    E,S = np.linalg.eigh(H)
    E = E[0]
    psi = S[0,:].reshape(xg,p,xd)
    state[oc] = tn.Tensor(psi,state[oc].inds,state[oc].tags)
    return E

def move_center(mps,current,destination):
    """
    déplace le centre d'orthogonalité vers la destination
    """
    ...

def DMRG(state:tn.MatrixProductState,MPO:tn.MatrixProductOperator,Energy_tol=1e-3,truncation_tol=1e-4):
    oc = state.calc_current_orthog_center()
    L = state.L
    assert(state.L == MPO.L, "incompatible MPS and MPO length")
    assert(state.phys_dim == MPO.phys_dim, "incompatible MPS and MPO hilbert space") 
    #quimb doesn't internally track the OC, but compute it on demand.
    try:
        oc = oc[0] #quimb can work with "partially" canonical MPS. We can ignore this.
    except:
        pass
    sweep_direction = 1
    env = initialise_env(state,MPO,oc)
    deltaE = 1000
    prev_E = 0
    curr_E = MPO.apply(state)@state.conj()#There's a cheaper way in terms of the env, but some consideration for the indices is necessary.
    while deltaE > Energy_tol:
        curr_E,prev_E = dmrg_update(state,MPO,env,oc),curr_E
        move_center(state,oc,oc+1)
        deltaE = abs(curr_E-prev_E)
        if sweep_direction > 0:
            update_left_env(state,MPO,env,oc)
        else:
            update_right_env(state,MPO,env,oc)
        sweep_direction += -2*(oc==(L-1)) + 2*(oc==0) 
        """
        This previous line is a condionnal free version of what follow
        if oc == L-1:
            sweep_direction = -1
        if oc == 0:
            sweep_direction = 1
        """
        oc += sweep_direction
