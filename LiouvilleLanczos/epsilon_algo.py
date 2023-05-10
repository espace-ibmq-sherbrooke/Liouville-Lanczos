

import numpy as np
from numpy.typing import NDArray

def epsilon(sequence:NDArray):
    e = np.array(sequence)
    em = np.zeros(len(e)+1,dtype=e.dtype)
    keep_going = len(e) >=2 and  ~np.any(np.isnan(e))
    c = 0
    while keep_going:
        print(c)
        print(e)
        d = (e[1:] - e[:-1])**(-1)
        print(d)
        print(em)
        ep = em[1:-1] + d
        em,e = e,ep
        c += 1
        keep_going = len(e) >=2 and ~np.any(np.isnan(e))
    if c%2:
        return em
    else:
        return e