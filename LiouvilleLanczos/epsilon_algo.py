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



import numpy as np
from numpy.typing import NDArray

def epsilon(sequence:NDArray):
    """
    epsilon algorithm for extrapolation of convergent and some divergent sequence.
    For many divergent sequence, it will compute correctly the value of the
    function that generate the divergent sequence.
    See https://doi.org/10.1016/S0377-0427(00)00355-1 for more information.
    """
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