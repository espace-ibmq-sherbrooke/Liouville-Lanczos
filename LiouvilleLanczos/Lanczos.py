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

from .Lanczos_components import Inner_product,Liouvillian,Summation

class Lanczos():
    def __init__(self,inner_product:Inner_product,liouvillian:Liouvillian,sum:Summation,epsilon=1e-13 ,logger = None):
        self.inner_prod  = inner_product
        self._logger = logger
        self.Liouvillian = liouvillian
        self.sum = sum
        self.epsilon = epsilon

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def setter(self,new_logger):
        self._logger = new_logger

    def polynomial_hybrid(self,H,f_0,other_ops,max_k,min_b=1e-10):
        i=0
        b = [np.sqrt(self.inner_prod(f_0,f_0,Name="b0"))]
        f_i = f_0/b[-1]
        multimoments = [[self.inner_prod(o,f_i,Name=f"m{m}_{0}") for m,o in enumerate(other_ops)] ]
        f_ip = self.Liouvillian(-H,f_i)
        a_i = self.inner_prod(f_ip,f_i,Name="a0")
        if self.logger:
            self.logger(i,f_i,a_i,b)
        f_ip = self.sum(f_ip, - a_i*f_i)
        b_ip = np.sqrt(self.inner_prod(f_ip,f_ip,Name="b1"))
        f_ip = f_ip / b_ip
        a = [a_i]
        f_i,f_im = f_ip,f_i
        for i in range(1,max_k):
            if b_ip < min_b:
                return a,b,multimoments
            if self.logger:
                self.logger(i,f_i,a[-1],b[-1])
            multimoments.append([self.inner_prod(o,f_i,Name=f"m{m}_{i}") for m,o in enumerate(other_ops)])
            f_ip = self.Liouvillian(-H,f_i)
            try:
                a_i = self.inner_prod(f_ip,f_i,Name=f"a_{i}")
                a.append(a_i)
            except Exception as e:
                print(f"anomalous termination a at iteration {i}")
                print(e)
                return a,b,multimoments
            b.append(b_ip)
            f_ip = self.sum(f_ip,- a_i*f_i,- b[-1]*f_im)
            try:
                b2 = self.inner_prod(f_ip,f_ip,Name=f"b^2_{i+1}")
                assert b2>self.epsilon , f"b^2={b2} is smaller than {self.epsilon}, terminating"
                b_ip = np.sqrt(b2)
            except Exception as e:
                print(f"anomalous termination b at iteration {i}")
                print(e)
                return a,b,multimoments
            f_ip = f_ip / b_ip
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger(i,f_i,a[-1],b[-1])
        return a,b,multimoments
         
    def __call__(self,H,f_0,max_k,min_b=1e-10):
        a,b,_ = self.polynomial_hybrid(H,f_0,[],max_k,min_b)
        return a,b
    