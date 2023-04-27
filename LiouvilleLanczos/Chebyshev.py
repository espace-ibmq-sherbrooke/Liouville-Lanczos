
import numpy as np


class Chebyshev():
    def __init__(self,inner_product,Liouvillian,sum, logger = None):
        self.inner_prod  = inner_product
        self._logger = logger
        self.Liouvillian = Liouvillian
        self.sum = sum

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def setter(self,new_logger):
        self._logger = new_logger
        
    def __call__(self,H,f_0,max_k,min_b=1e-12):
        i=0
        #f_{i+1} = 2Lf_i - f_{i-1}
        #mu_i = (f_i,f_0)
        #\im(G(\omega)) = 0 \forall |\omega| > 1
        mu = np.zeros(max_k)
        f_i = f_0
        mu_i = self.inner_prod(f_0,f_i)
        mu[i] = mu_i
        f_ip = self.Liouvillian(H,f_i)
        if self.logger:
            self.logger(i,f_i,mu_i)
        f_ip = self.sum(2*f_ip, -1*f_i)
        f_i,f_im = f_ip,f_i
        for i in range(1,max_k):
            mu_i = self.inner_prod(f_0,f_i)
            mu[i] = mu_i
            if self.logger:
                self.logger.log(i,f_i,mu_i)
            f_ip = self.Liouvillian(H,f_i)
            f_ip = self.sum(2*f_ip,- 1*f_im)
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger.log(i,f_i,mu_i)
        return mu_i