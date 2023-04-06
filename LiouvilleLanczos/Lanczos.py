import numpy as np

 

class Lanczos():
    def __init__(self,inner_product,Liouvillian, logger = None):
        self.inner_prod  = inner_product
        self._logger = logger
        self.Liouvillian = Liouvillian

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def setter(self,new_logger):
        self._logger = new_logger
        
    def __call__(self,H,f_0,max_k,min_b=1e-12):
        k=0
        if self.logger:
            self.logger.log(k)
        f_i = f_0
        f_ip = self.Liouvillian(H,f_i)
        a_i = self.inner_prod(f_ip,f_i)
        f_ip = f_ip - a_i*f_i
        b_ip = np.sqrt(self.inner_prod(f_ip,f_ip))
        f_ip = f_ip / b_ip
        a = [a_i]
        b = []
        f_i,f_im = f_ip,f_i
        for k in range(1,max_k):
            if b_ip < min_b:
                return a,b
            if self.logger:
                self.logger.log(k)
            b.append(b_ip)
            f_ip = self.Liouvillian(H,f_i)
            a_i = self.inner_prod(f_ip,f_i)
            f_ip = f_ip - a_i*f_i - b[-1]*f_im
            b_ip = np.sqrt(self.inner_prod(f_ip,f_ip))
            f_ip = f_ip / b_ip
            a.append(a_i)
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger.log(k)
        return a,b