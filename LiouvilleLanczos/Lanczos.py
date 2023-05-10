import numpy as np



class Lanczos():
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
        f_i = f_0
        f_ip = self.Liouvillian(H,f_i)
        a_i = self.inner_prod(f_ip,f_i)
        if self.logger:
            self.logger(i,a_i,1)
        f_ip = self.sum(f_ip, - a_i*f_i)
        b_ip = np.sqrt(self.inner_prod(f_ip,f_ip))
        f_ip = f_ip / b_ip
        a = [a_i]
        b = []
        f_i,f_im = f_ip,f_i
        for i in range(1,max_k):
            if b_ip < min_b:
                return a,b
            if self.logger:
                self.logger.log(i,f_i,a[-1],b[-1])
            b.append(b_ip)
            f_ip = self.Liouvillian(H,f_i)
            try:
                a_i = self.inner_prod(f_ip,f_i)
                a.append(a_i)
            except:
                return a,b
            f_ip = self.sum(f_ip,- a_i*f_i,- b[-1]*f_im)
            try:
                b2 = self.inner_prod(f_ip,f_ip)
                assert b2>0
                b_ip = np.sqrt(b2)
            except:
                return a,b
            f_ip = f_ip / b_ip
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger.log(i,f_i,a[-1],b[-1])
        return a,b
    