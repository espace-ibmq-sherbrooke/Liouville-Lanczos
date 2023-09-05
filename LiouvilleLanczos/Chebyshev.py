
import numpy as np


class triple_product:
    """
    class for the computation of the triple product
    $T_{ijk} = int_a^b f_i(x)f_j(x)f_k(x)d\mu(x)$
    where $\beta_{i+1}f_{i+1}(x) = (x-\alpha_i)f_i(x)-\beta_i f_{i-1}(x)$
    It can be shown that 
    $\beta_{i+1}T_{i+1,jk} = \beta_{j+1}T_{i,j+1,k} +  (\alpha_j-\alpha_i)T_{ijk} + \beta_j T_{i,j-1,k} * \beta_i T_{i-1,jk}$
    We obeserve that any to element have the same value if their index are a permutation of each other.
    $\int_a^bf_i(x)f_j(x)d\mu(x) = \delta_{ij}$ implies that $T_{ij0} = \delta_ij$
    and the polynomial nature of the $f_i$ implies that $T_{ijk}=0 \forall k>i+j$
    Because $f_i(x) = 0 \forall i<0$, $T_{ijk} = 0$ if any of $i,j$ or $k$ are less than 0.
    Tchebyshev polynomial have a very simple product structure, most of those formula can be greatly simplified in their case.
    """

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