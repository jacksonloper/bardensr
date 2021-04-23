
import numpy as np
import scipy as sp
import scipy.ndimage

import logging
logger = logging.getLogger(__name__)

def kill_whitespace(s):
    return re.sub("\s+",s)


def quadratic_form_to_nnls_form(Gamma,phi,lo=1e-10):
    A=sp.linalg.cholesky(Gamma+np.eye(Gamma.shape[0])*lo,lower=False)
    b=np.linalg.solve(A.T,phi)
    return A,b
