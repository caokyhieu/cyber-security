import jax.random as jxr
import jax.numpy as jnp
from jax import jit,vmap
from utils.pyro_utils import cg_inverse
from covariance.covariance_func import (squared_exponential_covariance,rational_quadratic_kernel,
                                        matern_kernel,matern_spatial_kernel,squared_exponential_spatial_covariance,
                                        vonMisesFisherkernel, periodic_kernel,linear_kernel, polynomial_kernel,
                                        locally_periodic_kernel,sm_kernel_single,ARD_kernel,
                                        # composite kernel
                                        linear_trend_multiply_matern_kernel,
                                        linear_trend_multiply_periodicity_kernel,
                                        linear_trend_multiply_rational_quadratic_kernel,
                                        linear_trend_multiply_square_exponential_kernel,
                                        linear_trend_with_matern_kernel,
                                        linear_trend_with_rational_quadratic_kernel,
                                        linear_trend_with_square_exponential_kernel,
                                        linear_trend_with_periodicity_kernel,
                                        mixture_of_linear_kernels,
                                        mixture_squared_exponential_covariance)

from covariance.params import Params

from functools import partial
import pdb

import numpy as np
from scipy.sparse import csc_matrix, eye, lil_matrix
from scipy.sparse.linalg import cg, inv

def compute_spai(A, tol=1e-2, max_iter=None):
    """
    Compute Sparse Approximate Inverse (SPAI) of a sparse matrix A.

    Parameters:
    - A: Sparse matrix in CSC format.
    - tol: Tolerance for the Conjugate Gradient solver.
    - max_iter: Maximum number of iterations for the CG solver.

    Returns:
    - M: Sparse Approximate Inverse of A in CSC format.
    """
    n = A.shape[0]
    M = lil_matrix((n, n), dtype=np.float64)  # Create M as a lil_matrix
    I = eye(n, format='csc')
    
    for i in range(n):
        b = I[:, i].toarray().ravel()  # Convert to dense array and flatten
        x, info = cg(A, b, tol=tol, maxiter=max_iter)
        if info == 0:
            M[:, i] = x.reshape(-1, 1)  # Efficiently modify columns in lil_matrix
        else:
            print(f"Warning: CG did not converge for column {i}")
    
    return M.tocsc() 

class CovarianceMatrix:

    def __init__(self,x):
        self._matrix = x
        # print(f"Size matrix: {self._matrix.shape}")
        pass

    def __matmul__(self,x):
        
        return  self._matrix @ x
    
    def __add__(self,x):
        return self._matrix + x
    
    def _check_invertable_(self):
        assert (self._matrix.T == self._matrix).all(), 'Matrix is not symmetric'
        assert (jnp.linalg.eigvals(self._matrix) > 0).all(), 'Matrix is not positive definite'

    def _fast_inverse(self):
        # self._check_invertable_()
        return cg_inverse(self._matrix)
    
    def _inverse(self):
        # self._check_invertable_()
        return jnp.linalg.inv(self._matrix)

 
class Block2x2CholeskyMatrix(CovarianceMatrix):
    """
    Construct 2x2 diagonal covariance matrix
    Used for the inducing points
    """

    def __init__(self, matrix1,matrix2):
        """
        ensure both matrix1 and matrix2 are cholesky decompistion (lower) of PSD matrix
        
        """
        self._matrix1 = matrix1
        self._matrix2 = matrix2 
        pass

    def _construct_matrix(self, matrix1, matrix2):
        n_1 = matrix1.shape[-2]
        m_1 = matrix1.shape[-1]
        n_2 = matrix2.shape[-2]
        m_2 = matrix2.shape[-1]
        zero_1 = jnp.zeros((n_1,m_2))
        zero_2 = jnp.zeros((n_2,m_1))
        return jnp.block([[matrix1,zero_1],[zero_2,matrix2]])
    
    
class NystroemApproximation:

    def __init__(self, n_components=100,covar_func=''):
        self.n_components = n_components

        pass 

def _check_input_dim(x):
    assert x.ndim ==2 , f"Input dimensions must be 2, not {x.ndim}"
    return x


def create_params(kernel_name:str):
    if kernel_name =='squared_exponential':
        return  Params(
    kernel_name=kernel_name,
    hyperparams={
        "length_scale": 1.0,
                }
            )
    elif kernel_name=='mixture_squared_exponential':
        return  Params(
    kernel_name=kernel_name,
    hyperparams={
        "length_scales": jnp.array([1.0, 1.0]),
        "mixture_weights":jnp.array([0.5,0.5])
                }
            )
    elif kernel_name =='squared_exponential_spatial':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1.}
        )
    elif kernel_name =='rational_quadratic':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1., 'alpha':0.5}
        )
    elif kernel_name =='matern':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1.}
        )
    elif kernel_name == 'periodic':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1., 'period':1.}
        )
    elif kernel_name == 'linear':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.}
        )
    elif kernel_name == 'polynomial':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'degree':2.,
                                    'coef0':1.}
        )
    elif kernel_name =='matern_spatial':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1.}
        )
    elif kernel_name == 'locally_periodic':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale_se':1.,
                                    'length_scale_pk':1.,
                                      'period':1.}
        )
    elif kernel_name =='sm':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'weights':1., 
                                    'scales':1., 
                                    'means':1.}
        )
    elif kernel_name =='ard':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'length_scale':1.}
        )
    elif kernel_name =='vonMisesFisher':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'sigma':1.}
        )
    elif kernel_name == 'linear_trend_multiply_matern':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1.}
        )
    elif kernel_name == 'linear_trend_multiply_periodicity':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale_se':1.,'length_scale_pk':1., 'period':1.}
        )
    elif kernel_name == 'linear_trend_multiply_rational_quadratic':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1., 'alpha':0.5}
        )
    elif kernel_name == 'linear_trend_multiply_square_exponential':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1.}
        )
    elif kernel_name == 'linear_trend_with_matern':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1.}
        )
    elif kernel_name == 'linear_trend_with_rational_quadratic':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1., 'alpha':0.5}
        )
    elif kernel_name == 'linear_trend_with_square_exponential':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1.}
        )
    elif kernel_name =='linear_trend_with_periodicity':
        return  Params(kernel_name=kernel_name,
                       hyperparams={'c':1.,'length_scale':1., 'period':1.}
        )
    elif kernel_name == 'mixture_linear':
        return  Params(kernel_name=kernel_name,
                       hyperparams={
                                    'weights':1., 
                                    'biases':1.}
        )        
    else:
        raise NotImplementedError



def create_covariance( name,x, y, *params):
    x = _check_input_dim(x)
    y = _check_input_dim(y)
    
    # x = jnp.repeat(x[:,None,:],repeats=dim_y,axis=1)
    # y = jnp.repeat(y[None,:,:],repeats=dim_x,axis=0)
    if name =='squared_exponential':
        covar_func =  vmap(lambda x,y: squared_exponential_covariance(x,y,*params), (None,0),0)
    elif name=='squared_exponential_spatial':
        covar_func = vmap(lambda x,y: squared_exponential_spatial_covariance(x,y,*params), (None,0),0)
    elif name =='rational_quadratic':
        covar_func = vmap(lambda x,y:rational_quadratic_kernel(x,y,*params), (None,0),0)
    elif name=='matern':
        covar_func = vmap(lambda x,y: matern_kernel(x,y,*params), (None, 0),0)
    elif name == 'periodic':
        covar_func = vmap(lambda x,y: periodic_kernel(x, y, *params),  (None, 0),0)
    elif name=='linear':
        covar_func = vmap(lambda x,y: linear_kernel(x, y, *params),  (None, 0),0)
    elif name=='polynomial':
        covar_func = vmap(lambda x,y: polynomial_kernel(x, y, *params), (None, 0),0)
    elif name=='matern_spatial':
        covar_func = vmap(lambda x,y: matern_spatial_kernel(x,y,*params),  (None, 0),0)
    elif name == 'locally_periodic':
        covar_func = vmap(lambda x,y: locally_periodic_kernel(x, y, *params),  (None, 0),0)
    elif name=='sm':
        covar_func = vmap(lambda x,y: sm_kernel_single(x,y, *params), (None, 0),0)
    elif name =='ard':
        covar_func = vmap(lambda x,y : ARD_kernel(x, y, *params), (None, 0),0)
    elif name=='vonMisesFisher':
        covar_func = lambda x,y : vonMisesFisherkernel(x, y, *params)
    
    elif name =='linear_trend_multiply_matern':
        covar_func = lambda x,y : linear_trend_multiply_matern_kernel(x, y, *params)
    elif name=='linear_trend_multiply_periodicity':
        covar_func = lambda x,y : linear_trend_multiply_periodicity_kernel(x, y, *params)
    elif name=='linear_trend_multiply_rational_quadratic':
        covar_func = lambda x,y : linear_trend_multiply_rational_quadratic_kernel(x, y, *params)
    elif name=='linear_trend_multiply_square_exponential':
        covar_func = lambda x,y : linear_trend_multiply_square_exponential_kernel(x, y, *params)
    elif name=='linear_trend_with_matern':
        covar_func = lambda x,y : linear_trend_with_matern_kernel(x, y, *params)
    elif name=='linear_trend_with_rational_quadratic':
        covar_func = lambda x,y : linear_trend_with_rational_quadratic_kernel(x, y, *params)
    elif name=='linear_trend_with_square_exponential':
        covar_func = lambda x,y : linear_trend_with_square_exponential_kernel(x, y, *params)
    elif name=='linear_trend_with_periodicity':
        covar_func = lambda x,y : linear_trend_with_periodicity_kernel(x, y, *params)
    
    
    else:
        raise NotImplementedError
    
    if name!='vonMisesFisher':
        map_2d_distance = vmap(covar_func, (0,None), 0)
    else:
        map_2d_distance = covar_func
    covar =  map_2d_distance(x,y)
    return CovarianceMatrix(covar)


def create_covariance_v2(name, x, y, params: Params):
    """
    Create a covariance matrix using a specified kernel and parameters.

    Args:
        name (str): Name of the kernel function (e.g., 'squared_exponential', 'matern').
        x (array): Input data points (N x D).
        y (array): Input data points (M x D).
        params (Params): A Params object containing kernel-specific hyperparameters.

    Returns:
        CovarianceMatrix: The covariance matrix computed using the specified kernel.
    """
    x = _check_input_dim(x)
    y = _check_input_dim(y)

    # Define the kernel function dynamically based on the name
    if name == 'squared_exponential':
        covar_func = vmap(lambda x, y: squared_exponential_covariance(x, y, **params.hyperparams), (None, 0), 0)
    elif name=='mixture_squared_exponential':
        covar_func = vmap(lambda x, y: mixture_squared_exponential_covariance(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'squared_exponential_spatial':
        covar_func = vmap(lambda x, y: squared_exponential_spatial_covariance(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'rational_quadratic':
        covar_func = vmap(lambda x, y: rational_quadratic_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'matern':
        covar_func = vmap(lambda x, y: matern_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'periodic':
        covar_func = vmap(lambda x, y: periodic_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear':
        covar_func = vmap(lambda x, y: linear_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'polynomial':
        covar_func = vmap(lambda x, y: polynomial_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'matern_spatial':
        covar_func = vmap(lambda x, y: matern_spatial_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'locally_periodic':
        covar_func = vmap(lambda x, y: locally_periodic_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'sm':
        covar_func = vmap(lambda x, y: sm_kernel_single(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'ard':
        covar_func = vmap(lambda x, y: ARD_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'vonMisesFisher':
        covar_func = lambda x, y: vonMisesFisherkernel(x, y, **params.hyperparams)
    elif name == 'linear_trend_multiply_matern':
        covar_func = vmap(lambda x, y: linear_trend_multiply_matern_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_multiply_periodicity':
        covar_func = vmap(lambda x, y: linear_trend_multiply_periodicity_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_multiply_rational_quadratic':
        covar_func = vmap(lambda x, y: linear_trend_multiply_rational_quadratic_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_multiply_square_exponential':
        covar_func = vmap(lambda x, y: linear_trend_multiply_square_exponential_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_with_matern':
        covar_func = vmap(lambda x, y: linear_trend_with_matern_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_with_rational_quadratic':
        covar_func = vmap(lambda x, y: linear_trend_with_rational_quadratic_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_with_square_exponential':
        covar_func = vmap(lambda x, y: linear_trend_with_square_exponential_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name == 'linear_trend_with_periodicity':
        covar_func = vmap(lambda x, y: linear_trend_with_periodicity_kernel(x, y, **params.hyperparams), (None, 0), 0)
    elif name=='mixture_linear':
        covar_func = vmap(lambda x, y: mixture_of_linear_kernels(x, y, **params.hyperparams), (None, 0), 0)
    else:
        raise NotImplementedError(f"Kernel {name} is not implemented.")

    # Handle vectorized computation
    if name != 'vonMisesFisher':
        map_2d_distance = vmap(covar_func, (0, None), 0)
    else:
        map_2d_distance = covar_func

    # Compute the covariance matrix
    covar = map_2d_distance(x, y)
    return CovarianceMatrix(covar)


