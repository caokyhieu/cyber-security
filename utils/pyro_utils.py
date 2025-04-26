from  sklearn.mixture import GaussianMixture
import jax.numpy as jnp
from jax import random
import numpy as np 
import matplotlib.pyplot as plt 
import skgstat as skg
import gstools as gs
from jax import pmap,random, vmap,jit, lax
import pdb
import torch 
import jax
from tqdm import tqdm,trange
import gc
from typing import List, Tuple, Union
from jax.interpreters.ad import JVPTracer
from jax.interpreters.batching import BatchTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer
from sklearn.preprocessing import StandardScaler
# from jaxlib.xla_extension import DeviceArray
from numpy import int64, ndarray

def torch_from_numpy(arr,device:str='cpu'):

    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [torch.from_numpy(i).to(device) for i in arr]

def estimated_covariogram(spatial,values):

    list_models = ['spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic']
    bins = ['uniform','even','kmeans','ward','sturges','scott','fd','sqrt','doane']
    ## centralize values
    val = values - values.mean()
    V = skg.Variogram(spatial, val,model='gaussian', maxlag='median', normalize=False,use_nugget=True,fit_method='trf')
    
    scores = {}
    for i, model in enumerate (list_models):
        for b in bins:
            try:
                V.model = model
                V.bin_func = b
                ## check var
                describe = V.describe()
                if float(describe["sill"] - describe["nugget"])>0:
                    scores[model + '_' + b] = V.rmse
            except:
                continue
        
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=False)
    ranking = list(ranking)
    fit_model = None
    for r in ranking:
        try:
            _m,_b = r[0].split('_')
            V.model = _m
            V.bin_func = _b
            fit_model = V.to_gstools()
            print(f'Best fit model : {fit_model}')
            break
        except:
            continue
    if fit_model is not None:
        return fit_model
    else:
        raise ValueError('Could not find fitted model')
    

def is_nearly_diagonal_relative(matrix, threshold=0.01):
    matrix = np.array(matrix)
    diag_abs = np.abs(np.diag(matrix))
    off_diag = np.abs(matrix - np.diagflat(np.diag(matrix)))

    # Avoid division by zero
    norm_matrix = off_diag / (diag_abs[:, None] + 1e-10)

    return np.all(norm_matrix < threshold)

def gaussian_mixture(key,weight,means,covariances,n_samples=1000,path='',name=''):
    gm = GaussianMixture(n_components=2)
    gm.weights_ = weight
    gm.means_ = means
    gm.covariances_ = covariances

    sample,_class = gm.sample(n_samples=n_samples)
    index = jnp.arange(0,len(sample+1))
    # key = random.PRNGKey(123)
    index = random.permutation(key,index,independent=True)
    sample = sample[index]
    _class = _class[index]
    
    s1 = jnp.array([s[0] for s in sample])
    s2 = jnp.array([s[1] for s in sample])
    ## check NAN and inf
    assert (not jnp.isnan(s1).any()) and ( not jnp.isinf(s1).any()), 'S1 contain NaN or inf'
    assert (not jnp.isnan(s2).any()) and ( not jnp.isinf(s2).any()), 'S2 contain NaN or inf'
    

   
    return s1,s2,_class

def distance_function(X,Y,power=1.,min_val=None,max_val=None):
    """
    Calculates the distance matrix between two vectors
    """
    # dist = jnp.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1) ## non differntiable??
    # dist_matrix = X[:, None, :] - Y[None, :, :]
    dist = jnp.sum((X[:, None, :] - Y[None, :, :])**2,axis=-1)

    # dist = jnp.power(dist, power - 1/2) ## should remove this line to get RBF kernel
    # dist = dist**(power - 1/2)
    # print(f"Max : {dist.max()}, min : {dist.min()}")
    if (min_val is not None) or (max_val is not None):
        dist = jnp.clip(dist,a_min=min_val, a_max=max_val)
    return dist


def covariance_matrix_helper(sigma_sq,alpha,distance_matrix):
    """
    localtion: arr 
    """
    # dist = DistanceMetric.get_metric('euclidean')
    # eps = 1e-4
    ## should we add small pertubation??
    cov= sigma_sq * jnp.exp(-alpha * distance_matrix) 
    # cov -= (1 - 0.3**2) * np.eye(len(X))
    return cov + 1e-6 * jnp.eye(cov.shape[0]) # 

def covariance_matrix(sigma_sq,alpha,X,Y,power=1.):
    """
    localtion: arr 
    """
    # dist = DistanceMetric.get_metric('euclidean')
    # eps = 1e-4
    dist = distance_function(X,Y,power=power)
    ## should we add small pertubation??
    cov= sigma_sq * jnp.exp(-alpha * dist) 
    # cov -= (1 - 0.3**2) * np.eye(len(X))
    return cov + 1e-6 * jnp.eye(cov.shape[0]) ## safe calculate the covariance



# def rff_features(X, D=256, lengthscale=1.0, seed=0):
#     """
#     X: (n, d) input
#     D: number of random features
#     returns: (n, D) matrix Φ such that K ≈ ΦΦᵀ
#     This one use for RBF kernel
#     """

#     key = jax.random.PRNGKey(seed)
#     n, d = X.shape

#     w_key, b_key = jax.random.split(key)
#     W = jax.random.normal(w_key, (D, d)) / lengthscale
    
#     b = jax.random.uniform(b_key, (D,), minval=0.0, maxval=2 * jnp.pi)

#     projection = jnp.dot(X, W.T) + b
#     Phi = jnp.sqrt(2.0 / D) * jnp.cos(projection)
#     return Phi  
def rff_features(X, D=256, lengthscale=1.0, seed=0):
    """
    Random Fourier Features for RBF kernel approximation.
    
    X: (n, d) input array
    D: number of features
    lengthscale: RBF kernel lengthscale
    Returns: (n, D) feature matrix Φ
    """
    # Input validation
    lengthscale = jnp.maximum(lengthscale, 1e-6)

    key = jax.random.PRNGKey(seed)
    n, d = X.shape

    w_key, b_key = jax.random.split(key)
    W = jax.random.normal(w_key, (D, d)) / lengthscale
    b = jax.random.uniform(b_key, (D,), minval=0.0, maxval=2 * jnp.pi)

    projection = jnp.dot(X, W.T) + b
    Phi = jnp.sqrt(2.0 / D) * jnp.cos(projection)

    return Phi


def cholesky_matrix(sigma_sq,alpha,X,Y,power=1.):
    """
    localtion: arr 
    """
    # dist = DistanceMetric.get_metric('euclidean')
    # eps = 1e-4
    dist = distance_function(X,Y,power=power)
    ## should we add small pertubation??
    cov= sigma_sq * jnp.exp(-alpha * dist) 
    # cov -= (1 - 0.3**2) * np.eye(len(X))
    return jnp.linalg.cholesky(cov)

def cholesky_matrix_helper(sigma_sq,alpha,cholesky_matrix):
    """
    localtion: arr 
    """
    # dist = DistanceMetric.get_metric('euclidean')
    # eps = 1e-4
    ## should we add small pertubation??
    cov= jnp.sqrt(sigma_sq) * jnp.exp(-alpha * cholesky_matrix) 
    # cov -= (1 - 0.3**2) * np.eye(len(X))
    return cov

def kernel(alpha,X,Y):
    """
    localtion: arr 
    """
    # dist = DistanceMetric.get_metric('euclidean')
   
    cov = jnp.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    cov= jnp.exp(-alpha * cov)
    # cov -= (1 - 0.3**2) * np.eye(len(X))
    return cov

def _transform_p(p: Union[JVPTracer, DynamicJaxprTracer, BatchTracer]) -> Union[JVPTracer, DynamicJaxprTracer, BatchTracer]:
    """
    This function help to transform p
    """
    if p.ndim == 2:
        p = jnp.expand_dims(p,(0,2))
    else:
        raise ValueError('p should be  2D')
    return p

def _transform_gamma(gamma: Union[DynamicJaxprTracer, JVPTracer, BatchTracer]) -> Union[DynamicJaxprTracer, JVPTracer, BatchTracer]:
    """
    This function help to transform gamma
    """
    
    if gamma.ndim == 2:
        func = lambda x: jnp.diag(x)
        func1 = vmap(func, 0, 0)  
        gamma = func1(1/gamma**2)
    else:
        raise ValueError('gamma should be 2D')
    return jnp.expand_dims(gamma,0)

def multi_radial_basis_function(x,p: Union[DynamicJaxprTracer, JVPTracer, BatchTracer],gamma: Union[DynamicJaxprTracer, JVPTracer, BatchTracer]) -> Union[DynamicJaxprTracer, JVPTracer, BatchTracer]:
    """
    This function is to compute the multi radial basis function
    """
    assert x.ndim == 2, 'x should be 2D'
    x = jnp.expand_dims(x,(1,2))
    p = _transform_p(p)
    gamma = _transform_gamma(gamma)
    # print(x.shape)
    # print(p.shape)
    # print(gamma.shape)
    return jnp.exp(-jnp.matmul(jnp.matmul(x-p,gamma),jnp.swapaxes(x-p,-1,-2)).squeeze())
                    
# Define a type for JAX tracers (to work with JAX transformations)
JaxTracer = Union[JVPTracer, DynamicJaxprTracer, BatchTracer, jnp.ndarray]

def gaussian_kernel_superposition(X: JaxTracer, centers: JaxTracer, variances: JaxTracer) -> JaxTracer:
    """
    Computes the unweighted superposition of multiple Gaussian kernels with diagonal covariance matrices.

    Args:
        X: Input data of shape (N, D), where N is the number of data points, D is the feature dimension.
        centers: Centers of the Gaussian kernels, shape (Q, D).
        variances: Diagonal elements of the covariance matrices, shape (Q, D).

    Returns:
        JaxTracer: Gaussian kernel values of shape (N, Q), where each column represents a kernel component.
    """
    N, D = X.shape
    Q = centers.shape[0]  # Number of Gaussian components

    # Compute squared Mahalanobis distance: (x - μ)^T Σ⁻¹ (x - μ)
    X_expanded = X[:, None, :]  # Shape (N, 1, D)
    centers_expanded = centers[None, :, :]  # Shape (1, Q, D)
    diff = X_expanded - centers_expanded  # Shape (N, Q, D)

    # Since Σ is diagonal, Σ⁻¹ is just 1/variance
    inv_variances = 1 / variances  # Shape (Q, D)

    # Compute the quadratic form: sum over dimensions
    quadratic_form = jnp.sum((diff ** 2) * inv_variances, axis=-1)  # Shape (N, Q)

    # Compute Gaussian kernel values (without weights)
    gaussians = jnp.exp(-0.5 * quadratic_form)  # Shape (N, Q)

    return gaussians  # Each column corresponds to a kernel component

def radial_basic_function(x,p,gamma):
    """
    TODOS: fix this function for vector inputs
    This function for scalar inputs
    """
    
    return jnp.exp(-(x.reshape(-1,1)-p.reshape(1,-1))**2/gamma.reshape(1,-1)**2)
                    # + jnp.random.normal(0,1,size=x.shape)

## need to modify
import jax.scipy as jsp

def _multivariate_SI_predict(rng_key, X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,dist_XX):
    # compute kernels between train and test data, etc.
    k_pp = covariance_matrix_helper(sigma_sq, alpha,dist_pp)
    k_pX = covariance_matrix_helper(sigma_sq, alpha, dist_pX)
    k_XX = covariance_matrix_helper(sigma_sq, alpha, dist_XX)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    # sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
    #     rng_key, X_test.shape[:1]
    # )
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X,p,gamma), theta)
    
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y.reshape(-1) - mean_y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

def helper_func_invert(mat,b):
    return jsp.linalg.solve_triangular(mat, b, lower=True)

def _multivariate_SI_predict_L(rng_key, X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,L_dist_XX):
    # compute kernels between train and test data, etc.
    k_pp = covariance_matrix_helper(sigma_sq, alpha,dist_pp)
    k_pX = covariance_matrix_helper(sigma_sq, alpha, dist_pX)
    K_XX = covariance_matrix_helper(sigma_sq, alpha,L_dist_XX)
    L_XX = jnp.linalg.cholesky(K_XX )
    ## need to calculate  L_xx_inv @ k_pX^T = k_L_pX
    k_L_pX = jsp.linalg.solve_triangular(L_XX,k_pX.T,lower=True)
    
    K = k_pp - k_L_pX.T @ k_L_pX
    
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X,p,gamma), theta)
    L_y_corrections =  jsp.linalg.solve_triangular(L_XX, Y.reshape(-1) - mean_y, lower=True)
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + jnp.matmul(k_L_pX.T, L_y_corrections)
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

## need to modify
def _SI_predict_L(rng_key, X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,L_dist_XX):
    # compute kernels between train and test data, etc.
    k_pp = covariance_matrix_helper(sigma_sq, alpha,dist_pp)
    k_pX = covariance_matrix_helper(sigma_sq, alpha, dist_pX)
    K_XX = covariance_matrix_helper(sigma_sq, alpha,L_dist_XX)
    L_XX = jnp.linalg.cholesky(K_XX)
    # k_L_pX = vmap(lambda x,y: helper_func_invert(y,x), in_axes=(1,None), )(L_XX,k_pX.T)
    # k_L_pX = k_L_pX.T
    k_L_pX = jsp.linalg.solve_triangular(L_XX,k_pX.T,lower=True)
    K = k_pp - k_L_pX.T @ k_L_pX

    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X,p,gamma), theta)
    L_y_corrections =  jsp.linalg.solve_triangular(L_XX, Y.reshape(-1) - mean_y, lower=True)
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) +jnp.matmul(k_L_pX.T, L_y_corrections)
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

def _SI_predict(rng_key, X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,dist_XX):
    # compute kernels between train and test data, etc.
    k_pp = covariance_matrix_helper(sigma_sq, alpha,dist_pp)
    k_pX = covariance_matrix_helper(sigma_sq, alpha, dist_pX)
    k_XX = covariance_matrix_helper(sigma_sq, alpha, dist_XX)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    # sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
    #     rng_key, X_test.shape[:1]
    # )
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X,p,gamma), theta)
    
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y.reshape(-1) - mean_y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

from utils.reweight_func import _get_weights_from_cost_matrix
def get_cov_factor_cov_diag(cov_factor, diag_factor, cost_matrix):
    ## reweight cov_factor and diag_factor
    distance =  _get_weights_from_cost_matrix(cost_matrix,name='')
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cost_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    
    cov_factor_reweight = jnp.sqrt(var).reshape(-1,1) * cov_factor 
    diag_factor_reweight = var * diag_factor
    return cov_factor_reweight, diag_factor_reweight

def fast_prediction(rng_key, X, Y , X_test, alpha, p, theta, gamma, sigma_sq,SI_train,SI_test):
    ''' Using RFF to calculate prediction'''
    Phi_train = rff_features(SI_train,D=32, lengthscale=jnp.sqrt(1/(2 * alpha)))
    Phi_test = rff_features(SI_test,D=32, lengthscale=jnp.sqrt(1/(2 * alpha)))
    Phi_train, Phi_test = jnp.sqrt(sigma_sq) * Phi_train, jnp.sqrt(sigma_sq) * Phi_test

    # Compute inverse using Woodbury identity
    sigma = 1e-3
    # (D x D) matrix
    A = Phi_train.T @ Phi_train  # (D, D)
    B = jnp.linalg.inv(sigma**2 * jnp.eye(A.shape[0]) +  A)  # (D, D)
    CC_T = Phi_test @ Phi_test.T  # (n_*, n_*)
    FC = Phi_test @ Phi_train.T    # (n_*, n)
    K_star_K_inv = (1 / sigma**2) * FC - (1 / sigma**2) * (FC @ Phi_train @ B @ Phi_train.T)  # (n_*, n)
    # Compute covariance
    cov = CC_T - K_star_K_inv @ FC.T 
    cov +=  sigma * jnp.eye(cov.shape[0]) ## make  cov stable

    # Posterior mean
    mean_y = jnp.matmul(gaussian_kernel_superposition(X,p,gamma), theta)
    mean = K_star_K_inv @ (Y.reshape(-1) - mean_y)
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + mean

    # Posterior variance 
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), cov)
    return mean_pred, mean_pred + sigma_noise

def _multivariate_SI_predict_from_sub_matrixes(rng_key, k_pp, k_pX, K_xx_inv, X_train, Y_train , X_test, p, theta, gamma):
    # compute kernels between train and test data, etc.
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    # sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
    #     rng_key, X_test.shape[:1]
    # )
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X_train,p,gamma), theta)
    
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y_train.reshape(-1) - mean_y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

def _SI_predict_from_sub_matrixes(rng_key, k_pp, k_pX, K_xx_inv, X_train, Y_train , X_test, p, theta, gamma):
    # compute kernels between train and test data, etc.
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    # sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
    #     rng_key, X_test.shape[:1]
    # )
    sigma_noise = random.multivariate_normal(rng_key, jnp.zeros( X_test.shape[:1]), K)
    mean_y = jnp.matmul(gaussian_kernel_superposition(X_train,p,gamma), theta)
    
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta) + jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y_train.reshape(-1) - mean_y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    # pdb.set_trace()
    return mean_pred, mean_pred + sigma_noise

def _regression_predict(rng_key, X_test, p, theta, gamma, sigma_sq):
    # compute kernels between train and test data, etc.
    sigma_noise =sigma_sq * random.normal(
        rng_key, X_test.shape[:1]
    )
    mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,jnp.array(p).reshape(-1,1),jnp.array(gamma).reshape(-1,1)), theta)
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    # pdb.set_trace()
    return mean_pred, mean_pred + sigma_noise

def _multivariate_regression_predict(rng_key, X_test, p: BatchTracer, theta: BatchTracer, gamma: BatchTracer, sigma_sq: BatchTracer) -> Tuple[BatchTracer, BatchTracer]:
    # compute kernels between train and test data, etc.
    sigma_noise =sigma_sq * random.normal(
        rng_key, X_test.shape[:1]
    )
    if theta.shape[-1]==1:
        mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma).reshape(-1,1), theta)
    else:
        mean_pred = jnp.matmul(gaussian_kernel_superposition(X_test,p,gamma), theta)
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean_pred, mean_pred + sigma_noise

## Data with spatial information
from scipy.stats import truncnorm ## use truncated normal to have the same domain
def get_data(random_seeds,m1,m2,sig1,sig2,n_l,n_u,p,gamma,theta,alpha, sigma_sq,*args,**kargs):
    key = random.PRNGKey(random_seeds)
    # folder_name = f"exp_{random_seeds}"
    ## check dimension
    assert len(m1) == len(m2), "m1 and m2 should have the same dimension"
    assert len(sig1) == len(sig2), "s1 and s2 should have the same dimension"
    assert len(m1) == len(sig1), "m1 and s1 should have the same dimension"
    dim = len(m1)
    # assert dim <=2, "dimension should be less than or equal 2"
    # num_train_samples = n_l * dim
    # num_test_samples = n_u * dim

    ### init data 
    # x_l = m1.reshape(1,-1) + random.generalized_normal(key,p=np.sqrt(sig1),shape=(num_train_samples,))
    # x_l = m1.reshape(1,-1) + jnp.vstack([random.generalized_normal(key,p=np.sqrt(variance),shape=(n_l,)) for variance in sig1]).T
    x_l = m1.reshape(1,-1) + np.sqrt(sig1).reshape(1,-1) * jax.random.normal(key,shape=(n_l,len(m1.flatten())))

    ## now take min and max from train data
    x_min, x_max = jnp.min(x_l, axis=0),jnp.max(x_l, axis=0)
    ## now loop through dimensions to got truncated normal values
    x_u = np.zeros((n_u, dim))
    for i in range(dim):
        x_u[:,i] = truncnorm.rvs(a=(x_min[i] - m2[i])/np.sqrt(sig2[i]), b=(x_max[i]-m2[i])/np.sqrt(sig2[i]), loc=m2[i], scale=np.sqrt(sig2[i]), size=n_u)



    # jnp.vstack([random.generalized_normal(key,p=np.sqrt(variance),shape=(n_l,)) for variance in sig1]).T
    # x_l = x_l.reshape(-1,dim)
    # x_u = m2 + random.generalized_normal(key,p=np.sqrt(sig2),shape=(num_test_samples,))
    # x_u = m2.reshape(1,-1) + jnp.vstack([random.generalized_normal(key,p=np.sqrt(variance),shape=(n_u,)) for variance in sig2]).T
    # x_u = m2.reshape(1,-1) + np.sqrt(sig2).reshape(1,-1) * jax.random.normal(key,shape=(n_u,len(m2.flatten())))
    # x_u = x_u.reshape(-1,dim)
    X = jnp.concatenate((x_l,x_u),axis=0)
    ## spatial information
    ## TODOs: fix hard code
    weight = jnp.array([1/2,1/2])
    means = jnp.array([[-1.,-1.],[4.,4.]])
    cov_space = jnp.array([[[4,1.],[1.,4.]],[[1,0.],[0.,1]]])
    s1,s2,_class = gaussian_mixture(key,weight,means,cov_space,n_samples=n_l+n_u,path='',name='mixture')
    ## normalzie the spatial information, using sklearn algorithm
    

    location = jnp.array(list(zip(s1[:n_l+n_u],s2[:n_l+n_u])))

    # ## standardize the location
    # scaler = StandardScaler()
    # location = scaler.fit_transform(location)

    ## covariance matrix
    covariance = covariance_matrix(sigma_sq,alpha,location,location)
    ## check covariance matrix

    print(f"any nan: {jnp.isnan(covariance).any()}")
    eivals = jnp.linalg.eigvals(covariance)
    print(f'Max values: {jnp.max(covariance)}')
    print(f'Min values: {jnp.min(covariance)}')
    assert jnp.all(eivals>0), 'Covariance matrix must be positive definite'
    ## check covariance matrix is not diagonal
    assert not is_nearly_diagonal_relative(covariance), 'Covariance matrix should not be diagonal'
    # print(f'Positive definite: {jnp.all(eivals>0)}')
    ### multi_variate
    if dim == 1:
        mean_ = jnp.matmul(gaussian_kernel_superposition(X,jnp.array(p).reshape(-1,1),jnp.array(gamma).reshape(-1,1)) ,theta).flatten()
    else:
        mean_ = jnp.matmul(gaussian_kernel_superposition(X,p,gamma) ,theta).flatten()
   
    y = random.multivariate_normal(key,mean_,covariance)
    y_u = y[len(x_l):]
    y_l = y[:len(x_l)]

    ## to torch
    locations_l = location[:n_l]
    locations_u = location[n_l:]
    
    return (x_l,locations_l,y_l), (x_u,locations_u,y_u)


def is_finite(matrix):
    """Check if all elements in the matrix are finite."""
    return jnp.all(jnp.isfinite(matrix))

def cholesky_with_jitter(matrix, max_iters=10, base_jitter=1e-8):
    """
    Attempt Cholesky decomposition, checking for NaNs in the result as an indication of failure.
    Adds jitter to the diagonal and retries if necessary.

    Args:
    - matrix (jax.numpy.ndarray): Square, symmetric matrix.
    - max_iters (int): Maximum iterations to attempt adding jitter.
    - base_jitter (float): Base jitter value.

    Returns:
    - jax.numpy.ndarray: Cholesky factor of the matrix, with added jitter if necessary.
    """
    
    def body_fun(i, mat):
        # Attempt Cholesky decomposition, dont forward gradient in this step
        stop_gradient_cholesky = lambda x: jnp.linalg.cholesky(jax.lax.stop_gradient(x))
        L = stop_gradient_cholesky(mat)
        # L = stable_cholesky(mat)
        
        # Check if the result is finite
        success = is_finite(L) 
        
        # If decomposition was successful (no NaNs), return the matrix; otherwise, add jitter
        return lax.cond(success,
                        lambda _: mat,  # If success, just return the matrix
                        lambda _: mat.at[jnp.diag_indices_from(mat)].add(base_jitter * (10 ** i)),  # Add jitter
                        operand=None)
    
    # Initialize loop variables
    initial_mat = matrix
    
    # Run the loop with static iteration bounds, checking for success each iteration
    final_mat = lax.fori_loop(0, max_iters, body_fun, initial_mat)

    # Perform the Cholesky decomposition on the final matrix
    L = jnp.linalg.cholesky(final_mat)
    return L

## Data without spatial information
def get_data_without_SI(random_seeds,m1,m2,sig1,sig2,n_l,n_u,p,gamma,theta,sigma_sq):
    key = random.PRNGKey(random_seeds)
    # folder_name = f"exp_{random_seeds}"
    
    ### init data 
    x_l = m1 + random.generalized_normal(key,p=np.sqrt(sig1),shape=(n_l,))
    x_u = m2 + random.generalized_normal(key,p=np.sqrt(sig2),shape=(n_u,))
    X = jnp.concatenate((x_l,x_u),axis=0)

    ### normal distributions
    mean_ = jnp.matmul(gaussian_kernel_superposition(X,p,gamma) ,theta).flatten()
    y = mean_ + sigma_sq * random.normal(key,shape=mean_.shape)
    y_u = y[len(x_l):]
    y_l = y[:len(x_l)]
    
    return (x_l,y_l), (x_u,y_u)

def plot_histmap(spatial,z_vals,error_vals):
    
    x_vals, x_idx = np.unique(spatial[:,0], return_inverse=True)
    y_vals, y_idx = np.unique(spatial[:,1], return_inverse=True)
    vals_array = np.empty(x_vals.shape + y_vals.shape)
    
    vals_array.fill(-10) # or whatever your desired missing data flag is
    vals_array[x_idx, y_idx] = z_vals
    
    x, y = np.meshgrid(x_vals, y_vals)
    z = vals_array[:-1,:-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(15,5))
    
    c = ax1.pcolormesh(x, y, z, cmap='hot', vmin=z_min, vmax=z_max)
    ax1.set_title('z_values')
    # set the limits of the plot to the limits of the data
    ax1.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax1)
    
    
    vals_array = np.empty(x_vals.shape + y_vals.shape)
    
    vals_array.fill(-10) # or whatever your desired missing data flag is
    vals_array[x_idx, y_idx] = error_vals
    
    z = vals_array[:-1,:-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    
    c = ax2.pcolormesh(x, y, z, cmap='hot', vmin=z_min, vmax=z_max)
    ax2.set_title('error_values')
    # set the limits of the plot to the limits of the data
    ax2.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax2)
    plt.show()

from itertools import product
from jax import grad

class MatrixPoints:
    """
    This class for 2-dimentional spatial data
    """

    def __init__(self,spatial):
        """
        spatial: spatial data (2-3 coordinate)
        data: data for each points
        length of data == len of spatial
        spatial,data: numpy type
        """
        self.key = random.PRNGKey(123)
        
        self.n_points = len(spatial)
        # x_unit = np.min(np.diff(np.sort(spatial[:,0])))
        # y_unit = np.min(np.diff(np.sort(spatial[:,1])))
        min_x = jnp.min(spatial[:,0])
        max_x = jnp.max(spatial[:,0])
        min_y = jnp.min(spatial[:,1])
        max_y = jnp.max(spatial[:,1])
        self.x_axis = jnp.linspace(min_x,max_x,num=self.n_points)
        self.y_axis =  jnp.linspace(min_y,max_y,num=self.n_points)
        # self.data_dim = data.shape[1]
        # self.data = jnp.zeros((self.n_points,self.n_points,self.data_dim))
        self.x_indexes =  jnp.searchsorted( self.x_axis,spatial[:,0] , side='left')
        self.y_indexes = jnp.searchsorted( self.y_axis,spatial[:,1] , side='left')
        self.dict_mask = self.create_mask()

    # def conv(self,kernel):
    #     pos = np.abs(self.data).sum(-1)>0. ### pos
    #     data = signal.convolve2d(pos, kernel, boundary='wrap', mode='same') 
    #     return np.einsum('ij,ijk->ijk',data,self.data)

    
    def make_Q(self,*args):
        theta =  self.create_theta(*args)
        Q = jnp.zeros((self.n_points,self.n_points))
        indexes = list(zip(range(self.n_points),range(self.n_points)))

        mat_indexes = jnp.array(list(product(indexes,indexes))).reshape(self.n_points,self.n_points,2,2)
        mat_indexes = mat_indexes[:,:,1,:] - mat_indexes[:,:,0,:] 
        print(f"Matrix index: {mat_indexes.shape}")
        for i,j in list(product(range(self.n_points),range(self.n_points))):
            # pdb.set_trace()
            if ((mat_indexes[i,j,0]>=-2) and   (mat_indexes[i,j,0]<=2)) and((mat_indexes[i,j,1]>=-2) and   (mat_indexes[i,j,1]<=2)):
                Q = Q.at[i,j].set(theta[mat_indexes[i,j,0]+2,mat_indexes[i,j,1]+2]) 

        return Q 

    def create_theta(self,*args):
        assert len(args)==5, print('just implemented for kernel 5 * 5')
        # ### create theta
        theta = jnp.zeros((5,5))
        theta = theta.at[2,2].set( 1)
        theta = theta.at[[1,2,2,3],[2,1,3,2]].set( args[0])
        theta = theta.at[[1,1,3,3],[1,3,1,3]].set(  args[1])
        theta = theta.at[[0,2,2,4],[2,0,4,2]].set(  args[2])
        theta = theta.at[[0,0,1,1,3,3,4,4],[1,3,0,4,0,4,1,3]].set( args[3])
        theta = theta.at[[0,0,4,4],[0,4,0,4]].set( args[4])
        return theta 
    
    def create_mask(self):
        theta =  self.create_theta(2,3,4,5,6)
        # Q = jnp.zeros((self.n_points,self.n_points))
        dict_mask = {i:jnp.zeros((self.n_points,self.n_points)) for i in range(1,7)}
        
        indexes = list(zip(range(self.n_points),range(self.n_points)))

        mat_indexes = jnp.array(list(product(indexes,indexes))).reshape(self.n_points,self.n_points,2,2)
        mat_indexes = mat_indexes[:,:,1,:] - mat_indexes[:,:,0,:] 
        print(f"Matrix index: {mat_indexes.shape}")
        for i,j in list(product(range(self.n_points),range(self.n_points))):
            if ((mat_indexes[i,j,0]>=-2) and   (mat_indexes[i,j,0]<=2)) and((mat_indexes[i,j,1]>=-2) and   (mat_indexes[i,j,1]<=2)):
                index_mask = theta.at[mat_indexes[i,j,0]+2,mat_indexes[i,j,1]+2].get()
                index_mask = int(index_mask)
                dict_mask[index_mask] = dict_mask[index_mask].at[i,j].set(1) 
                # Q = Q.at[i,j].set(theta[mat_indexes[i,j,0]+2,mat_indexes[i,j,1]+2]) 
        
        return dict_mask
    
    def loss(self,target,args1,args2,args3,args4,args5,args6):
        temp_Q = args1 * self.dict_mask[1] + args2 * self.dict_mask[2] + args3 * self.dict_mask[3] + args4 * self.dict_mask[4] + args5 * self.dict_mask[5] + args6 * self.dict_mask[6]
        l = jnp.sum(jnp.power(target-temp_Q,2))
        return l
    
    def optimize(self,target,n_iter=100,lam=1e-3,args1=1., args2=2.,args3=3.,args4=4.,args5=5.,args6=6.):
        loss_func = lambda args1,args2,args3,args4,args5,args6: self.loss(jnp.linalg.inv(target),args1,args2,args3,args4,args5,args6)
        for i in range(n_iter):
            g = grad(loss_func,(0,1,2,3,4,5))(args1,args2,args3,args4,args5,args6)
            # update
            args1 -= lam* g[0]
            args2 -= lam* g[1]
            args3 -= lam* g[2]
            args4 -= lam* g[3]
            args5 -= lam* g[4]
            args6 -= lam* g[5]
        return (args1,args2,args3,args4,args5,args6)

def estimate_variogram_covariance_2(spatial,stack_spatial,values):

    list_models = ['spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic']
    bins = ['uniform','even','kmeans','ward','sturges','scott','fd','sqrt','doane']
    val = values - values.mean()
    V = skg.Variogram(spatial, val,model='gaussian', maxlag='median', normalize=False,use_nugget=True,fit_method='trf')
    
    scores = {}
    for i, model in enumerate (list_models):
        for b in bins:
            try:
                V.model = model
                V.bin_func = b
                scores[model + '_' + b] = V.rmse
            except:
                continue
        
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=False)
    
    _m,_b = list(ranking)[0][0].split('_')
    V.model = _m
    V.bin_func = _b
    fit_model = V.to_gstools()
    print(f'Best fit model : {fit_model}')

    ## test the true covariance matrix
    # true covarianceclear
    distance_matrix = jnp.linalg.norm(stack_spatial[:, None, :] - stack_spatial[None, :, :], axis=-1)
    ## estimated covariance
    cv = fit_model.covariance(r=distance_matrix)
    cr = fit_model.correlation(distance_matrix)

    return cr,cv

def estimate_GMRF_precision(stack_spatial, covariance,n_iter=100):
    Q = MatrixPoints(stack_spatial)
    estiamted_params = Q.optimize(covariance,n_iter=10000)
    estimated_Q =  estiamted_params[0] * Q.dict_mask[1] + estiamted_params[1] * Q.dict_mask[2] + estiamted_params[2] * Q.dict_mask[3] +\
                    estiamted_params[3] * Q.dict_mask[4] + estiamted_params[4] * Q.dict_mask[5] + estiamted_params[5] * Q.dict_mask[6]

    return estimated_Q

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from metric_learn import NCA


def read_real_data(photo_path: str,
                   spectral_path: str,
                   cols: List[str] = ['u_mod','g_mod','r_mod','i_mod','z_cmod'],
                   n_samples: int=300,
                   random_state: int64=42,
                   normalize: bool=False
                    ) -> Tuple[Tuple[ndarray, None, ndarray], Tuple[ndarray, None, ndarray]]:
    """
    Helper function to read real photometric and spectral data
    Args:
        photo_path (str): path to photometric data (txt)
        spectral_path (str): path to spectral data (txt)
        cols (list): list of photometric bands to keep
    Returns:
        photometric (np.ndarray): photometric data
        spectral (np.ndarray): spectral data
    """
    
    photo_df = pd.read_table(photo_path,skiprows=2,sep=' ')
    spectral_df = pd.read_table(spectral_path,sep=' ')
    ## take the extreme values of spectral data as the range of features
    extreme_dict = {col:(spectral_df[col].min(),spectral_df[col].max()) for col in cols}
    ## filter the photometric data by the extreme values
    length_photo = len(photo_df)
    for col in cols:
        photo_df = photo_df[(photo_df[col]>=extreme_dict[col][0]) & (photo_df[col]<=extreme_dict[col][1])]
        print(f"Photometric data remaining: {len(photo_df)/length_photo * 100:.2f} percent. Filtered by {col}")
    
    nca = NCA(random_state=random_state)
    X1 = spectral_df.sample(n=n_samples,random_state=random_state)
    X2 = photo_df.sample(n=n_samples,random_state=random_state)
    X = np.concatenate((X1[cols].values,X2[cols].values),axis=0)
    y = np.concatenate((np.ones((len(X1),))
                    , np.zeros((len(X2),))),axis=0)
    nca.fit(X, y)
    knn = KNeighborsClassifier(metric=nca.get_metric())
    knn.fit(X, y)
    print("KNN model predicting")
    # result = []
    # stepsize = 500
    # for i in trange(0,len(spectral_df),stepsize):
    #     result.append(knn.predict(spectral_df[cols].values[i:i+stepsize]))
    # y_pred = np.concatenate(result,axis=0).flatten()
    y_pred = knn.predict(spectral_df[cols].values)
    

    pseudo_photo = spectral_df[y_pred==0][cols].to_numpy(dtype=np.float64)
    photo_label = spectral_df[y_pred==0]['redshift'].to_numpy(dtype=np.float64)
    pseudo_spectro = spectral_df[y_pred==1][cols].to_numpy(dtype=np.float64)

    spectro_label = spectral_df[y_pred==1]['redshift'].to_numpy(dtype=np.float64)
    print("Done")

    # # test code
    # pseudo_spectro = np.random.normal(size=(1000,5))
    # spectro_label = np.random.normal(size=(1000,))
    # pseudo_photo = np.random.normal(size=(1000,5))
    # photo_label = np.random.normal(size=(1000,))
    # n_choice_photo = int(len(pseudo_photo)*0.5)
    # n_choice_spectro = int(len(pseudo_spectro)*0.5)
    # n_choice_photo = 1000
    # n_choice_spectro = 1000

    n_choice_photo = int(len(pseudo_photo)*0.5)
    n_choice_spectro = int(len(pseudo_spectro)*0.5)
    choice_photo_index = np.random.permutation(len(pseudo_photo))[:n_choice_photo]
    choice_spectro_index = np.random.permutation(len(pseudo_spectro))[:n_choice_spectro]

    if normalize:
        covariate_mean = pseudo_spectro.mean()
        covariate_std = pseudo_spectro.std()
        label_mean = spectro_label.mean()
        label_std = spectro_label.std()
        pseudo_spectro = (pseudo_spectro - covariate_mean)/covariate_std
        spectro_label = (spectro_label - label_mean)/label_std
        pseudo_photo = (pseudo_photo - covariate_mean)/covariate_std
        photo_label = (photo_label - label_mean)/label_std

        return (pseudo_spectro[choice_spectro_index],None,spectro_label[choice_spectro_index]),(pseudo_photo[choice_photo_index],None,photo_label[choice_photo_index] ),(covariate_mean,covariate_std,label_mean,label_std)
    else:
        return (pseudo_spectro[choice_spectro_index],None,spectro_label[choice_spectro_index]),(pseudo_photo[choice_photo_index],None,photo_label[choice_photo_index])
    
    #     return (pseudo_spectro,None,spectro_label),(pseudo_photo,None,photo_label ),(covariate_mean,covariate_std)
    # else:
    #     return (pseudo_spectro,None,spectro_label),(pseudo_photo,None,photo_label)


def get_gradients(svi, svi_state, rng_key, *args, **kwargs):
    params = svi.get_params(svi_state)
    
    def loss_fn(params):
        return svi.loss.loss(rng_key, params, svi.model, svi.guide, *args, **kwargs)

    
    loss, grads = jax.value_and_grad(loss_fn)(params)

    # # Check for NaNs in gradients
    # if any(jnp.isnan(jnp.array(list(grads.values()))).any() for grad in grads.values()):
    #     raise ValueError("NaN detected in gradients")

    return grads, loss


def check_nan_or_inf(arr):

    return jnp.isnan(arr).any() or jnp.isinf(arr).any()


import optax
from numpyro.optim import _NumPyroOptim

def create_custom_optimizer(step_size,clip_norm=100.0):
    # Define the optimizer from optax, e.g., using Adam optimization algorithm.
    # optimizer = optax.adam(step_size)
    optimizer = optax.adam(step_size)

    def init_fun(params):
        opt_state = optimizer.init(params)
        return params, opt_state

    def update_fun(step, grads, state):
        params, opt_state = state
        # Preprocess the gradients using the safe_grad 
        safe_grads = jax.tree_map(lambda g: jnp.clip(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),a_min=-clip_norm, a_max=clip_norm), grads)
        updates, opt_state = optimizer.update(safe_grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return updated_params, opt_state
       
    def get_params_fun(state):
        params, _ = state
        return params

    return _NumPyroOptim(lambda x, y, z: (x, y, z), init_fun, update_fun, get_params_fun)

def scale_optimizer(step_size,scale_factors ):
    # Define the optimizer from optax, e.g., using Adam optimization algorithm.
    # optimizer = optax.adam(step_size)
    optimizer = optax.adam(step_size)
    scale_factors = scale_factors

    def init_fun(params):
        opt_state = optimizer.init(params)
        return params, opt_state

    def update_fun(step, grads, state):
        params, opt_state = state
        # Preprocess the gradients using the safe_grad 
        # safe_grads = jax.tree_map(lambda g: g * scale_factors.get(k, 1.0), grads)
        grads = {k: v * scale_factors.get(k, 1.0) for k, v in grads.items()}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return updated_params, opt_state
       
    def get_params_fun(state):
        params, _ = state
        return params

    return _NumPyroOptim(lambda x, y, z: (x, y, z), init_fun, update_fun, get_params_fun)



def intervent_optimizer(step_size,stopping_params,freeze_params ):
    """
    Optimizer for intervent events with two dictionaries:
        stopping_params: a dict of params and the step that we will stop optimizing them.
        freeze_params: a dict of params and the step that we will optimize them.
    """
    # Define the optimizer from optax, e.g., using Adam optimization algorithm.
    
    optimizer = optax.adam(step_size)
    stopping_params = stopping_params
    freeze_params = freeze_params
    assert len(set(stopping_params.keys()).intersection(set(freeze_params.keys())))==0, "stopping_params and freeze_params must be exlusively"

    def init_fun(params):
        opt_state = optimizer.init(params)
        return params, opt_state
    
    def update_fun(step, grads, state):
        params, opt_state = state
        def true_fun(v):
            return jnp.zeros_like(v)
        def false_fun(v):
            return v
        ## first we filter the gradients for stopping parameters
        if len(stopping_params) > 0:
            grads = {k: jax.lax.cond(stopping_params.get(k, np.inf) < step, true_fun, false_fun, v) for k, v in grads.items()}
        ## filter the gradients for freeze parameters
        if len(freeze_params) >0:
            grads = {k: jax.lax.cond(freeze_params.get(k,-1) > step, true_fun, false_fun, v) for k, v in grads.items()}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        return updated_params, opt_state
       
    def get_params_fun(state):
        params, _ = state
        return params

    return _NumPyroOptim(lambda x, y, z: (x, y, z), init_fun, update_fun, get_params_fun)



@jit
def jit_calculate_distance(x,y):
    x = x.reshape(1,-1)
    y = x- y 
    return jnp.sum(y**2,axis=1)**(1/2)

@jit
def calculate_distance_matrix(x,y):
    n_cols = len(y)
    n_rows = len(x)
    result = jnp.zeros((n_rows,n_cols))
    def body_fn(i, val):
        val = val.at[i,:].set(jit_calculate_distance(x[i],y))
        return val

    return lax.fori_loop(0,n_rows , body_fn, result)

import jax.scipy as jsc

# from utils.decorator_func import time_decorator
# @time_decorator
@jit
def cg_inverse(A, tol=1e-5, maxiter=1000):
    size = len(A)
    dentity = jnp.eye(size)
    result = jnp.zeros((size,size))
    ## preconditioner
    M = jnp.linalg.inv(A)
    def body_fn(i, val):
        val = val.at[:,i].set(jsc.sparse.linalg.cg(A,dentity[:,i],M=M, tol=tol, maxiter=maxiter)[0])
        return val
    return lax.fori_loop(0, size , body_fn, result)        

def truncated_svd_inverse_psd(A, k=None):
    U, sigma, _ = jnp.linalg.svd(A, full_matrices=False)

    if k is None:
        k = jnp.sum(sigma > jnp.finfo(sigma.dtype).eps * max(A.shape))

    sigma_inv = jnp.zeros_like(sigma)
    sigma_inv = sigma_inv.at[:k].set(1. / sigma[:k])

    A_pseudo_inv = jnp.dot(U, jnp.dot(jnp.diag(sigma_inv), U.T))
    return A_pseudo_inv

def linear_kernel(X, X_prime=None, sigma_b=1.0, sigma_v=1.0, c=0.0):
    """
    Computes the linear kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    sigma_b : float, optional
        Variance of the bias term.
    sigma_v : float, optional
        Variance of the linear term.
    c : float, optional
        Offset term.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed linear kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    
    X = X - c
    X_prime = X_prime - c
    
    K = sigma_b**2 + sigma_v**2 * jnp.dot(X, X_prime.T)
    return K

def periodic_kernel(X, X_prime=None, period=1.0, length_scale=1.0):
    """
    Computes the periodic kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    period : float, optional
        The period of the function.
    length_scale : float, optional
        The length scale of the kernel.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed periodic kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    
    dists = jnp.abs(X[:, None] - X_prime[None, :])
    arg = jnp.pi * dists / period
    sin_arg = jnp.sin(arg)
    
    K = jnp.exp(-2.0 * (sin_arg ** 2) / (length_scale ** 2))
    return K
def polynomial_kernel(X, X_prime=None, alpha=1.0, c=1.0, degree=3):
    """
    Computes the polynomial kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    alpha : float, optional
        The influence of higher-order versus lower-order terms.
    c : float, optional
        The constant term.
    degree : int, optional
        The polynomial degree.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed polynomial kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    
    K = jnp.dot(X, X_prime.T)
    K = alpha * K + c
    K = K ** degree
    return K

def rational_quadratic_kernel(X, X_prime=None, alpha=1.0, length_scale=1.0):
    """
    Computes the rational quadratic kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    alpha : float, optional
        The scale-mixture parameter.
    length_scale : float, optional
        The length scale of the kernel.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed rational quadratic kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    
    dists_squared = jnp.sum((X[:, None, :] - X_prime[None, :, :]) ** 2, axis=-1)
    K = (1 + dists_squared / (2 * alpha * length_scale ** 2)) ** (-alpha)
    return K

from scipy.special import kv, gamma

def matern_kernel(X, X_prime=None, nu=1.5, length_scale=1.0):
    """
    Computes the Matérn kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    nu : float, optional
        The smoothness parameter.
    length_scale : float, optional
        The length scale of the kernel.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed Matérn kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    
    dists = jnp.sqrt(jnp.sum((X[:, None, :] - X_prime[None, :, :]) ** 2, axis=-1))
    dists = jnp.sqrt(2 * nu) * dists / length_scale
    
    K = (2 ** (1 - nu) / gamma(nu)) * (dists ** nu) * kv(nu, dists)
    K = jnp.where(dists == 0.0, 1.0, K)  # Handling the case when dists is 0
    return K

@jit
def haversine(x1, x2):
    # Convert celestial coordinates from degrees to radians
    ra1, dec1 = x1
    ra2, dec2 = x2
    # Convert from degrees to radians
    # Haversine formula
    d_ra = ra2 - ra1 
    d_dec = dec2 - dec1 
    a = jnp.sin(d_dec/2.0)**2 + jnp.cos(dec1) * jnp.cos(dec2) * jnp.sin(d_ra/2.0)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a)) 

    # Convert distance from radians to degrees
    # distance_degrees = jnp.degrees(c)

    # return distance_degrees
    return c**2

@jit
def celestial_distance(x1, x2):
    ra1, dec1 = x1
    ra2, dec2 = x2
    # Convert from degrees to radians
    ra1 = jnp.radians(ra1)
    dec1 = jnp.radians(dec1)
    ra2 = jnp.radians(ra2)
    dec2 = jnp.radians(dec2)

    # Calculate the argument for acos
    arg = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(dec2) * jnp.cos(ra1 - ra2)

    # Angular separation formula
    angle = jnp.arccos(arg)
    # angle = arccos_sigmoid_approx(arg)

    # Convert back to degrees
    angle = jnp.degrees(angle)

    return angle

def arccos_sigmoid_approx(x):
    # Scale and shift a sigmoid function to approximate arccos
    # Parameters for scaling and shifting would need to be adjusted for better accuracy
    return (jnp.tanh(-x) + 1) * (jnp.pi / 2)

# def arccos_approx(x):
#     # Example polynomial approximation of arccos(x)
#     # This is a simple example; for better accuracy, you might need a higher degree polynomial
#     return jnp.pi / 2 - x - x**3 / 6
# def arccos_sigmoid_approx(x, scale=2., shift=0.):
#     # Sigmoid-based approximation of arccos
#     # 'scale' and 'shift' adjust the shape of the sigmoid
#     sigmoid = 1 / (1 + jnp.exp(-scale * (x - shift)))
#     return (sigmoid - 0.5) * jnp.pi

def rbf_kernel(X, X_prime=None, length_scale=1.):
    """
    Computes the RBF kernel between two sets of points.
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    X_prime : array-like, shape (n_samples_X_prime, n_features), optional
        Second set of points. If None, the values are assumed to be equal to X.
    length_scale : float, optional
        The length scale of the kernel.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed RBF kernel matrix.
    """
    if X_prime is None:
        X_prime = X
    ## use euclid distance
    # dists_squared = jnp.sum((X[:, None, :] - X_prime[None, :, :]) ** 2, axis=-1)
    ## use celestial_distance
    map_celestial_distance = vmap(haversine, (0,0),0)
    map_2d_distance = vmap(map_celestial_distance, (0,0), 0)
    
    ## expand dim for X and ~X_prime 
    X = jnp.repeat(X[:, None, :],repeats=X_prime.shape[0],axis=1)
    X_prime = jnp.repeat( X_prime[None, :, :],repeats=X.shape[0],axis=0)
    dists_squared = map_2d_distance(jnp.radians(X),jnp.radians(X_prime))
    # print("Mean",jnp.mean(dists_squared))
    # print("Max",jnp.max(dists_squared))

    K = jnp.exp(-0.5 * dists_squared / length_scale**2)
    return K

from low_rank.balltree_covariance import BallTreeCovarianceMatrix

def haversine_v2(x1, x2):
    # Convert celestial coordinates from degrees to radians
    # Convert from degrees to radians
    ra1,dec1 = x1[:,0],x1[:,-1]
    ra2,dec2 = x2[:,0],x2[:,-1]
 
    # Haversine formula
    d_ra = ra2 - ra1 
    d_dec = dec2 - dec1 
    a = jnp.sin(d_dec/2.0)**2 + jnp.cos(dec1) * jnp.cos(dec2) * jnp.sin(d_ra/2.0)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a)) 

    # Convert distance from radians to degrees
    return c

    
from functools import lru_cache, wraps
from collections import OrderedDict
def checksum_key(array):
    return np.sum(array)

def hashable_key(array):
    return (array.shape, array.dtype)

class HashableArray2D:
    def __init__(self, array):
        self.array = array
    
    def __hash__(self):
        return hash(self.array.tobytes())
    
    def __eq__(self, other):
        return jnp.array_equal(self.array, other.array)

def jnp_cache(function):
    global_dict =  OrderedDict()
    @wraps(function)
    def wrapper(key,array):
        if key not in global_dict:
            global_dict[key] = array
        return cached_wrapper(key)

    @lru_cache(maxsize=64)
    def cached_wrapper(key):
        cache_info = cached_wrapper.cache_info()
        current_size = cache_info.currsize
        maxsize = cache_info.maxsize
        if current_size > maxsize:
            global_dict.popitem(last=False)

        array = global_dict[key]
        return function(array)

   

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
@jnp_cache
def calculate_sparse_matrix(X):
    # dist_func = map_2d_distance
    # init ball tree
    ball = BallTreeCovarianceMatrix(jnp.radians(X),metric='haversine',
                                    radius=0.5,
                                    num_iter=100, ## maximum iter for algorithm
                                    dtype=jnp.float64,
                                    )
    ## use celestial_distance
    dists_squared = ball.covariance()
    return dists_squared



# ## inverse sparse matrix
# @jit
# def cg_inverse(A):
#     size = len(A)
#     dentity = jnp.eye(size)
#     result = jnp.zeros((size,size))
#     def body_fn(i, val):
#         val = val.at[:,i].set(jsc.sparse.linalg.cg(A,dentity[:,i])[0])
#         return val
#     return lax.fori_loop(0, size , body_fn, result)     
    
def rbf_kernel_with_sparse_matrix(X,X_prime,  length_scale=1.):
    """
    Computes the RBF kernel between two sets of points and return sparse matrix
    
    Args:
    X : array-like, shape (n_samples_X, n_features)
        First set of points.
    
    length_scale : float, optional
        The length scale of the kernel.
        
    Returns:
    K : array, shape (n_samples_X, n_samples_X_prime)
        The computed RBF kernel matrix.
    """
    # if X_prime is None:
    #     X_prime = X
    ## function
    map_celestial_distance = vmap(haversine, (0,0),0)
    map_2d_distance = vmap(map_celestial_distance, (0,0), 0)
    # dist_func = map_2d_distance
    # init ball tree
    # dists_squared = calculate_sparse_matrix(hashable_key(X),X).todense()
    X = jnp.repeat(X[:, None, :],repeats=X_prime.shape[0],axis=1)
    X_prime = jnp.repeat( X_prime[None, :, :],repeats=X.shape[0],axis=0)
    X_prime_2 = jnp.repeat( X_prime[0][ :,None, :],repeats=X_prime.shape[1],axis=1)
    X_prime_3 = jnp.repeat( X_prime[0][ None,:, :],repeats=X_prime.shape[1],axis=0)
    # X_prime_1 = jnp.repeat(X[:, 0:1, :],repeats=X.shape[0],axis=1)
    dists_squared_2 = map_2d_distance(jnp.radians(X),jnp.radians(X_prime))
    dists_3 = map_2d_distance(jnp.radians(X_prime_2),jnp.radians(X_prime_3))
    # dists_squared_2 = jnp.where(dists_squared_2>0.2,jnp.inf,dists_squared_2)
    # dists_3 = jnp.where(dists_3>0.2,jnp.inf,dists_3)
    # dists_1 = map_2d_distance(jnp.radians(X_prime_1),jnp.radians(X_prime_1))
    # print("Mean",jnp.mean(dists_squared))
    # print("Max",jnp.max(dists_squared))
    ## be careful with BCOO object
    # K1 = dists_squared
    # K1.data =  jnp.exp(-0.5 * K1.data/ length_scale **2)
    # pdb.set_trace()

    # K1 = jnp.exp(-0.5 * dists_squared * (1 / length_scale**2))
    ## filter the distance matrix, where >0.5 will set 0
    # dists_squared_2 = dists_squared_2.at[dists_squared_2>0.2].set(0)
    # dists_3 = dists_3.at[dists_3>0.2].set(0)
    # K1 = jnp.exp(-0.5 * dists_1 / length_scale**2)
    K2 = jnp.exp(-0.5 * dists_squared_2 / length_scale**2)
    K3 = jnp.exp(-0.5 * dists_3 / length_scale**2) 
    return None, K2, K3

# ## inverse sparse matrix
# @jit
# def cg_inverse(A):
#     size = len(A)
#     dentity = jnp.eye(size)
#     result = jnp.zeros((size,size))
#     def body_fn(i, val):
#         val = val.at[:,i].set(jsc.sparse.linalg.cg(A,dentity[:,i])[0])
#         return val
#     return lax.fori_loop(0, size , body_fn, result)     
def lanczos_algorithm(A, m):
    n = A.shape[0]
    Q = jnp.zeros((n, m + 1))
    alpha = jnp.zeros(m)
    beta = jnp.zeros(m + 1)

    q = jax.random.normal(jax.random.PRNGKey(0), (n, ))
    q = q / jnp.linalg.norm(q)
    Q = Q.at[:, 0].set(q)

    for k in range(m-1):
        v = jnp.dot(A, Q[:, k]) - beta[k] * Q[:, k - 1]
        alpha = alpha.at[k].set(jnp.dot(Q[:, k], v))

        v = v - alpha[k] * Q[:, k]
        beta = beta.at[k + 1].set(jnp.linalg.norm(v))
        # if beta[k + 1] < 1e-10:  # Adjust the tolerance as needed
        #     break
        Q = Q.at[:, k + 1].set(v / beta[k + 1])

    T = jnp.diag(alpha) + jnp.diag(beta[1:m], k=1) + jnp.diag(beta[1:m], k=-1)
    return jnp.linalg.eigvalsh(T)



@jit
def rbf_kernel_(x1, x2, length_scale, variance):
    sqdist = jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    return variance * jnp.exp(-0.5 / length_scale**2 * sqdist)

@jit
def periodic_kernel(x1, x2, length_scale, variance, period):
    sqdist = jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    return variance * jnp.exp(-2 * jnp.sin(jnp.pi * jnp.sqrt(sqdist) / period)**2 / length_scale**2)

@jit
def spectral_mixture_kernel(x1, x2, weights, means, variances):
    kernel = 0
    sqdist = jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    for w, m, v in zip(weights, means, variances):
        
        kernel += w * jnp.exp(-2 * jnp.pi**2 * sqdist * v) * jnp.cos(2 * jnp.pi * jnp.sqrt(sqdist) * m)
    return kernel


from itertools import product
def gen_mesh(min_vals,max_vals,num=10,movement=0.5):
    """
    Generate uniformly constrianted mesh
    """
    assert (len(min_vals)==len(max_vals)), "min and max arr have same dim"
    n_dim = len(min_vals)
    dim_num = int(np.power(num,1/n_dim))
    ## non overlapped constraints
    constraints = [np.linspace(min_val,max_val,dim_num+ 1) for min_val,max_val in zip(min_vals,max_vals)]
    ## calculate delta gap
    delta = [a[-1] - a[-2] for a in constraints]
    lower_constraints = list(product(*[constraint[:-1] for constraint in constraints]))
    upper_constraints =list(product(*[constraint[1:] for constraint in constraints]))
    ## make overlapped constraints
    lower_constraints = [[lower_constraint[i] - movement * delta[i] for i in range(len(delta))] for lower_constraint in lower_constraints]
    upper_constraints = [[upper_constraint[i] + movement * delta[i] for i in range(len(delta))] for upper_constraint in upper_constraints]

    points = [ np.random.uniform(low=lower_constraint,high= upper_constraint) for lower_constraint,\
                          upper_constraint in zip(lower_constraints,upper_constraints)]
    
    return points, lower_constraints, upper_constraints


