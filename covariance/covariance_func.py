import jax.numpy as jnp
from jax import jit
from jax import vmap
## distance

@jit
def haversine_distance(x1, x2):
    # Convert celestial coordinates from degrees to radians
    dec1, ra1  = x1
    dec2, ra2 = x2
    # Haversine formula
    d_ra = ra2 - ra1 
    d_dec = dec2 - dec1 
    a = jnp.sin(d_dec/2.0)**2 + jnp.cos(dec1) * jnp.cos(dec2) * jnp.sin(d_ra/2.0)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a)) 

    # Convert distance from radians to degrees
    # distance_degrees = jnp.degrees(c)
    return c

@jit
def spherical_law_of_cosines(x1, x2):
    # Convert celestial coordinates from degrees to radians
    dec1, ra1 = x1
    dec2, ra2 = x2
    
    # Spherical law of cosines formula
    cos_d = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(dec2) * jnp.cos(ra2 - ra1)
    distance = jnp.arccos(cos_d)

    return distance

@jit
def celestial_distance(x1, x2):
    dec1, ra1  = x1
    dec2, ra2  = x2

    # Calculate the argument for acos
    arg = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(dec2) * jnp.cos(ra1 - ra2)

    # Angular separation formula
    angle = jnp.arccos(arg)
    # angle = arccos_sigmoid_approx(arg)
    # # Convert back to degrees
    # angle = jnp.degrees(angle)

    return angle

@jit 
def square_distance(x1,x2):

    return jnp.sum((x1 - x2) ** 2)


def spherical_to_cartesian(ra, dec):
    x = jnp.cos(dec) * jnp.cos(ra)
    y = jnp.cos(dec) * jnp.sin(ra)
    z = jnp.sin(dec)
    return jnp.stack((x, y, z), axis=-1)


def vonMisesFisherkernel(X, Y, sigma):
        X_cartesian = spherical_to_cartesian(X[:, 0], X[:, 1])
        Y_cartesian = spherical_to_cartesian(Y[:, 0], Y[:, 1])
        dot_product = jnp.dot(X_cartesian, Y_cartesian.T)
        return jnp.exp(dot_product / sigma**2)

@jit
def euclidean_distance(x1,x2):
    ### need to use fractional exponetials
    return square_distance(x1,x2)**(1/2)
@jit 
def absolute_distance(x1,x2):
    return jnp.sum(jnp.abs(x1 - x2))

def get_distance_func(name='square'):

    if name == 'square':
        return square_distance
    
    elif name=='absolute':
        return absolute_distance
    elif name=='haversine':
        return haversine_distance
    elif name=='celestial':
        return celestial_distance
    elif name=='euclidean':
        return euclidean_distance
    else:
        raise NotImplementedError('Unknown')
    
@jit
def squared_exponential_covariance(x, x_prime, length_scale):
    """Compute the squared exponential covariance."""
    # sqdist = get_distance_func(name=distance)(x,x_prime)
    sqdist = square_distance(x,x_prime)
    return jnp.exp(-0.5 * sqdist / length_scale ** 2)

@jit
def mixture_squared_exponential_covariance(x, x_primes, length_scales, mixture_weights):
    """Compute the squared exponential covariance."""
    # sqdist = get_distance_func(name=distance)(x,x_prime)
    # sqdist = vmap(square_distance ,(None,0))(x,x_primes)
    r = vmap(squared_exponential_covariance, (None, None, 0))(x, x_primes, length_scales)
    return jnp.sum(r * mixture_weights)

@jit
def squared_exponential_spatial_covariance(x, x_prime, length_scale):
    """Compute the squared exponential covariance."""
    # sqdist = get_distance_func(name=distance)(x,x_prime)
    sqdist =    spherical_law_of_cosines(x,x_prime)
    sqdist = sqdist**2 

    return jnp.exp(-0.5 * sqdist / length_scale ** 2)

@jit
def rational_quadratic_kernel(x, x_prime, length_scale, alpha):
    """Rational Quadratic kernel function."""
    # sqdist = get_distance_func(name=distance)(x,x_prime)
    sqdist = square_distance(x,x_prime)
    return (1 + sqdist / (2 * alpha * length_scale ** 2)) ** (-alpha)

@jit
def periodic_kernel(x, x_prime, length_scale, period):
    """Periodic kernel function."""
    dist = euclidean_distance(x, x_prime)
    arg = jnp.pi * jnp.sum(dist) / period
    return jnp.exp(-2 * (jnp.sin(arg) ** 2) / length_scale ** 2)

@jit
def matern_kernel(x, x_prime, length_scale):
    """Matérn kernel function for nu=1.5."""
    dist = euclidean_distance(x,x_prime)
    scaled_dist = jnp.sqrt(3.0) * dist / length_scale
    return (1 +  scaled_dist ) * jnp.exp(-scaled_dist)

@jit
def matern_spatial_kernel(x, x_prime, length_scale):
    """Matérn kernel function for nu=1.5."""
    dist = haversine_distance(x,x_prime)
    scaled_dist = jnp.sqrt(3.0) * dist / length_scale
    return (1 +  scaled_dist ) * jnp.exp(- scaled_dist)


@jit
def linear_kernel(x, x_prime, c=0.):
    """Linear kernel function."""
    return jnp.dot(x, x_prime) + c
@jit
def mixture_of_linear_kernels(x, x_prime, weights, biases):
    """
    Mixture of Linear Kernels.

    Args:
        x (jnp.ndarray): Input vector of shape (d,).
        x_prime (jnp.ndarray): Input vector of shape (d,).
        weights (jnp.ndarray): Weights for the mixture, shape (M,).
        biases (jnp.ndarray): Biases for the mixture, shape (M,).

    Returns:
        jnp.ndarray: Kernel similarity score.
    """
    dot_product = jnp.dot(x, x_prime)  # Compute dot product
    linear_terms = weights * (dot_product + biases)
    return jnp.sum(linear_terms)  # Sum over all mixture components
@jit
def polynomial_kernel(x, x_prime, degree, coef0=1.):
    """Polynomial kernel function."""
    return (jnp.dot(x, x_prime) + coef0) ** degree


@jit
def locally_periodic_kernel(x, x_prime, length_scale_se,length_scale_pk, period):
    """Locally periodic kernel function."""
    return squared_exponential_covariance(x, x_prime, length_scale_se) * periodic_kernel(x, x_prime, length_scale_pk, period)

@jit
def linear_trend_with_periodicity_kernel(x, x_prime, c, length_scale, period):
    return linear_kernel(x, x_prime, c) + periodic_kernel(x, x_prime, length_scale, period)

@jit
def linear_trend_with_square_exponential_kernel(x, x_prime, c, length_scale):
    return linear_kernel(x, x_prime, c) + squared_exponential_covariance(x, x_prime, length_scale)

@jit
def linear_trend_with_rational_quadratic_kernel(x, x_prime, c, length_scale, alpha):
    return linear_kernel(x, x_prime, c) + rational_quadratic_kernel(x, x_prime, length_scale, alpha)

@jit
def linear_trend_with_matern_kernel(x, x_prime, c, length_scale):
    return linear_kernel(x, x_prime, c) + matern_kernel(x, x_prime, length_scale)

@jit 
def linear_trend_multiply_periodicity_kernel(x, x_prime, c, length_scale_se,length_scale_pk, period):
    return linear_kernel(x, x_prime, c) * locally_periodic_kernel(x, x_prime, length_scale_se,length_scale_pk, period)

@jit
def linear_trend_multiply_square_exponential_kernel(x, x_prime, c, length_scale):
    return linear_kernel(x, x_prime, c) * squared_exponential_covariance(x, x_prime, length_scale)

@jit
def linear_trend_multiply_rational_quadratic_kernel(x, x_prime, c, length_scale, alpha):
    return linear_kernel(x, x_prime, c) * rational_quadratic_kernel(x, x_prime, length_scale, alpha)

@jit
def linear_trend_multiply_matern_kernel(x, x_prime, c, length_scale):
    return linear_kernel(x, x_prime, c) * matern_kernel(x, x_prime, length_scale)
@jit
def ARD_kernel(x,y, length_scale):
    '''
    x: 1D array p
    y: 1D array p
    length scale: 1D array p
    '''

    scaled_diff = (x-y)/ length_scale 
    return jnp.exp(-0.5 * jnp.sum(scaled_diff ** 2) )
@jit
def smooth_trend_and_variation_kernel(x, x_prime, length_scale_se, length_scale_rq, alpha):
    return squared_exponential_covariance(x, x_prime, length_scale_se) + rational_quadratic_kernel(x, x_prime, length_scale_rq, alpha)


@jit
def sm_kernel_single(x, y, weights, scales, means):
    """
    Single instance of the Spectral Mixture (SM) kernel computation.

    Args:
    x: array-like, single input shape (d,)
    y: array-like, single input shape (d,)
    weights: array-like, weights of the mixture components shape (q)
    scales: array-like, scales of the mixture components (q,d)
    means: array-like, means of the mixture components shape (q,d)

    Returns:
    float, kernel value for the input pair
    """
    diff = x - y
    ## expand diff
    diff = diff[None,...] ## shape (1, d)
    # weights = weights[...,None] ## shape (q, 1)
    exp_term = jnp.exp(jnp.sum(-2 * jnp.pi**2 * diff**2 * scales**2,axis=-1))
    cos_term = jnp.cos(jnp.sum(2 * jnp.pi * diff * means,axis=-1))
    return jnp.sum(weights * exp_term * cos_term)







# from utils.pyro_utils import cg_inverse
# from jax import vmap
# def _check_input_dim(x):
#     assert x.ndim ==2 , f"Input dimensions must be 2, not {x.ndim}"
#     return x
# def create_covariance( name,x, y, *params):
#     x = _check_input_dim(x)
#     y = _check_input_dim(y)
    
#     # x = jnp.repeat(x[:,None,:],repeats=dim_y,axis=1)
#     # y = jnp.repeat(y[None,:,:],repeats=dim_x,axis=0)
#     if name =='squared_exponential_covariance':
#         covar_func =  vmap(lambda x,y: squared_exponential_covariance(x,y,*params), (None,0),0)
#     elif name=='squared_exponential_spatial_covariance':
#         covar_func = vmap(lambda x,y: squared_exponential_spatial_covariance(x,y,*params), (None,0),0)
#     elif name =='rational_quadratic_kernel':
#         covar_func = vmap(lambda x,y:rational_quadratic_kernel(x,y,*params), (None,0),0)
#     elif name=='matern_kernel':
#         covar_func = vmap(lambda x,y: matern_kernel(x,y,*params), (None, 0),0)
#     elif name == 'periodic_kernel':
#         covar_func = vmap(lambda x,y: periodic_kernel(x, y, *params),  (None, 0),0)
#     elif name=='linear_kernel':
#         covar_func = vmap(lambda x,y: linear_kernel(x, y, *params),  (None, 0),0)
#     elif name=='polynomial_kernel':
#         covar_func = vmap(lambda x,y: polynomial_kernel(x, y, *params), (None, 0),0)
#     elif name=='matern_spatial_kernel':
#         covar_func = vmap(lambda x,y: matern_spatial_kernel(x,y,*params),  (None, 0),0)
#     elif name == 'locally_periodic_kernel':
#         covar_func = vmap(lambda x,y: locally_periodic_kernel(x, y, *params),  (None, 0),0)
#     elif name=='sm_kernel':
#         covar_func = vmap(lambda x,y: sm_kernel_single(x,y, *params), (None, 0),0)
#     elif name=='vonMisesFisherkernel':
#         covar_func = vonMisesFisherkernel
#     else:
#         raise NotImplementedError
#     map_2d_distance = vmap(covar_func, (0,None), 0)
#     covar =  map_2d_distance(x,y)
#     return CovarianceMatrix(covar)


# class CovarianceMatrix:

#     def __init__(self,x):
#         self._matrix = x
#         # print(f"Size matrix: {self._matrix.shape}")
#         pass

#     def __matmul__(self,x):
#         return  self._matrix @ x
    
#     def _check_invertable_(self):
#         assert (self._matrix.T == self._matrix).all(), 'Matrix is not symmetric'
#         assert (jnp.linalg.eigvals(self._matrix) > 0).all(), 'Matrix is not positive definite'

#     def _fast_inverse(self):
#         # self._check_invertable_()
#         return cg_inverse(self._matrix)
    
#     def _inverse(self):
#         # self._check_invertable_()
#         return jnp.linalg.inv(self._matrix)

 