import jax.numpy as jnp
from jax import random, vmap
import jax
import numpy as np
import pdb

"""
All kernel function apply for 2d vectors,
and return a matrix
"""

def rbf_kernel(x1,x2,gamma=1.):

    kernel = jnp.exp(-gamma * jnp.linalg.norm(x1[:,jnp.newaxis,:]-x2[jnp.newaxis,:,:],axis=-1)**2)
    return kernel

def H_matrix(alpha,x1,x2):
    """
    x1: shape (n1,m)
    x2 : shape (n2,m)
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    k1 =  rbf_kernel(x1,x1) ## shape (n1,n1)
    k2 = rbf_kernel(x1,x2) ## shape (n1,n2)
    # k3 = rbf_kernel(x2,x2)  ## shape (n2,n2)
    H = alpha/n1 * jnp.matmul(k1,k1.T) + (1 - alpha)/n2 * jnp.matmul(k2,k2.T) ## shape (n1,n1)
    return H 

def optimal_theta(H,h,gamma=1):
    H_shape = H.shape[0]

    return jnp.matmul(jnp.linalg.inv(H + gamma * jnp.eye(H_shape)),h)

def loss_function(params,H,h,gamma):

    """
    params : shape (n1,1)
    H : shape (n1,n1)
    h : shape (n1,1)
    gamma : scalar
    """
    l = 1/2 * jnp.matmul(jnp.matmul(params.T,H),params) \
            - jnp.matmul(h.T,params) \
            + gamma/2 *jnp.matmul(params.T,params)
    
    return l.squeeze()

def update_params(params,grads,lr=1e-2):
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def r_hat(params,kernel):

    return jnp.matmul(kernel,params)

# x1 = np.random.normal(size=(100,2))
# x2 = np.random.normal(size=(50,2))
# norm_jnp = jnp.linalg.norm(x1-x2,axis=-1)
# norm_  = jnp.sum((x1-x2)**2,axis=-1)
# print(jnp.allclose(norm_jnp**2, norm_))
grad_fn = jax.grad(loss_function)

def density_ratio_optimize(x1,x2,alpha,gamma,n_iter=100,lr=1e-2):
    kernel = rbf_kernel(x1,x2,gamma=1.)
    params = jnp.zeros((kernel.shape[0],1))
    H_hat = H_matrix(alpha,x1,x2)
    h = jnp.mean(kernel,axis=1,keepdims=True)
    for i in range(n_iter):
        grads = grad_fn(params,H_hat,h,gamma)
        # print(f"Norm grads: {jnp.linalg.norm(grads)}")
        params = update_params(params,grads,lr=1e-2)
    params = jnp.where(params>=0,params,0)
    r = r_hat(params,rbf_kernel(x1,x1,gamma=1.))

    return r
# alpha = 0.3
# gamma = 1/2
# r = density_ratio_optimize(x1,x2,alpha,gamma,n_iter=100,lr=1e-2)
# print(r)


