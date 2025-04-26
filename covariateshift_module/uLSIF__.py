"""
\alpha- relative divergence : q_{\alpha}(x) = \alphap(x) + (1- \alpha)p'(x)
combined with Pearson divergence : PE[p(x),p'(x)] = 1/2 E_{p'(x)}[(r(x) - 1)^2]
r(x) = p(x)/p'(x)
"""
import jax.numpy as jnp
from jax import random, vmap
import jax
import numpy as np
import pdb

key = random.PRNGKey(seed=42)
def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse

grad_fn = jax.grad(loss_fn)

def update(params, grads):
    return jax.tree_map(lambda p, g: p - 0.0001* g, params, grads)

# def full_update(i, data):
#     params,data = data 
#     grads = grad_fn(params,*data)
#     params = update(params, grads)
#     return params,data

X = random.normal(key=key,shape=(300,2))
true_params = {'w': jnp.array([[4],[1]]),'b':5}
label = jnp.dot(X, true_params['w']**2)  + jnp.dot(X, true_params['w']) + true_params['b'] + random.normal(key=key,shape=X.shape)
params  = {
                'w': jnp.zeros((X.shape[-1],1)),
                'b': 0.
                    }

for i in range(50000):
    grads = grad_fn(params,X,label)
    params = update(params,grads)
    if i%100 ==0:
        error =loss_fn(params,X, label)
        print(f"Iter : {i}, Loss: {error:.2f}")


