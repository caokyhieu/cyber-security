import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jax import jit
import numpy as np
import time
from jax.scipy.linalg import cho_factor, cho_solve

def inverse_psd_matrix_cholesky(A):
    # Compute the Cholesky decomposition of A
    c, lower = cho_factor(A)
    # Use cho_solve to solve Ax = I for x, which gives the inverse of A
    A_inv = cho_solve((c, lower), jnp.eye(A.shape[0]))
    return A_inv

# def solve_without_inversion(A, b):
#     x, _ = cg(A, b)
#     return x

def compute_spai_jax(A, n_iter=10):
    """
    Compute a Sparse Approximate Inverse (SPAI) for matrix A using JAX.
    This simplified version works with dense arrays and iteratively improves
    the approximation for the inverse.

    Parameters:
    - A: Square matrix (jax.numpy.ndarray).
    - n_iter: Number of iterations to refine the approximation.

    Returns:
    - M: Approximation of the inverse of A.
    """
    n = A.shape[0]
    M = jnp.eye(n)  # Start with the identity matrix as the initial approximation

    @jit
    def update_M(M):
        """Perform one iteration of SPAI update."""
        for i in range(n):
            # Define the vector for the current column of the identity matrix
            b = jnp.eye(n)[:, i]
            # Solve the linear system A * x = b
            x, _ = cg(A, b, maxiter=1000)  # Using a single iteration of CG for demonstration
            # Update the column in M
            M = M.at[:, i].set(x)
        return M

    # Iteratively update M to improve the approximation
    for _ in range(n_iter):
        M = update_M(M)

    return M

# Example usage with a small matrix
n = 10  # Using a small matrix for demonstration purposes
A = jnp.array(np.random.rand(n, n))  # Example dense matrix
A = A @ A.T  # Making sure A is symmetric and positive definite
## start time
start = time.time()

# Compute the Sparse Approximate Inverse
# M_approx = compute_spai_jax(A, n_iter=1000)
M_approx = inverse_psd_matrix_cholesky(A)
print(f"Time to inverse with cholesky: {time.time() - start} seconds")

## time to inverse by cg
# start = time.time()
# M_approx = compute_spai_jax(A, n_iter=1000)
# Test the approximation (result should be close to the identity matrix)
# I_approx = A @ M_approx
# print(I_approx)
# print(jnp.allclose(I_approx, jnp.eye(n), atol=1e-4))  # Using a tolerance for numerical stability

## directly inverse
start = time.time()
A_inv = jnp.linalg.inv(A)
print(f"Time to inverse directly: {time.time() - start} seconds")