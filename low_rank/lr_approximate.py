import numpy as np
import scipy.linalg

def randomized_low_rank_approximation(A, k, l=None):
    """
    Compute a low-rank approximation of the positive semi-definite matrix A.

    Parameters:
    A (numpy.ndarray): The input PSD matrix of size (m, n).
    k (int): The target rank for the approximation.
    l (int, optional): The oversampling parameter. Should be >= k. If None, set to k + 5.

    Returns:
    U (numpy.ndarray): An (m, k) matrix with orthonormal columns.
    S (numpy.ndarray): A (k, k) diagonal matrix with non-negative real numbers on the diagonal.
    Vh (numpy.ndarray): A (k, n) matrix with orthonormal rows.
    """
    if l is None:
        l = k + 5
    
    # Step 1: Generate a random Gaussian matrix
    Omega = np.random.randn(A.shape[1], l)

    # Step 2: Compute the matrix Y = A * Omega
    Y = A @ Omega

    # Step 3: Perform QR decomposition on Y
    Q, _ = np.linalg.qr(Y)

    # Step 4: Compute B = Q^T * A
    B = Q.T @ A

    # Step 5: Perform SVD on B
    U_b, S, Vh = scipy.linalg.svd(B, full_matrices=False)

    # Step 6: Compute U = Q * U_b
    U = Q @ U_b

    # Keep only the top k components
    U = U[:, :k]
    S = np.diag(S[:k])
    Vh = Vh[:k, :]

    return U, S, Vh

# Example usage:
if __name__ == "__main__":
    # Creating a PSD matrix A
    # A = np.array([[4, 2], [2, 3]])
    A = np.random.normal(size=(400,5))
    A = A @ A.T
    
    # Setting the target rank k
    k = 1
    
    # Running the low-rank approximation algorithm
    U, S, Vh = randomized_low_rank_approximation(A, k=2)
    
    # Printing the results
    print("U:\n", U)
    print("S:\n", S)
    print("Vh:\n", Vh)
    
    # Reconstructing the low-rank approximation
    A_k = U @ S @ Vh
    print("Low-Rank Approximation:\n", A_k)
