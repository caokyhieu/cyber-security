import numpy as np
from scipy.linalg import toeplitz,circulant
from numpy.fft import fft, ifft
import time
def toeplitz_matrix_vector_mult(toeplitz_col, toeplitz_row, v):
    """
    Multiply Toeplitz matrix defined by toeplitz_col by vector v using FFT.
    """
    # Number of rows in Toeplitz matrix
    n = len(toeplitz_col)

    # Create a circulant matrix column from toeplitz_col
    circulant_col = np.concatenate([toeplitz_col,toeplitz_row[:1], toeplitz_row[-1:0:-1]])
    print("circulant matrix:",circulant_col )
    # print(circulant(circulant_col))

    # Use FFT to perform circulant matrix-vector multiplication
    result = ifft(fft(circulant_col) * fft(np.concatenate([v, np.zeros(n)])))

    return result[:n]

# Example
col = np.array([1, 2, 3, 4] + [i for i in range(100)])  # First column of the Toeplitz matrix
row = np.array([1, -5, -2, -1] + [i for i in range(200,300)])  # First row of the Toeplitz matrix
T = toeplitz(col, row)
v = np.array([1, 0, -1, 1] + [i for i in range(100)])

print("Toeplitz Matrix:")
print(T)
print("Vector v:", v)
start = time.time()
result = np.dot(T, v)

print("Direct multiplication result:",result, 'Time:',time.time() - start)
start = time.time()
result = toeplitz_matrix_vector_mult(col, row, v)
print("Efficient multiplication result:",result, 'Time:',time.time() - start )
