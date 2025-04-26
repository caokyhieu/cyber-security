from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax 
import jax.numpy as jnp
from scipy.linalg import expm, logm

def interpolate_rotation_matrices_nd(R1, R2, t):
    """
    Interpolate between two n-dimensional rotation matrices R1 and R2.

    Parameters:
    - R1: First rotation matrix (n x n).
    - R2: Second rotation matrix (n x n).
    - t: Interpolation factor (0 = R1, 1 = R2).

    Returns:
    - R_t: Interpolated rotation matrix.
    """
    # Compute the relative rotation matrix
    R_relative = np.dot(R1.T, R2)
    
    # Compute the matrix logarithm (skew-symmetric generator)
    Omega = logm(R_relative)
    
    # Interpolate in the tangent space
    Omega_t = t * Omega
    
    # Exponentiate to get the interpolated rotation
    R_t = np.dot(R1, expm(Omega_t))
    
    return R_t


class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_transforms=5, 
                 start_sd=0.1, end_sd=1.0,
                 start_mean=0.0, end_mean=1.0):
        self._stds = None
        self.n_transforms = n_transforms
        self.start_sd = start_sd
        self.end_sd = end_sd
        self.sds = np.linspace(start=start_sd,stop=end_sd, num=n_transforms+1)
        self.sds = self.sds[1:]
        self.means  = np.linspace(start=start_mean,stop=end_mean, num=n_transforms+1)
        self.means  = self.means[1:]
        
    def fit(self, X, y=None):
        """
        No fitting necessary for log transformation, so just return self.
        """
        self.mean_ = np.mean(X, axis=0)
        self.sd_ = np.std(X, axis=0)
        return self
    
    def transform(self, X, **kwargs):
        """
        Apply the log transformation.
        """
        ## get index in sequence of transformation
        to_idx = kwargs.get('to_idx',0)
        assert (to_idx < self.n_transforms) and (to_idx >=0), "to_idx must be greater or equal  0 and less than n_transforms"
        X = np.asarray(X)
        X = (X - self.mean_)/(self.sd_) * self.sds[to_idx]
        return X + self.means[to_idx]
    
    def inverse_transform(self, X, **kwargs):
        """
        Apply the inverse normal transformation, return original scale.
        """
        ## get index in sequence of transformation
        from_idx = kwargs.get('from_idx',0)
        assert (from_idx < self.n_transforms) and (from_idx >=0), "from_idx must be greater or equal  0 and less than n_transforms"
        
        X = np.asarray(X)
        X = (X - self.means[from_idx]) / self.sds[from_idx]
        X = X * self.sd_
        return X + self.mean_
    


class RegularizedPCAWhitening(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.alpha = alpha
        self.mean_ = None
        self.eigvecs_ = None
        self.eigvals_ = None

    def fit(self, X, y=None):
        # Compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        
        # PCA: Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        self.eigvals_, self.eigvecs_ = eigvals, eigvecs
        return self

    def transform(self, X, **kwargs):

        # get alpha
        alpha= kwargs.get('alpha', 0.5)
        # Center the data
        X_centered = X - self.mean_
        
        # PCA whitening
        whitening_matrix = self.eigvecs_ @ np.diag(1.0 / np.sqrt(self.eigvals_)) @ self.eigvecs_.T
        X_pca = X_centered @ whitening_matrix

        # Regularized transformation
        X_transformed = (1 - alpha) * X_centered + alpha * X_pca
        return X_transformed + self.mean_

    def inverse_transform(self, X_transformed, **kwargs):
        # get alpha
        alpha = kwargs.get('alpha', 0.5)
        ## center the data
        X_transformed -= self.mean_
        # Reconstruct original data
        whitening_matrix = self.eigvecs_ @ np.diag(1.0 / np.sqrt(self.eigvals_)) @ self.eigvecs_.T
        X_original = X_transformed @ np.linalg.inv( (1 - alpha) * np.eye(X_transformed.shape[-1]) + alpha * whitening_matrix) + self.mean_
        return X_original
    

class SequenceRegularizedPCAWhitening(BaseEstimator, TransformerMixin):
    def __init__(self, n_transforms=5, 
                 start_mean=0.0, end_mean=1.0,
                 dest_cov=0.5):
        
        ## get sequences fo means
        self.means = np.linspace(start=start_mean, stop=end_mean, num=n_transforms+1)
        self.means = self.means[1:]
        
        ## get sequences of blending
        self.dest_cov = dest_cov
        self.alphas = np.linspace(start=1., stop=0., num=n_transforms+1)
        self.alphas = self.alphas[:-1]

        ## initialize vec
        self.eigvals_ = None 
        self.eigvecs_ = None
        
        ## 
        self.dest_eigvals_, self.dest_eigvecs_ = np.linalg.eigh(dest_cov)
        

    def fit(self, X, y=None):

        # Compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        
        # PCA: Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        self.eigvals_, self.eigvecs_ = eigvals, eigvecs
        return self
    
    def transform(self, X, **kwargs):
        # get index in sequence of transformation
        to_idx = kwargs.get('to_idx',0)
        assert (to_idx < len(self.means)) and (to_idx >=0), "to_idx must be greater or equal  0 and less than n_transforms"

        # Center the data
        X_centered = X - self.mean_
        
        # PCA whitening
        whitening_matrix = self.eigvecs_ @ np.diag(1.0 / np.sqrt(self.eigvals_)) @ self.eigvecs_.T

        # Compute the blended target covariance
        blended_eigvals = (1 - self.alphas[to_idx]) * self.eigvals_ + self.alphas[to_idx] * self.dest_eigvals_
        blended_eigvecs = interpolate_rotation_matrices_nd( self.eigvecs_, self.dest_eigvecs_, self.alphas[to_idx])

        X_pca = X_centered @ whitening_matrix

        # Regularized transformation
        X_transformed = X_pca @ blended_eigvecs @ np.diag(np.sqrt(blended_eigvals)) @ blended_eigvecs.T
        return X_transformed + self.means[to_idx]
    
    def inverse_transform(self, X_transformed, **kwargs):

        # get index in sequence of transformation
        from_idx = kwargs.get('from_idx',0)
        assert (from_idx < len(self.means)) and (from_idx >=0), "from_idx must be greater or equal  0 and less than n_transforms"

        ## center the data
        X_transformed -= self.means[from_idx]

        # Compute the blended target covariance
        blended_eigvals = (1 - self.alphas[from_idx]) * self.eigvals_ + self.alphas[from_idx] * self.dest_eigvals_
        blended_eigvecs = interpolate_rotation_matrices_nd( self.eigvecs_, self.dest_eigvecs_, self.alphas[from_idx])

        X_transformed = X_transformed @ blended_eigvecs @ np.diag(1/np.sqrt(blended_eigvals)) @ blended_eigvecs.T
        # Reconstruct original data
        inverse_whitening_matrix = self.eigvecs_ @ np.diag(np.sqrt(self.eigvals_)) @ self.eigvecs_.T

        X_original = X_transformed  @ inverse_whitening_matrix + self.mean_
        return X_original
        


        
        
