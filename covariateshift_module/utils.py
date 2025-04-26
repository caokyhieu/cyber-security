from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from abc import ABCMeta, abstractmethod
import random
import jax.numpy as jnp

from jaxopt._src import tree_util
from typing import Any
from typing import Tuple
import jax
import pdb

def estimate_sigma(x_train, x_test, n_samples=10000):
    # Randomly subsample
    idx_train = np.random.choice(len(x_train), size=min(n_samples, len(x_train)), replace=False)
    idx_test = np.random.choice(len(x_test), size=min(n_samples, len(x_test)), replace=False)

    sub_x_train = x_train[idx_train]
    sub_x_test = x_test[idx_test]

    from sklearn.metrics import pairwise_distances

    p_dist = pairwise_distances(sub_x_train, sub_x_test).flatten()

    sigma_min = np.percentile(p_dist, 10)
    sigma_max = np.percentile(p_dist, 90)

    print(f'sigma_min: {sigma_min:.4f}')
    print(f'sigma_max: {sigma_max:.4f}')

    return sigma_min, sigma_max

def projection_non_negative_and_hyperplane(x: Any, hyperparams=Tuple) -> Any:
  r"""Projection onto the non-negative orthant:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  Args:
    x: pytree to project.
    hyperparams: ignored.
  Returns:
    projected pytree, with the same structure as ``x``.
  """
  a, b = hyperparams
  ## non-negative
  x = tree_util.tree_map(jax.nn.relu, x)

  scale = (tree_util.tree_vdot(a, x) -b) / tree_util.tree_vdot(a, a)
  return tree_util.tree_add_scalar_mul(x, -scale, a)


def pairwise_distances(X, Y=None):
    """
    Computes the pairwise distances between two sets of points.

    Args:
        X: A (n_samples, n_features) array of points.
        Y: Optional, a (m_samples, n_features) array of points.
           If None, computes distances between points in X.

    Returns:
        A (n_samples, m_samples) array of pairwise distances.
    """
    if Y is None:
        Y = X

    # Efficient computation of pairwise squared distances
    X_norm = jnp.sum(X ** 2, axis=1)[:, jnp.newaxis]  # Shape: (n_samples, 1)
    Y_norm = jnp.sum(Y ** 2, axis=1)[jnp.newaxis, :]  # Shape: (1, m_samples)
    cross_term = jnp.dot(X, Y.T)  # Shape: (n_samples, m_samples)

    # Pairwise squared distances
    pairwise_sq_dists = X_norm + Y_norm - 2 * cross_term

    # Ensure numerical stability (avoid negative values due to precision errors)
    pairwise_sq_dists = jnp.maximum(pairwise_sq_dists, 0.0)

    # Return pairwise distances
    return jnp.sqrt(pairwise_sq_dists)

def select_points_within_anchor(data, anchor_min, anchor_max):
    """
    Select points from a high-dimensional dataset that fall within the given anchor.

    Parameters:
    - data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features).
    - anchor_min (np.ndarray or list): Lower bounds for each dimension.
    - anchor_max (np.ndarray or list): Upper bounds for each dimension.

    Returns:
    - selected_data (np.ndarray): Filtered data points within the anchor.
    """
    if not (len(anchor_min) == len(anchor_max) == data.shape[1]):
        raise ValueError("Anchor bounds must match the number of features in the dataset")

    mask = np.all((data >= anchor_min) & (data <= anchor_max), axis=1)
    selected_data = data[mask]
    
    return selected_data

def find_intersection_anchor(dataset1, dataset2):
    """
    Finds the intersection anchor (bounding box) of two high-dimensional datasets.

    Parameters:
    - dataset1 (np.ndarray): First dataset of shape (n_samples1, n_features).
    - dataset2 (np.ndarray): Second dataset of shape (n_samples2, n_features).

    Returns:
    - intersection_min (np.ndarray): Minimum values of the intersection anchor.
    - intersection_max (np.ndarray): Maximum values of the intersection anchor.
    - valid (bool): Whether the intersection exists (True) or not (False).
    """
    if dataset1.shape[1] != dataset2.shape[1]:
        raise ValueError("Both datasets must have the same number of dimensions.")

    # Compute min/max for each dimension in both datasets
    d1_min = np.min(dataset1, axis=0)
    d1_max = np.max(dataset1, axis=0)
    
    d2_min = np.min(dataset2, axis=0)
    d2_max = np.max(dataset2, axis=0)

    # Find intersection bounds
    intersection_min = np.maximum(d1_min, d2_min)
    intersection_max = np.minimum(d1_max, d2_max)

    # Check if the intersection is valid
    valid = np.all(intersection_min <= intersection_max)

    return intersection_min, intersection_max, valid
class DensityRatioEstimator(metaclass=ABCMeta):

    def __init__(self):
        pass 

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def kfold_tuning(self, *args, **kwargs):
        pass

from sklearn.cluster import KMeans,MiniBatchKMeans, DBSCAN

def random_select(x_test,n_centers=100):
    index_centers = np.array(random.sample(list(range(len(x_test))), k=n_centers))
    centers = x_test[index_centers]

    return centers
def centroid_select(x_test, n_centers=100):
    """
    Select centroids using KMeans or MiniBatchKMeans based on the size of the input data.
    
    Parameters:
    - x_test: array-like, shape (n_samples, n_features)
        The data to fit.
    - n_centers: int, optional, default=100
        The number of cluster centers to find.
    
    Returns:
    - cluster_centers_: array, shape (n_centers, n_features)
        Coordinates of cluster centers.
    """
    # If dataset is small, use KMeans directly
    if len(x_test) <= 10_000:
        while n_centers > 1:
            try:
                # Try fitting KMeans with the given number of clusters
                kmeans = KMeans(n_init='auto', n_clusters=n_centers, random_state=0).fit(x_test)
                return kmeans.cluster_centers_
            except ValueError as e:
                print(f"Reducing n_centers due to error: {e}")
                # Reduce number of clusters by half if an error occurs
                n_centers //= 2
        raise ValueError("Unable to find appropriate number of clusters with the given data.")
    
    # For larger datasets, use MiniBatchKMeans for efficiency
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_centers,
                                 random_state=0,
                                 n_init='auto',
                                 batch_size=128).fit(x_test)
        return kmeans.cluster_centers_
    
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def density_based_sampling(x_test, n_centers=10):
    """
    Select a subset of points from the data based on density estimation.

    Parameters:
    - data: np.ndarray of shape (n_samples, n_features), the high-dimensional data points.
    - num_samples: int, the number of points to select.
    - bandwidth: float, bandwidth for the KDE (controls smoothness of density estimation).

    Returns:
    - selected_indices: np.ndarray of shape (num_samples,), indices of selected points.
    """
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(x_test)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    kde = grid.best_estimator_
    
    # Compute density for each point
    log_density = kde.score_samples(x_test)
    density = np.exp(log_density)

    # Normalize densities to sum to 1 (probabilities for sampling)
    density_prob = density / np.sum(density)

    # Sample indices based on density probability
    selected_indices = np.random.choice(len(x_test), size=n_centers, replace=False, p=density_prob)

    return x_test[selected_indices]

def dbscan_select(x_test,n_centers=100):
    """
    Using dbscan to select core samples
    """
    two_random_points = np.random.choice(len(x_test),size=(2,),replace=False)
    d = np.sqrt(np.sum((two_random_points[0] - two_random_points[1])**2))
    selected_points = np.zeros((n_centers,) + x_test.shape[1:])
    max_iters = 5
    iter = 0
    while iter<max_iters:
        clustering = DBSCAN(eps=d, min_samples=5).fit(x_test)
        clusters = np.unique(clustering.labels_)
        if len(clusters) > n_centers:
            for i,cluster in enumerate(clusters):
                if cluster!=-1:
                    ## random select
                    index_cluster = np.argwhere(clustering.labels_==cluster)
                    random_choice = np.random.choice(index_cluster,repalce=False)
                    selected_points[i] = x_test[random_choice]
            break
        else:
            d *= 1.25
            iter+=1

    return selected_points


def select_centers(x_test,n_centers=100, algo='kmeans',anchor_min=None, anchor_max=None):
    """
    Select centers using different algorithms based on the size of the input data.
    
    Parameters:
    - x_test: array-like, shape (n_samples, n_features)
    - n_centers: int, optional, default=100
    - algo: str, optional, default='kmeans'
    - anchor_min: np.ndarray or list, optional, default=None
    - anchor_max: np.ndarray or list, optional, default=None

    Returns:
    - cluster_centers_: array, shape (n_centers, n_features)
    Coordinates of cluster centers.
    """
    if anchor_min is not None and anchor_max is not None:
        x_test = select_points_within_anchor(x_test, anchor_min, anchor_max)
        print(f"Selected {len(x_test)} points within the anchor.")
    x_test = np.array(x_test, copy=True)  # copy original data from pandas DataFrame
    if algo == 'kmeans':
        return centroid_select(x_test,n_centers=n_centers)
    elif algo == 'random':
        return random_select(x_test,n_centers=n_centers)
    elif algo=='dbscan':
        return dbscan_select(x_test,n_centers=n_centers)
    elif algo=='density_based':
        return density_based_sampling(x_test, n_centers=n_centers)
    else:
        raise ValueError("Unknown algorithm")



def radial_bf(x,y, sigma):
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma ** 2))


def matern_bf(x,y,length_scale,nu):
    kernel = Matern(length_scale=length_scale, nu=nu)
    return kernel(x,y)

def polynomial_bf(x, degree):
    return np.array([x**i for i in range(degree + 1)]).T

from scipy.interpolate import BSpline

def bspline_bf(x, knots, degree):
    nknots = len(knots)
    B = np.zeros((nknots, len(x)))
    for i in range(nknots):
        B[i, :] = BSpline.basis_element(knots[i:i+degree+2])(x)
    return B.T

def _get_basis_func(name='radial'):

    if name == 'radial':
        return radial_bf
    elif name == 'matern':
        return matern_bf
    elif name == 'polynomial':
        return polynomial_bf
    elif name == 'bspline':
        return bspline_bf
    else:
        raise ValueError('Invalid basis function: %s' % name)
        
