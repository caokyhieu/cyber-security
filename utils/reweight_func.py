import jax.numpy as jnp
import numpy as np


def concentration_weights(weights,value=1.):
    """
    Concentrate the weight around the value
    """
    # weights = np.exp(weights)
    # print(weights)
    # print(value + (weights - (np.max(weights)+np.min(weights))/2)/(np.max(weights)-np.min(weights)))
    result = value + (weights - (jnp.max(weights)+jnp.min(weights))/2)/(jnp.max(weights)-jnp.min(weights) + 1e-8)
    ## fill nan array
    return jnp.nan_to_num(result,1)

def normalize_weights(weights):
    """
    Standize weights to sum to 1
    """
    return weights/weights.sum()

def mean_normalize_weights(weights):
    """
    Standize weights by the mean
    Args:
        weights (np.array): weights to be normalized
    Returns:
        np.array: normalized weights
    """
    return weights/weights.mean()

def median_normalize_weights(weights,threshold=0.5):
    """
    Standize weights by the median
    Args:
        weights (np.array): weights to be normalized
    Returns:
        np.array: normalized weights
    """
    if threshold == 0.5:
        return weights/jnp.median(weights)
    else:
        flatten_arr = np.ravel(weights)
        flatten_arr = jnp.sort(flatten_arr)
        n = int(len(flatten_arr) * threshold)
        return weights/flatten_arr[n]

def max_normalize_weights(weights):
    """
    Standize weights by the max
    Args:
        weights (np.array): weights to be normalized
    Returns:
        np.array: normalized weights
    """
    return weights/weights.max()

### reweights for covariance matrix
def _get_weights_from_cost_matrix(cost_matrix,name='concentration'):
    """
    get weights from cost matrix
    Args: 
        cost_matrix (np.array): cost/distance
    """
    if len(cost_matrix.shape)==1:
        distance = cost_matrix
    elif len(cost_matrix.shape)==2:
        distance = jnp.array(jnp.sum(cost_matrix,axis=1))
    else:
        raise ValueError("cost matrix must be a vector or a matrix")
    ##
    if name=='median':
        return  median_normalize_weights(distance,threshold=0.7)
    elif name =='mean':
        return mean_normalize_weights(distance)
    elif name=='':
        return distance
    elif name=='concentration':
        return concentration_weights(distance,value=1.)
    elif name=='average':
        result = concentration_weights(distance,value=1.)
        new_result = jnp.ones(result.shape)
        new_result = new_result.at[:len(result)].set(result.mean())
        return new_result
    else:
        raise ValueError('scale method not found')
    


def EV_scale_covariance_matrix(cov_matrix, cost_matrix):
    """
    Reweight covariance matrix by eigen decomposition
    Args:
        cov_matrix (np.array): covariance matrix/ vector KMM
        cost_matrix (np.array): cost/distance matrix
    Returns:
        np.array: reweighted covariance matrix
    """
    # eigen decomposition
    w, v = jnp.linalg.eigh(cov_matrix)
    # get weights from cost matrix
    distance =  _get_weights_from_cost_matrix(cost_matrix)
    
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cov_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    w = jnp.multiply(w,var)
    ## reweight covariance matrix through eigen values
    k = (v @ jnp.diag(w)) @ v.T

    return k

def VM_scale_covariance_matrix(cov_matrix, cost_matrix):
    """
    Reweight covariance matrix by variance multiplication
    Args:
        cov_matrix (np.array): covariance matrix/vector KMM
        cost_matrix (np.array): cost/distance matrix
    Returns:
        np.array: reweighted covariance matrix
    """
    
    distance =  _get_weights_from_cost_matrix(cost_matrix)
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cov_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    # covar = jnp.matmul(jnp.sqrt(var).reshape(-1,1),jnp.sqrt(var).reshape(1,-1))
    diag_matrix = jnp.diag(jnp.sqrt(var).reshape(-1,))
    ## reweight covariance matrix through variance multiplication
    # k = jnp.multiply(covar,cov_matrix)
    k = diag_matrix @ cov_matrix @ diag_matrix.T
    return k


def power_scale_covariance_matrix(cov_matrix, cost_matrix):
    """
    Reweight covariance matrix by power function
    Args:
        cov_matrix (np.array): covariance matrix/vector KMM
        cost_matrix (np.array): cost/distance matrix
    Returns:
        np.array: reweighted covariance matrix
    """
    distance =  _get_weights_from_cost_matrix(cost_matrix)
    distance = distance.mean()
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cov_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    covar = jnp.matmul(jnp.sqrt(var).reshape(-1,1),jnp.sqrt(var).reshape(1,-1))
    ## reweight covariance matrix through power function
    k = jnp.multiply(covar,cov_matrix)

    return k

# Function for Cholesky decomposition

def VM_scale_cholesky_matrix(cholesky_matrix, cost_matrix):
    """
    Reweight cholesky matrix by variance multiplication
    Args:
        cholesky_matrix (np.array): cholesky matrix/vector KMM
        cost_matrix (np.array): cost/distance matrix
    Returns:
        np.array: reweighted covariance matrix
    """
    distance =  _get_weights_from_cost_matrix(cost_matrix)
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cholesky_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    covar = jnp.diag(jnp.sqrt(var).reshape(-1,))
    ## reweight covariance matrix through variance multiplication
    k = jnp.matmul(covar,cholesky_matrix)

    return k

def power_scale_cholesky_matrix(cholesky_matrix, cost_matrix):
    """
    Reweight cholesky matrix by power function
    Args:
        cholesky_matrix (np.array): cholesky matrix/vector KMM
        cost_matrix (np.array): cost/distance matrix
    Returns:
        np.array: reweighted covariance matrix
    """
    distance =  _get_weights_from_cost_matrix(cost_matrix)
    distance = distance.mean()
    N_LABELS = len(cost_matrix)
    var = jnp.ones((cholesky_matrix.shape[0])) 
    var = var.at[:N_LABELS].set(distance)
    covar = jnp.diag(jnp.sqrt(var).reshape(-1))
    ## reweight covariance matrix through power function
    k = jnp.matmul(covar,cholesky_matrix)

    return k


