import torch 
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.exact_prediction_strategies import prediction_strategy
import jax
import jax.numpy as jnp
from jax import vmap,pmap
import pdb
import numpy as np
from gpytorch import settings
from utils.pyro_utils import covariance_matrix_helper,cholesky_matrix_helper,multi_radial_basis_function,radial_basic_function
from utils.pyro_utils import torch_from_numpy

def batch_multivariate_SI_predict_L( X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,dist_XX):
    """
    Multivariate Normal prediction with spatially correlated noise

    Args:
        X: training inputs
        Y: training targets
        X_test: test inputs
        alpha: spatial correlation parameter
        p: Mean function parameter
        theta: Mean function parameter
        gamma: Mean function parameter
        sigma_sq: spatial correlation parameter
        dist_pp: pairwise distance matrix between test points
        dist_pX: pairwise distance matrix between test and training points
        L_dist_XX: cholesky of pairwise distance matrix between training points
    
    Returns:
        predictive distribution MultivariateNormal
    """
    #### get batch shape
    
    # ##### extend X, X_test, Y to batch shape
    # X = jnp.expand_dims(X,axis=0).repeat(batch_shape,axis=0)
    # X_test = jnp.expand_dims(X_test,axis=0).repeat(batch_shape,axis=0)
    # Y = jnp.expand_dims(Y,axis=0).repeat(batch_shape,axis=0)

    ## use vmap
    batch_shape = alpha.shape[0]
    train_shape = len(X)
    test_shape = len(X_test)
    
    k_pp_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha,dist_pp)
    k_pX_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha, dist_pX)
    k_XX_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha,dist_XX)
    ## concat columns
    full_trainr_cov = jnp.concatenate((k_XX_batch,jnp.transpose(k_pX_batch,axes=(0,2,1))),axis=2)
    full_testr_cov = jnp.concatenate((k_pX_batch,k_pp_batch),axis=2)
    ## concat rows
    full_cov = jnp.concatenate((full_trainr_cov,full_testr_cov),axis=1)
    full_inputs = jnp.concatenate((X,X_test),axis=0)
    ## compute mean
    full_mean = vmap(multi_radial_basis_function,(None,0,0))(full_inputs,p,gamma) @ jnp.expand_dims(theta,axis=2)
    full_mean = full_mean.squeeze(-1)
    train_mean = full_mean[:,:train_shape]
    train_cov = full_cov[:,:train_shape,:train_shape]
    full_mean,train_mean,full_cov,train_cov =  torch_from_numpy([np.array(full_mean),
                                                                 np.array(train_mean),
                                                                 np.array(full_cov),
                                                                 np.array(train_cov)])
    # pdb.set_trace()
    train_output = MultivariateNormal(train_mean,train_cov)
    full_output = MultivariateNormal(full_mean,full_cov)
    
    ### change to pytorch
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood.noise_covar = 1.
    likelihood.eval()
    train_inputs = torch.tensor(np.array(X)).unsqueeze(0).expand(batch_shape,-1,-1)
    train_targets = torch.tensor(np.array(Y)).unsqueeze(0).expand(batch_shape,-1)
    _prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=train_targets,
                    likelihood=likelihood,
                )

    with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = _prediction_strategy.exact_prediction(full_mean, full_cov)

    # Reshape predictive mean to match the appropriate event shape
    predictive_mean = predictive_mean.view(batch_shape, test_shape).contiguous()

    return full_output.__class__(predictive_mean, predictive_covar)

def batch_SI_predict_L( X, Y , X_test, alpha, p, theta, gamma, sigma_sq,dist_pp,dist_pX,dist_XX):
    """
    Multivariate Normal prediction with spatially correlated noise

    Args:
        X: training inputs
        Y: training targets
        X_test: test inputs
        alpha: spatial correlation parameter
        p: Mean function parameter
        theta: Mean function parameter
        gamma: Mean function parameter
        sigma_sq: spatial correlation parameter
        dist_pp: pairwise distance matrix between test points
        dist_pX: pairwise distance matrix between test and training points
        L_dist_XX: cholesky of pairwise distance matrix between training points
    
    Returns:
        predictive distribution MultivariateNormal
    """
    #### get batch shape
    
    # ##### extend X, X_test, Y to batch shape
    # X = jnp.expand_dims(X,axis=0).repeat(batch_shape,axis=0)
    # X_test = jnp.expand_dims(X_test,axis=0).repeat(batch_shape,axis=0)
    # Y = jnp.expand_dims(Y,axis=0).repeat(batch_shape,axis=0)

    ## use vmap
    batch_shape = alpha.shape[0]
    train_shape = len(X)
    test_shape = len(X_test)
    
    k_pp_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha,dist_pp)
    k_pX_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha, dist_pX)
    k_XX_batch = vmap(covariance_matrix_helper,(0,0,None))(sigma_sq, alpha,dist_XX)
    ## concat columns
    full_trainr_cov = jnp.concatenate((k_XX_batch,jnp.transpose(k_pX_batch,axes=(0,2,1))),axis=2)
    full_testr_cov = jnp.concatenate((k_pX_batch,k_pp_batch),axis=2)
    ## concat rows
    full_cov = jnp.concatenate((full_trainr_cov,full_testr_cov),axis=1)
    full_inputs = jnp.concatenate((X,X_test),axis=0)
    ## compute mean
    # pdb.set_trace()
    full_mean = vmap(radial_basic_function,(None,0,0))(full_inputs,p,gamma) @ jnp.expand_dims(theta,axis=2)
    full_mean = full_mean.squeeze(-1)
    train_mean = full_mean[:,:train_shape]
    train_cov = full_cov[:,:train_shape,:train_shape]
    full_mean,train_mean,full_cov,train_cov =  torch_from_numpy([np.array(full_mean),
                                                                 np.array(train_mean),
                                                                 np.array(full_cov),
                                                                 np.array(train_cov)])
    train_output = MultivariateNormal(train_mean,train_cov)
    full_output = MultivariateNormal(full_mean,full_cov)
    
    ### change to pytorch
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood.noise_covar = 1.
    likelihood.eval()
    train_inputs = torch.tensor(np.array(X)).unsqueeze(0).expand(batch_shape,-1,-1)
    train_targets = torch.tensor(np.array(Y)).unsqueeze(0).expand(batch_shape,-1)
    _prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=train_targets,
                    likelihood=likelihood,
                )

    with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = _prediction_strategy.exact_prediction(full_mean, full_cov)

    # Reshape predictive mean to match the appropriate event shape
    predictive_mean = predictive_mean.view(batch_shape, test_shape).contiguous()

    return full_output.__class__(predictive_mean, predictive_covar)