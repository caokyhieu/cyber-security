

import os
import sys
import numpy as np 
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from tqdm import tqdm
from covariateshift_module.utils import radial_bf
import numpy as np
from cvxopt import matrix, solvers
from tqdm import tqdm
import math
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold

# # an implementation of Kernel Mean Matching
# # referenres:
# #  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
# #  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def compute_mmd(X, Z, sigma):
    K_XX = radial_bf(X, X, sigma=sigma)
    K_ZZ = radial_bf(Z, Z, sigma=sigma)
    K_XZ = radial_bf(X, Z, sigma=sigma)
    
    return np.mean(K_XX) + np.mean(K_ZZ) - 2 * np.mean(K_XZ)



def kernel_mean_matching(X, Z, kern='rbf', B=10.0, eps=None, kfold=5, sigma_values=None):
    nx = X.shape[0]
    nz = Z.shape[0]

    if eps is None:
        eps = 0.1  # Set a reasonable default

    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':

        # Define sigma search space if not provided
        if sigma_values is None:
            sigma_values = np.logspace(-2, 2, 10)  # 10 values between 0.01 and 100

        best_sigma = None
        best_qp_value = np.inf  # We want to minimize this

        # K-fold cross-validation
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        for sigma in tqdm(sigma_values, desc='Tuning sigma'):
            qp_values = []

            for train_idx, val_idx in kf.split(Z):
                Z_train, Z_val = Z[train_idx], Z[val_idx]

                # Compute kernel matrix for train set
                K_train = radial_bf(Z_train, Z_train, sigma=sigma)
                kappa_train = np.sum(radial_bf(Z_train, X, sigma=sigma), axis=1) * float(len(Z_train)) / float(nx)

                # Solve Quadratic Programming on train set
                K_train += np.eye(len(Z_train)) * 1e-6  # Regularization
                K_qp = matrix(K_train)
                kappa_qp = matrix(kappa_train)

                G_qp = matrix(np.vstack([
                    np.ones((1, len(Z_train))), 
                    -np.ones((1, len(Z_train))), 
                    np.eye(len(Z_train)), 
                    -np.eye(len(Z_train))
                ]))
                h_qp = matrix(np.hstack([
                    len(Z_train) * (1 + eps), 
                    len(Z_train) * (eps - 1), 
                    B * np.ones(len(Z_train)), 
                    np.zeros(len(Z_train))
                ]))

                sol = solvers.qp(K_qp, -kappa_qp, G_qp, h_qp, options={'show_progress': False})

                if sol['status'] == 'optimal':
                    beta_train = np.array(sol['x']).flatten()

                    # Compute QP objective function on validation set
                    K_val = radial_bf(Z_val, Z_train, sigma=sigma)
                    qp_value = 0.5 * np.dot(beta_train.T, np.dot(K_train, beta_train)) - np.dot(kappa_train.T, beta_train)
                    qp_values.append(qp_value)
                else:
                    qp_values.append(np.inf)  # Penalize non-converging solutions

            avg_qp_value = np.mean(qp_values)
            if avg_qp_value < best_qp_value:
                best_qp_value = avg_qp_value
                best_sigma = sigma

        print(f'Best sigma selected: {best_sigma}')

        # Compute final kernel with best sigma
        K = radial_bf(Z, Z, sigma=best_sigma)
        kappa = np.sum(radial_bf(Z, X, sigma=best_sigma), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('Unknown kernel')

    # Regularization to avoid numerical instability
    K += np.eye(nz) * 1e-6

    K = matrix(K)
    kappa = matrix(kappa)

    G = matrix(np.vstack([
        np.ones((1, nz)), 
        -np.ones((1, nz)), 
        np.eye(nz), 
        -np.eye(nz)
    ]))

    h = matrix(np.hstack([
        nz * (1 + eps), 
        nz * (eps - 1), 
        B * np.ones(nz), 
        np.zeros(nz)
    ]))

    sol = solvers.qp(K, -kappa, G, h, options={'show_progress': True})

    if sol['status'] != 'optimal':
        print(f"Warning: QP solver did not converge! Status: {sol['status']}")

    coef = np.array(sol['x'])

    # Debug output
    print(f"Solver status: {sol['status']}")
    print(f"First 10 coefficients: {coef[:10].flatten()}")

    return coef


# def kernel_mean_matching(X, Z, kern='rbf', B=1.0, eps=None):
#     nx = X.shape[0]
#     nz = Z.shape[0]
#     if eps == None:
#         eps = B/math.sqrt(nz)
#     if kern == 'lin':
#         K = np.dot(Z, Z.T)
#         kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
#     elif kern == 'rbf':
#         K = radial_bf(Z,Z,sigma=1)
#         kappa = np.sum(radial_bf(Z,X,sigma=1),axis=1)*float(nz)/float(nx)
#     else:
#         raise ValueError('unknown kernel')
#     K = matrix(K)
    
#     kappa = matrix(kappa)
#     G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
#     h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
#     sol = solvers.qp(K, -kappa, G, h,options={'show_progress':True})
#     coef = np.array(sol['x'])
#     return coef

## reinovate KMM methods
from jaxopt import OSQP
import jax.numpy as jnp
from covariance.covariance_matrix import create_covariance
def kernel_mean_matching_jax(X, Z, kern='rbf', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B/jnp.sqrt(nz)
    if kern == 'lin':
        ## c = 0
        K = create_covariance('linear_kernel',Z,Z,0) @ jnp.eye(len(Z))
        kappa = jnp.sum(create_covariance('linear_kernel',Z,X,0) @ jnp.eye(len(X)) *float(nz)/float(nx), axis=-1)
    elif kern == 'rbf':
        ## set length scale = 1
        K = create_covariance('squared_exponential',Z,Z,1.) @ jnp.eye(len(Z))
        kappa = jnp.sum(create_covariance('squared_exponential',Z,X,1.) @ jnp.eye(len(X)) *float(nz)/float(nx), axis=-1)
    else:
        raise ValueError('unknown kernel')
        
    G = jnp.r_[jnp.ones((1,nz)), -jnp.ones((1,nz)), jnp.eye(nz), -jnp.eye(nz)]
    h = jnp.r_[nz*(1+eps), nz*(eps-1), B*jnp.ones((nz,)), jnp.zeros((nz,))]
    
    qp = OSQP()
    sol = qp.run(params_obj=(K, -kappa), params_eq=None, params_ineq=(G, h)).params
    print(f"Solver status: {sol['status']}")
    coef = sol.primal
    return coef

def test_kmm():
    np.random.seed(0)
    nx = 1000
    nz = 100
    X = np.random.randn(nx,1)
    Z = np.random.randn(nz,1)
    coef = kernel_mean_matching(X, Z, kern='lin', B=1.0)
    ## compare with jax
    coef_jax = kernel_mean_matching_jax(jnp.array(X), jnp.array(Z), kern='lin', B=1.0)

    assert jnp.allclose(coef, coef_jax,atol=1e-3), 'The result of KMM is not consistent with jax'




from covariateshift_module.uLSIF import DensityRatioEstimator
from covariateshift_module.utils import _get_basis_func
from jaxopt import GradientDescent,ProjectedGradient,ScipyBoundedMinimize
from jaxopt.projection import projection_non_negative,projection_box
import pdb
class KernelMeanMatching(DensityRatioEstimator):

    def __init__(self,):
        
        pass 
    
    def solve_qp(self, x_denom,x_num):
        '''
        Args:
            x_denom: denominator 
            x_num: numerator 
            length_scale: hyperparam of kernel
        returns:
            coefficients
        '''
        ## first save the test data (x_num)
        self.x_num = x_num
        # pdb.set_trace()
        def loss_func(params, x_denom, x_num):
            weights = params[:-1]
            length_scale = params[-1].squeeze()
            H = create_covariance('squared_exponential',x_denom,x_denom,length_scale) @ jnp.eye(len(x_denom))
            kappa = create_covariance('squared_exponential',x_denom,x_num,length_scale) @ jnp.eye(len(x_num)) * len(x_denom)/len(x_num)
            remained = create_covariance('squared_exponential',x_num,x_num,length_scale) @ jnp.eye(len(x_num)) * ( (len(x_denom)/len(x_num))**2)
            return jnp.sum(0.5 * weights.T @ H @ weights) - jnp.sum(weights.T @ kappa) + 0.5  * jnp.sum(remained)
        

        # solver = ProjectedGradient(fun=loss_func, projection=projection_non_negative,maxiter=1000,stepsize=1e-2)
        init_params = 0.01 * jnp.ones((len(x_denom)+1,1))
        # res = solver.run(init_params, x_denom=x_denom, x_num=x_num)
        lbfgsb = ScipyBoundedMinimize(fun=loss_func, method="l-bfgs-b")
        lower_bounds = jnp.zeros_like(init_params)
        upper_bounds = jnp.ones_like(init_params) * jnp.inf
        bounds = (lower_bounds, upper_bounds)
        res = lbfgsb.run(init_params, bounds=bounds, x_denom=x_denom, x_num=x_num)

        print(res.state)
        ## save the params for kernel
        self.length_scale = res.params[-1].squeeze()
        return res.params[:-1].flatten()
    
    def kfold_tuning(self, x_denom, x_num):
        pass 
    def fit(self, x_denom,x_num):
        x_denom = jnp.array(x_denom)
        x_num = jnp.array(x_num)

        return self.solve_qp(x_denom,x_num)
    
    def compute_weights(self,x_denom):
        '''
        Args:
            x_denom: denominator data
        returns:
            weights
        '''
        x_denom = jnp.array(x_denom)
        def reduce_loss_func(weights, x_denom):
            H = create_covariance('squared_exponential',x_denom,x_denom,self.length_scale) @ jnp.eye(len(x_denom))
            kappa = create_covariance('squared_exponential',x_denom,self.x_num,self.length_scale) @ jnp.eye(len(self.x_num)) * len(x_denom)/len(self.x_num)
            return jnp.sum(0.5 * weights.T @ H @ weights) - jnp.sum(weights.T @ kappa) + 0.5 * (len(self.x_num)/len(x_denom))**2 * jnp.sum(H)
        solvers = ProjectedGradient(fun=reduce_loss_func, projection=projection_non_negative,maxiter=500)
        init_params = jnp.ones((len(x_denom),1))
        res = solvers.run(init_params, x_denom=x_denom)
        print(res.state)
        return res.params.flatten()
        

    
from covariateshift_module.utils import select_centers, find_intersection_anchor
from covariance.params import create_default_range_params, Params, generate_param_ranges_jax, batch_params
from covariance.covariance_matrix import create_params,create_covariance_v2
from covariateshift_module.metric_learning import NeuralNCA
import jax
from jax import random, lax
from itertools import product

class KernelMeanMatchingv2(DensityRatioEstimator):
    def __init__(self, alpha=0.1,num_basis=500, k_fold=5, basis_func='squared_exponential',**kwargs):
        
        # self.alpha = alpha
        self.num_basis = num_basis
        self.k_fold = k_fold
        self.fitting = False
        self.kernel_params = create_params(basis_func)
        self.lam = kwargs.get('lamda', 0.1)
        
        self.basis_func = basis_func
        self.set_centers = False
        self.used_nca = kwargs.get('nca', False)
        self.center_method = kwargs.get('center_method', 'kmeans')
        if self.used_nca:
            n_train = kwargs.get('n_train', 1000)
            input_dim = kwargs.get('input_dim', 1)
            output_dim = kwargs.get('output_dim', 10)
            self.n_epochs = int(kwargs.get('n_epochs', 40))
            ## rule of thumb for layer size
            layer_size = int(n_train/(2. * (input_dim + output_dim)))
            ## two layer network
            self.nca = NeuralNCA(input_dim=1,embedding_dim=1,
                                  key=jax.random.PRNGKey(123), layers= [layer_size, output_dim],)
        if self.used_nca:
            assert self.basis_func=='squared_exponential', 'Only support squared_exponential kernel for NCA'
            self.covar_func = lambda x, y, params: jnp.exp(-self.nca.pair_distance(x,y)/(1e5 * params.hyperparams['length_scale']))
        else:
            self.covar_func = lambda x, y, params: create_covariance_v2(name=self.basis_func, x=x, y=y, params=params) @ jnp.eye(len(y))
        
    def compute_score(self, coef, phi_y, yy, xy ):
        '''
        Helper fucntion to compute the score to evaluate learned parameters
        phi_x: numerator after projecting on the Hilbert space
        phi_y : denominator after projecting on the Hilbert space
        '''
        len_x = len(xy)
        H = phi_y.T @ yy @ phi_y
        h = ( xy @ phi_y ).sum(axis=0, keepdims=True).T * float(len(phi_y)) / float(len_x)
        score = coef.T @ H  @ coef/2 - h.T @ coef
        return score.ravel()
    
    def set_centers_manual(self, center):
        """
        Set centers manually
        """
        self.__centers = center
        self.set_centers = True
        pass
    
    def _choose_centers(self, x_test, n_centers,algo='kmeans',archor_min=None, archor_max=None):
        """
        Strategy to choose centers
        """

        points = select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=archor_min, anchor_max=archor_max)
        ## set center to true
        self.set_centers = True
        return points
    
    def solve_qp(self, phi_y,kernel_matrix_y, kernel_matrix_x_y):
        '''
        Args:
            phi_x: numerator after projecting on the Hilbert space
            phi_y: denominator after projecting on the Hilbert space

            alpha: weights to merge denominator with numerator
            lambda: the regularization
        returns:
            coefficients
        '''
        len_x = kernel_matrix_x_y.shape[-1]
        b = phi_y.shape[-1]
        P = phi_y.T @ kernel_matrix_y @ phi_y
        h = ( kernel_matrix_x_y @ phi_y ).sum(axis=0, keepdims=True).T * float(len(phi_y)) / float(len_x)
        q = -h
        qp = OSQP()
        G = -np.eye(b)
        h_ = np.zeros((b,1))
        constr = phi_y.mean(axis=0, keepdims=True)
        sol = qp.run(params_obj=(P, q), 
                     params_ineq=(G, h_),
                     params_eq=(constr, jnp.array([[1.]]))
                     ).params

        coef = jnp.atleast_2d(sol.primal)

        
        return coef

    def kfold_tuning(self, x_nu, x_de, centers,list_params, len_params=100):
        '''
        Function to do kfold for selecting the optimal hyperparameters
        '''
        
        n_nu = x_nu.shape[0]
        n_de = x_de.shape[0]
        key = random.PRNGKey(123)
        # score_cv = jnp.zeros((len(grid_kernel_params), len(grid_algo_params)))
        cv_index_nu = random.permutation(key, n_nu)
        cv_index_de = random.permutation(key, jnp.arange(n_nu, n_nu + n_de))
        x = jnp.vstack((x_nu, x_de))
        
        fold_size_nu = int(jnp.ceil(n_nu / self.k_fold))
        fold_size_de = int(jnp.ceil(n_de / self.k_fold))
        ## product grid_kernel_params and grid_algo_params
        ## using itertools
        def params_loop(param_carry, idx):
            par = param_carry
            sigma = par.slice_at_index(idx) 
            XX = self.covar_func(x=x, y=x, params=sigma)
            K =  self.covar_func(x=x, y=centers, params=sigma)
            def cv_loop(carry, params):
                # sigma, lambda_ = params
                cv_index = carry 
                
                index_te_nu = lax.dynamic_slice(cv_index_nu, (cv_index * fold_size_nu,),(fold_size_nu,))
                index_tr_nu = jnp.setdiff1d(cv_index_nu, index_te_nu,size=n_nu - fold_size_nu)

                index_te_de = lax.dynamic_slice(cv_index_de, (cv_index * fold_size_de,),(fold_size_de,))
                index_tr_de = jnp.setdiff1d(cv_index_de, index_te_de,size=n_de - fold_size_de)

                phi_y = K[index_tr_de, :]
                yy = XX[index_tr_de,:][:,index_tr_de]
                xy =    XX[index_tr_nu,:][:,index_tr_de]

                coef = self.solve_qp(phi_y, yy, xy)

                phi_y_1 = K[index_te_de, :]

                yy_te = XX[index_te_de,:][:,index_te_de]
                xy_te =    XX[index_te_nu,:][:,index_te_de]

                score = self.compute_score(coef, phi_y_1,yy_te, xy_te)

                return cv_index +1 , score
            ## first, run cv
            _, cv_score = lax.scan(cv_loop, 0, jnp.arange(self.k_fold))
            return par, jnp.nanmean(cv_score)
        ## second, run outer loop
        _, final_score = lax.scan(params_loop, list_params, jnp.arange(len_params))
        ## find min score
        score_cv_min_index = jnp.nanargmin(final_score)
        sigma_chosen = list_params.slice_at_index(score_cv_min_index)
        print('Finish K fold')
        return sigma_chosen
    
    def fit(self,x_train,x_test):
        '''
        Fitting the parameters of the model
        Args:
            x_train: covariates of training data (denominator)
            x_test: covariates of testing data (numerator)
        Notes:
            x_train and x_test should be normalized
        
        returns:
            estimated weights: array of weights (length == length of training)
        '''

        p_dist = pairwise_distances(x_train, x_test).flatten()
        sigma_min = jnp.percentile(p_dist, 10)  # 10th percentile
        sigma_max = jnp.percentile(p_dist, 90)  # 90th percentile

        sigma_range = create_default_range_params(self.kernel_params.kernel_name, {'length_scale': (sigma_min, sigma_max)})
        sigma_list = generate_param_ranges_jax(self.kernel_params.hyperparams, sigma_range, num_values=20)
        sigma_list = [Params(kernel_name=self.kernel_params.kernel_name, hyperparams=p) for p in sigma_list]
        # sigma_list = jnp.logspace(jnp.log10(sigma_min), jnp.log10(sigma_max), num=10)

        # need strategy to chose centers
        if not self.set_centers:
            archor_min, archor_max, valid = find_intersection_anchor(x_train,x_test)
            if not valid:
                archor_min = None 
                archor_max = None

            self.__centers = self._choose_centers(x_test,n_centers=self.num_basis,algo=self.center_method,archor_min=archor_min,archor_max=archor_max)

        ## fit nca if used
        if self.used_nca:
            ## prepare training data
            trainData = jnp.concatenate([x_train, x_test],axis=0)
            trainlabels = jnp.concatenate([jnp.zeros(len(x_train)), jnp.ones(len(x_test))],axis=0)
            self.nca.train(trainData, trainlabels,batchsize=256, stepsize=1e-2, n_epochs=self.n_epochs)
            print('Finish training NCA')
        
        ## product params
        list_params = sigma_list
        len_params = len(list_params)
        list_kernel_params = sigma_list
        batched_kernel_params = batch_params(list_kernel_params)
      
        sigma_chosen = self.kfold_tuning(x_test, x_train, self.__centers, batched_kernel_params ,len_params=len_params )
        
        phi_denominator = self.covar_func(x=x_train, y=self.__centers, params=sigma_chosen)
        yy = self.covar_func(x=x_train,y=x_train, params=sigma_chosen)
        xy = self.covar_func(x=x_test, y=x_train, params=sigma_chosen)
        self.__coef = jnp.maximum(self.solve_qp(phi_denominator,yy, xy),0.)

        ## reassign hyperparameters
        self.kernel_params = sigma_chosen

      
        score = self.compute_score(self.__coef,phi_denominator,yy,xy )
        print(f'Finished training, the minimized function value is:  {score}')
        self.fitting = True

    def compute_weights(self, x_train):
        '''
        Compute the weights for the training
        Args:
            x_train: training covariates
            centers: 
        returns:
            weights: the weights for the training
        '''
        assert self.fitting, 'Need to fit the model before computing the weights'
        jax.debug.print("Compute weight ") 
        # phi_x = create_covariance_v2(name=self.basis_func, x=x_train, y=self.__centers, params=self.kernel_params) @ jnp.eye(len(self.__centers))
        phi_x = self.covar_func(x=x_train, y=self.__centers, params=self.kernel_params)
        weights =  phi_x @ self.__coef
        
        return weights
