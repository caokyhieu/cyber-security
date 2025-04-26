import jax
import jax.numpy as jnp
from jax import random
from covariance.covariance_matrix import create_covariance,create_covariance_v2, create_params
from covariance.params import Params
import jax.numpy as jnp
from jax import jit, vmap,lax,random
from covariateshift_module.utils import DensityRatioEstimator

import jax.numpy as jnp
import pdb

from jaxopt import  ProjectedGradient
from covariance.covariance_matrix import create_covariance
from covariateshift_module.utils import (select_centers, pairwise_distances, find_intersection_anchor,
                                          projection_non_negative_and_hyperplane,estimate_sigma)
import numpy as np
from typing import Any
import optax
from jax import  random,value_and_grad
from tqdm import tqdm
from covariance.params import Params, generate_param_ranges_jax, create_default_range_params, batch_params


class KLIEP(DensityRatioEstimator):

    def __init__(self, num_basis:int=500 ,basis_func:str='squared_exponential', 
                 learning_rate:float=1e-4,a_tol:float=1e-6,
                 cv:int=5,max_iter:int=1000,input_dim=2,**kwargs):
        
        self.num_basis = num_basis
        self.kernel_params = create_params(basis_func)
        self.basis_func = basis_func
        # if basis_func == 'squared_exponential':
        #     self.basis_func = lambda x,y,length_scale: create_covariance(basis_func,x,y,length_scale) @ jnp.eye(len(y))
        self.fitting = False
        self.center_method = kwargs.get('center_method','kmeans')

        # ### init params
        self.theta = jnp.zeros((self.num_basis,1)) + 1e-6
        # self._weights = jnp.ones((self.num_basis,))/self.num_basis ## set number of basis as the number of mixtures
        # self._scales = jnp.ones((self.num_basis,input_dim ))
        # self._means = 1e-2 * jnp.ones((self.num_basis,input_dim))
        self.learning_rate = learning_rate
        self.fitting = False
        self.a_tol= a_tol
        self.cv= cv
        self.max_iter=max_iter

    def _choose_centers(self, x_test, n_centers,algo='kmeans',anchor_min=None,anchor_max=None):
        """
        Strategy to choose centers
        """

        return select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=anchor_min,anchor_max=anchor_max)
    

    def epoch_update(self, x_nu, x_de, centers, sigma):
       
        init_params = jnp.zeros(self.theta.shape,dtype="float32") + 1e-6
        a = jnp.mean(create_covariance_v2(name=self.basis_func, x=x_de, y=centers, params=sigma) @ jnp.eye(len(centers)) ,axis=0,keepdims=True).T
        
        # a = jnp.mean(self.basis_func(x_de, centers, sigma),axis=0,keepdims=True).T
        
        pg = ProjectedGradient(fun=lambda x: -self.unormalized_KL(x_nu, centers, x, sigma), 
                               projection=projection_non_negative_and_hyperplane,
                               maxiter=self.max_iter,
                               stepsize=self.learning_rate,
                               tol=self.a_tol,)
        pg_sol = pg.run(init_params=init_params,hyperparams_proj=(a,1.)).params
        print("Finished updating theta")
        return pg_sol


    def _compute_weights(self, x, centers,theta, sigma):
        """
        
        """
        theta = theta.reshape(-1,1)
        # phi_x = self.basis_func(x, centers, sigma)
        phi_x = create_covariance_v2(name=self.basis_func, x=x, y=centers, params=sigma) @ jnp.eye(len(centers)) 
        return phi_x @ theta 
    
    def unormalized_KL(self, x_nu,  centers, theta, sigma):
        score = self._compute_weights(x_nu, centers, theta, sigma)
        return jnp.mean(jnp.log(score))
    
    # def kfold_tuning(self, x_train, x_test, centers, sigma_list, cv=5,len_params= 10):
    #     """
    #     Perform k-fold cross-validation to tune the sigma parameter.

    #     Args:
    #     x_train: Training data (x_de).
    #     x_test: Test data (x_nu).
    #     centers: Centers for the Gaussian kernels.
    #     sigma_list: List of sigma values to tune.
    #     cv: Number of cross-validation folds.

    #     Returns:
    #     choiced_sigma: The sigma value that optimizes the unnormalized KL divergence.
    #     """
    #     n_nu = len(x_test)
    #     x_test = random.permutation(random.PRNGKey(0), x_test)  # Ensure index_nu is of integer type
    #     fold_size = int(jnp.ceil(n_nu / cv))
    #     ## change to list of batch
    #     x_test_list = [x_test[i:i+fold_size,...] for i in range(0, len(x_test), fold_size)]
    #     if len(x_test_list[-1]) < fold_size:
    #         x_test_list[-1] =  jnp.pad(x_test_list[-1], ((0,fold_size - len(x_test_list[-1])),(0,0)), mode='reflect')
        
    #     x_test_list = jnp.array(x_test_list)
       
    #     def outer_scan_func(carry, idx):
    #         params = carry
    #         inp_sigma = params.slice_at_index(idx)
    #         def inner_scan_func(carry, batch_data):
    #             '''
    #             inner scan will scan through test data
    #             '''
    #             i = carry 

    #             theta = self.epoch_update(batch_data, x_train, centers, inp_sigma)
    #             new_score = self.unormalized_KL(batch_data, centers, theta, inp_sigma)
    #             return i+1, new_score 
    #         idx,all_score = lax.scan(inner_scan_func, 0, x_test_list)

    #         return params, jnp.nanmean(all_score)
        
    #     _, score = lax.scan(outer_scan_func, sigma_list, jnp.arange(len_params))
    

    #     # Optimized sigma
    #     opt_args = jnp.nanargmax(score)
    #     choiced_sigma = sigma_list.slice_at_index(opt_args)
    #     print(f"Optimized UKL: {score[opt_args]:.2f}")
    #     return choiced_sigma
    
    def kfold_tuning(self, x_train, x_test, centers, sigma_list, cv=5, len_params=10):
        """
        Perform k-fold cross-validation to tune the sigma parameter.
        """

        n_nu = len(x_test)
        x_test = random.permutation(random.PRNGKey(0), x_test)

        fold_size = int(jnp.ceil(n_nu / cv))

        best_score = -jnp.inf
        # choiced_sigma = sigma_list[0]
        all_avg_scores = []  # will store JAX scalars

        for sigma in tqdm(sigma_list, desc="Tuning sigmas"):
            fold_scores = []

            for fold_idx in range(cv):
                start = fold_idx * fold_size
                end = min((fold_idx + 1) * fold_size, n_nu)
                batch_data = x_test[start:end]

                theta = self.epoch_update(batch_data, x_train, centers, sigma)
                theta = jax.lax.stop_gradient(theta)
                score = self.unormalized_KL(batch_data, centers, theta, sigma)
                fold_scores.append(score)

            fold_scores = jnp.stack(fold_scores)  # turn list into (cv,) array
            avg_score = jnp.nanmean(fold_scores)
            all_avg_scores.append(avg_score)
            
            
        all_avg_scores = jnp.stack(all_avg_scores)  # (len_params,)

        best_idx = jnp.nanargmax(all_avg_scores)
        # best_idx = np.array(jax.device_get(best_idx))

        sigma_list = batch_params(sigma_list)
        # choiced_sigma = sigma_list[best_idx]
        choiced_sigma =sigma_list.slice_at_index(best_idx)

        # print(f"Optimized UKL: {all_avg_scores[best_idx]:.2f}")
        return choiced_sigma


    # def kfold_tuning(self, *args, **kwargs):
    #     pass
    def fit(self, x_train,x_test):
        ## find anchor first
        anchor_min, anchor_max,valid = find_intersection_anchor(x_train,x_test)
        if not valid:
            anchor_min = None 
            anchor_max = None
        centers = self._choose_centers(x_test, n_centers=self.num_basis, algo=self.center_method,anchor_min=anchor_min,anchor_max=anchor_max)
        ## kfold
        # sigma_list = jnp.linspace(0.1,2,10) ## here we tune the sigma in range(0.1 ,2)
        # p_dist = pairwise_distances(x_train, x_test).flatten()
        # # pdb.set_trace()
        # sigma_min = np.percentile(p_dist, 10)  # 10th percentile
        # print(f'sigma_min')

        # sigma_max = np.percentile(p_dist, 90)  # 90th percentile
        # print(f'sigma_max')
        sigma_min, sigma_max = estimate_sigma(x_train,x_test, n_samples=10_000)

        sigma_range = create_default_range_params(self.kernel_params.kernel_name, {'length_scale': (sigma_min, sigma_max)})
        sigma_list = generate_param_ranges_jax(self.kernel_params.hyperparams, sigma_range, num_values=20)
        sigma_list = [Params(kernel_name=self.kernel_params.kernel_name, hyperparams=p) for p in sigma_list]
        len_sigma_list = len(sigma_list)
        # sigma_list = batch_params(sigma_list) ## just batch params when using kfold parallel
        self.sigma = self.kfold_tuning(x_train,x_test,centers,sigma_list,cv=self.cv,len_params=len_sigma_list)
        ## do again with optimal sigma
        self.theta = jnp.maximum(self.epoch_update(x_test, x_train, centers,self.sigma),0.)
        self.centers = centers
        # self.centers = x_test
        self.fitting = True 

    def compute_weights(self, x_train):
        assert self.fitting, "Need to fitting model first"
        return self._compute_weights(x_train,  self.centers, self.theta, self.sigma)
    
class GradKLIEP(DensityRatioEstimator):

    def __init__(self, alpha=0.1,num_basis=500, k_fold=5, basis_func='squared_exponential',**kwargs):
        
        self.alpha = alpha
        self.num_basis = num_basis
        self.k_fold = k_fold
        self.fitting = False
        
        self.kernel_params = create_params(basis_func)
        self.basis_func = basis_func
        self.fitting = False

    def _choose_centers(self, x_test, n_centers, algo='kmeans'):
        """
        Strategy to choose centers using k-means or random selection.
        """
        return select_centers(x_test, n_centers=n_centers, algo=algo)
    
    def loss_function(self,params:Params,  theta, x_nu, x_de, centers):
        """
        Loss function: negative unnormalized KL divergence.
        Args:
        params: The parameters of kernel
        theta: The current estimate of weights
        x_nu: The test data
        x_de: The training data
        centers: The centers (choosen from test data)
        """
        phi_nu = create_covariance_v2(name=self.basis_func, x= x_nu, y=centers, params=params) @ jnp.eye(len(centers))
        ## make sure theata >0
        theta = jax.nn.relu(theta)
        r_nu = phi_nu @ theta
        loss = -jnp.mean(jnp.log(r_nu))
        
        # # Regularization term for hyperplane
        # phi_de = create_covariance_v2(name=self.basis_func, x= x_de, y=centers, params=params) @ jnp.eye(len(centers))
        # constr = (jnp.mean(phi_de @ theta) -1.)**2
        
        return loss 
    
    def tune_hyperparameters(self, x_nu, x_de, centers, init_params: Params,
                            learning_rate=0.01, num_steps=100, batch_size=64, seed=42):
        """
        Tune parameters using gradient descent.
        """
        key = random.PRNGKey(seed)
        d = x_nu.shape[-1]
        
        # Convert hyperparameters to JAX arrays
        hyperparams = {k: jnp.array(v) for k, v in init_params.hyperparams.items()}
        
        # Initialize hyperparameters based on the kernel type
        if self.basis_func == 'ard':
            hyperparams["length_scale"] = jnp.ones((d,))
        elif self.basis_func == 'sm':
            hyperparams['weights'] = jnp.ones((5,))
            hyperparams['scales'] = jnp.ones((5, d))
            hyperparams['means'] = jnp.ones((5, d))
        elif self.basis_func == 'mixture_linear':
            hyperparams['weights'] = jnp.ones((5,))
            hyperparams['biases'] = jnp.ones((5, ))
            

        params = Params(init_params.kernel_name, hyperparams)
        init_coef = jnp.ones((len(centers),1))
        
        # Define optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init((params, init_coef))

        loss_and_grad_fn = value_and_grad(self.loss_function, argnums=(0, 1))
        
        # Function to process a single batch
        def step_fn(carry, batch):
            params,coef, opt_state = carry
            x_nu_batch, x_de_batch = batch[0], batch[1]
            ## 
            # loss_value, (grad_params, grad_coef) = loss_and_grad_fn(params, x_nu_batch[0], x_de_batch[0])
            (loss_value, (grad_params, grad_coef)) = loss_and_grad_fn(params, coef, x_nu_batch[0], x_de_batch[0], centers)
        
            updates, opt_state = optimizer.update((grad_params, grad_coef), opt_state,(params, coef))
            new_params, new_coef = optax.apply_updates((params, coef), updates)
            ## cosntraints
            a = create_covariance_v2(name = self.basis_func, x=x_de,y=centers,params=new_params) @ jnp.eye(len(centers))
            a = jnp.mean(a,axis=0,keepdims=True).T
            new_coef = projection_non_negative_and_hyperplane(new_coef,(a,1.))

            return (new_params, new_coef, opt_state), loss_value
           

        n_steps = tqdm(range(num_steps), desc="Training Progress")

        for step in n_steps:
            # Shuffle data at the start of each epoch
            key, subkey = random.split(key)
            shuffled_indices_nu = random.permutation(subkey, len(x_nu))
            shuffled_indices_de = random.permutation(subkey, len(x_de))

            x_nu = x_nu[shuffled_indices_nu]
            x_de = x_de[shuffled_indices_de]

            # Prepare batches
            x_nu_batches = [x_nu[i: i +batch_size][jnp.newaxis,...] for i in range(0, len(x_nu), batch_size)]
            ## truncated the batch size
            if len(x_nu_batches[-1][0]) < batch_size:
                pad_size = batch_size - x_nu_batches[-1].shape[1]
                x_nu_batches[-1] = jnp.pad(x_nu_batches[-1], ((0, 0), (0, pad_size), (0, 0)), mode='reflect')
            # x_nu_batches[0] = x_nu_batches[0][:batch_size]
            x_de_batches = x_de[jnp.newaxis,...]
            
            batches = list(zip(x_nu_batches, [x_de_batches] * len(x_nu_batches)))
            # pdb.set_trace()

            # Scan over the mini-batches
            (params,init_coef, opt_state), losses = lax.scan(step_fn, (params,init_coef, opt_state), batches)

            # Calculate the average loss
            avg_loss = jnp.mean(losses)
            n_steps.set_postfix_str(f"Step {step}/{num_steps} - Avg Loss: {avg_loss:.6f}")

        return (params,init_coef)
    
    def kfold_tuning(self, x_nu, x_de, sigma_list, lambda_list, centers):
        '''
        Function to do kfold for selecting the optimal hyperparameters
        '''
        pass

    def fit(self, x_train, x_test,learning_rate=0.01, num_steps=100,batch_size=128):
        """
        Fit the model and tune hyperparameters dynamically.

        Args:
            x_train: Covariates of training data (denominator).
            x_test: Covariates of testing data (numerator).
        """
        # Initialize centers
        self.__centers = self._choose_centers(x_test, n_centers=self.num_basis, algo='kmeans')

        # Tune hyperparameters dynamically
        tuned_params = self.tune_hyperparameters(x_test, x_train, self.__centers, self.kernel_params,
                                                 learning_rate=learning_rate,num_steps=num_steps,
                                                 batch_size=batch_size)
        ## reassign hyperparameters
        self.kernel_params = tuned_params[0]
        self.__coef = jnp.maximum(tuned_params[1],0.)


        self.fitting = True
        print(f"Finished training with tuned parameters: {tuned_params}")
    
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
        # jax.debug.print("Compute weight ") 
        # phi_x = self.basis_func(x_train,self.__centers,self.__sigma)
        phi_x = create_covariance_v2(name=self.basis_func, x=x_train,y=self.__centers,params=self.kernel_params) @ jnp.eye(len(self.__centers)) 
        weights =  phi_x @ self.__coef
        
        return weights



# import numpy as np
# from covariateshift_code.utils import _get_basis_func
# from scipy.optimize import minimize
# from tqdm import tqdm

# class KLIEP(DensityRatioEstimator):

#     def __init__(self, num_basis:int=50 ,basis_func:str='radial', 
#                  learning_rate:float=1e-2,a_tol:float=1e-6,
#                  cv:int=5,max_iter:int=1000):
        
#         self.num_basis = num_basis
#         self.basis_func = _get_basis_func(name=basis_func)
#         self.fitting = False

#         ### init params
#         self.theta = np.zeros((self.num_basis,1)) + 1e-6
#         self.learning_rate = learning_rate
#         self.fitting = False
#         self.a_tol= a_tol
#         self.cv= cv
#         self.max_iter=max_iter

#         pass
    
#     def _choose_centers(self, x_test, n_centers,algo='kmeans'):
#         """
#         Strategy to choose centers
#         """

#         return select_centers(x_test,n_centers=n_centers, algo=algo)


#     def epoch_update(self,x_nu, x_de, centers, sigma):
       
#         ## init theta 
#         theta = np.zeros((self.num_basis,),dtype="float64") + 1e-6
#         n_de = len(x_de)
        
#         bnds = ((0, None),) * len(theta)
#         result = minimize(lambda x: -self.unormalized_KL(x_nu, x, centers, sigma), theta,
#                           method='SLSQP',bounds=bnds,
#                           options= {
#                                     "maxiter":self.max_iter,
#                                     "ftol":self.a_tol
#                                     },
#                         constraints=({'type': 'eq', 'fun': lambda w: sum(self._compute_weights(x_de,w,centers,sigma)) - n_de}))
#         print("Finished updating theta")
#         return result.x

#     def _compute_weights(self, x, theta, centers, sigma):
#         """
        
#         """
#         theta = theta.reshape(-1,1)
#         phi_x = self.basis_func(x, centers, sigma)
#         return np.where((phi_x @ theta==0),1e-12,phi_x @ theta) 
    
#     def unormalized_KL(self, x_nu, theta, centers, sigma):
#         theta = theta.reshape(-1,1)

#         score = self._compute_weights(x_nu, theta, centers, sigma)
#         return np.mean(np.log(score))
    
#     def kfold_tuning(self, x_train,x_test,centers, sigma_list, cv=5):
#         """
#         x_train here like x_de
#         x_test: x_nu
        
#         """
#         n_nu =  len(x_test)
#         index_nu = np.random.permutation(n_nu)
#         fold_size = int(np.ceil(n_nu/cv))
#         n_sigma = len(sigma_list)
#         # optimized_theta = np.zeros((self.num_basis, n_sigma))

#         score = np.zeros((n_sigma,))
#         iterations = tqdm(enumerate(sigma_list))
#         for j,sigma in iterations:
#             temp_score = np.zeros((cv,))
#             # print(f"temp score: {temp_score}")
#             for i,start_index in enumerate(range(0,n_nu, fold_size)):
#                 ## begin each fold
#                 x_nu = x_test[index_nu[start_index:start_index + fold_size]]
#                 theta = self.epoch_update(x_nu,x_train, centers, sigma)
#                 # optimized_theta[:,i:i+1] = theta 
#                 # print(f"score: {self.unormalized_KL(x_nu, theta, centers, sigma)}")
#                 temp_score[i] = self.unormalized_KL(x_nu, theta, centers, sigma)
#             score[j] = np.mean(temp_score)
#             iterations.set_description(f"sigma: {sigma:.2f}, score: {score[j]:.2f}")
#         ## I want to print score in tqdm iteration


#         ## optimized sigma 
#         opt_args = np.argmax(score)
#         choiced_sigma = sigma_list[opt_args]
#         print(f"optimized UKL: {score[opt_args]:.2f}")
#         return choiced_sigma
    
#     def fit(self, x_train,x_test):
#         n_nu =  len(x_test)
#         index_nu = np.random.permutation(n_nu)

#         centers = self._choose_centers(x_test, n_centers=self.num_basis, algo='kmeans')
#         p_dist = pairwise_distances(x_train, x_test).flatten()
#         sigma_min = np.percentile(p_dist, 10)  # 10th percentile
#         sigma_max = np.percentile(p_dist, 90)  # 90th percentile

#         sigma_list = jnp.logspace(jnp.log10(sigma_min), jnp.log10(sigma_max), num=10)
#         #sigma_list = np.linspace(0.1,2,10) ## here we tune the sigma in range(0.1 ,2)

#         self.sigma = self.kfold_tuning(x_train,x_test,centers,sigma_list,cv=self.cv)
#         ## do again with optimal sigma
#         self.theta = self.epoch_update(x_test, x_train, centers, self.sigma)
#         self.centers = centers
#         self.fitting = True 
#         pass 

#     def compute_weights(self, x_train):
#         assert self.fitting, "Need to fitting model first"
#         return self._compute_weights(x_train, self.theta, self.centers, self.sigma)
#         ## first split x_test to k_fold