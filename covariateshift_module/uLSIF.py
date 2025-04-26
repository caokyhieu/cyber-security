import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from covariateshift_module.utils import  projection_non_negative_and_hyperplane
import warnings

from tqdm import tqdm
from covariance.covariance_matrix import create_params,create_covariance_v2
from covariateshift_module.KLIEP import KLIEP
from covariance.params import Params, generate_param_ranges_jax, create_default_range_params, batch_params
from jax import random
import jax.numpy as jnp
import jax
from covariateshift_module.utils import DensityRatioEstimator, select_centers, pairwise_distances, estimate_sigma
from itertools import product

# Set warnings to be treated as errors
warnings.filterwarnings("error")
    
from jaxopt import OSQP
from jaxopt.linear_solve import solve_cg

from jax import lax
from covariateshift_module.utils import find_intersection_anchor
from covariateshift_module.metric_learning import NeuralNCA
class uLSIF(DensityRatioEstimator):

    def __init__(self, alpha=0.1,num_basis=500, k_fold=5, basis_func='squared_exponential',**kwargs):
        
        # self.alpha = alpha
        self.num_basis = num_basis
        self.k_fold = k_fold
        self.fitting = False
        self.kernel_params = create_params(basis_func)
        self.lam = kwargs.get('lamda', 0.1)
        self.algo_params = Params(kernel_name='uLSIF', hyperparams={'alpha': alpha,'lamda':self.lam})
        
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
        
    def compute_score(self, coef,phi_x,phi_y, alpha, lamda ):
        '''
        Helper fucntion to compute the score to evaluate learned parameters
        phi_x: numerator after projecting on the Hilbert space
        phi_y : denominator after projecting on the Hilbert space
        '''

        H = (1. - alpha) * (jnp.dot(phi_y.T, phi_y) / len(phi_y)) + alpha * (jnp.dot(phi_x.T, phi_x) / len(phi_x)) 
        h = phi_x.mean(axis=0, keepdims=True).T 
        b = len(H)
        score = coef.T @ (H + lamda * np.eye(b) ) @ coef/2 - h.T @ coef
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
    
    def solve_qp(self, phi_x, phi_y, alpha, lamda):
        '''
        Args:
            phi_x: numerator after projecting on the Hilbert space
            phi_y: denominator after projecting on the Hilbert space

            alpha: weights to merge denominator with numerator
            lambda: the regularization
        returns:
            coefficients
        '''
        H = (1. - alpha) * (jnp.dot(phi_y.T, phi_y) / len(phi_y)) + alpha * (jnp.dot(phi_x.T, phi_x) / len(phi_x)) 
        h = phi_x.mean(axis=0, keepdims=True).T 
        b = len(H)
        P = (H + lamda * np.eye(b) )
        q = -h
        G = -np.eye(b)
        h_ = np.zeros((b,1))
        qp = OSQP()
        constr = phi_y.mean(axis=0, keepdims=True)
        sol = qp.run(params_obj=(P, q), 
                    #  params_ineq=(G, h_),
                     params_eq=(constr, jnp.array([[1.]]))
                     ).params

        # sol = solvers.qp(P, q, G, h_, options={'show_progress': False}) ## turn off logging
        # coef = np.array(sol['x'])
        coef = jnp.atleast_2d(sol.primal)

        # matvec = lambda x: jnp.dot(P, x)

        # coef = solve_cg(matvec, b=-q)
        
        return coef

    # def kfold_tuning(self, x_nu, x_de, centers,list_params, len_params=100):
    #     '''
    #     Function to do kfold for selecting the optimal hyperparameters
    #     '''
        
    #     n_nu = x_nu.shape[0]
    #     n_de = x_de.shape[0]
    #     key = random.PRNGKey(123)
    #     # score_cv = jnp.zeros((len(grid_kernel_params), len(grid_algo_params)))
    #     cv_index_nu = random.permutation(key, n_nu)
    #     cv_index_de = random.permutation(key, jnp.arange(n_nu, n_nu + n_de))
    #     x = jnp.vstack((x_nu, x_de))
    #     fold_size_nu = int(jnp.ceil(n_nu / self.k_fold))
    #     fold_size_de = int(jnp.ceil(n_de / self.k_fold))
    #     ## product grid_kernel_params and grid_algo_params
    #     ## using itertools
    #     def params_loop(param_carry, idx):
    #         par = param_carry
    #         sigma, lambda_ = par[0].slice_at_index(idx) , par[1].slice_at_index(idx)
    #         # unflatten parameters
    #         # sigma = tree_unflatten(self.kernel_def, sigma)
    #         # lambda_ = tree_unflatten(self.algo_def, lambda_)
    #         # compute score
    #         # K =  create_covariance_v2(name=sigma.kernel_name, x=x, y=centers, params=sigma) @ jnp.eye(len(centers)) 
    #         K =  self.covar_func(x=x, y=centers, params=sigma)
    #         def cv_loop(carry, params):
    #             # sigma, lambda_ = params
    #             cv_index = carry 
                
    #             index_te_nu = lax.dynamic_slice(cv_index_nu, (cv_index * fold_size_nu,),(fold_size_nu,))
    #             index_tr_nu = jnp.setdiff1d(cv_index_nu, index_te_nu,size=n_nu - fold_size_nu)

    #             index_te_de = lax.dynamic_slice(cv_index_de, (cv_index * fold_size_de,),(fold_size_de,))
    #             index_tr_de = jnp.setdiff1d(cv_index_de, index_te_de,size=n_de - fold_size_de)

    #             Ktmp = K[jnp.hstack((index_tr_nu, index_tr_de)), :]

    #             phi_x = Ktmp[:len(index_tr_nu), :]
    #             phi_y = Ktmp[len(index_tr_nu):, :]

    #             coef = self.solve_qp(phi_x, phi_y, **lambda_.hyperparams)

    #             Ktmp_ = K[jnp.hstack((index_te_nu, index_te_de)), :]
    #             phi_x_1 = Ktmp_[:len(index_te_nu), :]
    #             phi_y_1 = Ktmp_[len(index_te_nu):, :]

    #             score = self.compute_score(coef, phi_x_1, phi_y_1, **lambda_.hyperparams)

    #             return cv_index +1 , score
    #         ## first, run cv
    #         _, cv_score = lax.scan(cv_loop, 0, jnp.arange(self.k_fold))
    #         return par, jnp.nanmean(cv_score)
    #     ## second, run outer loop
    #     _, final_score = lax.scan(params_loop, list_params, jnp.arange(len_params))
    #     ## find min score
    #     score_cv_min_index = jnp.nanargmin(final_score)
    #     sigma_chosen, lambda_chosen = list_params[0].slice_at_index(score_cv_min_index), list_params[1].slice_at_index(score_cv_min_index)
    #     print('Finish K fold')
    #     return sigma_chosen, lambda_chosen



    def kfold_tuning(self, x_nu, x_de, centers, list_params, len_params=100):
        """
        K-Fold tuning using Python loops (tuning both sigma and lambda), low memory version.
        """

        n_nu = x_nu.shape[0]
        n_de = x_de.shape[0]
        key = jax.random.PRNGKey(123)

        cv_index_nu = jax.random.permutation(key, n_nu)
        cv_index_de = jax.random.permutation(key, jnp.arange(n_nu, n_nu + n_de))
        x = jnp.vstack((x_nu, x_de))

        fold_size_nu = int(jnp.ceil(n_nu / self.k_fold))
        fold_size_de = int(jnp.ceil(n_de / self.k_fold))

        list_sigma, list_lambda = list_params  # Unpack sigma and lambda lists

        all_cv_scores = []

        for idx in tqdm(range(len_params), desc="Hyperparam tuning"):
            sigma = list_sigma[idx]
            lambda_ = list_lambda[idx]

            fold_scores = []

            for fold_idx in range(self.k_fold):
                # Slice indices
                start_nu = fold_idx * fold_size_nu
                end_nu = jnp.minimum(start_nu + fold_size_nu, n_nu)

                start_de = fold_idx * fold_size_de
                end_de = jnp.minimum(start_de + fold_size_de, n_de)

                index_te_nu = cv_index_nu[start_nu:end_nu]
                index_tr_nu = jnp.setdiff1d(cv_index_nu, index_te_nu, size=n_nu - (end_nu - start_nu))

                index_te_de = cv_index_de[start_de:end_de]
                index_tr_de = jnp.setdiff1d(cv_index_de, index_te_de, size=n_de - (end_de - start_de))

                # Training data
                idx_train = jnp.hstack((index_tr_nu, index_tr_de))
                x_train = x[idx_train]

                # Validation data
                idx_val = jnp.hstack((index_te_nu, index_te_de))
                x_val = x[idx_val]

                # Compute K slices ON THE FLY
                K_train = self.covar_func(x=x_train, y=centers, params=sigma)
                phi_x = K_train[:len(index_tr_nu), :]
                phi_y = K_train[len(index_tr_nu):, :]

                coef = self.solve_qp(phi_x, phi_y, **lambda_.hyperparams)

                K_val = self.covar_func(x=x_val, y=centers, params=sigma)
                phi_x_1 = K_val[:len(index_te_nu), :]
                phi_y_1 = K_val[len(index_te_nu):, :]

                score = self.compute_score(coef, phi_x_1, phi_y_1, **lambda_.hyperparams)

                fold_scores.append(score)

            fold_scores = jnp.stack(fold_scores)
            avg_cv_score = jnp.nanmean(fold_scores)
            all_cv_scores.append(avg_cv_score)

        all_cv_scores = jnp.stack(all_cv_scores)

        best_idx = jnp.nanargmin(all_cv_scores)
        best_idx = int(best_idx)  # safely move to Python int

        sigma_chosen = list_sigma[best_idx]
        lambda_chosen = list_lambda[best_idx]

        print('Finish K-fold tuning.')

        return sigma_chosen, lambda_chosen

    
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

        # p_dist = pairwise_distances(x_train, x_test).flatten()
        # sigma_min = jnp.percentile(p_dist, 10)  # 10th percentile
        # sigma_max = jnp.percentile(p_dist, 90)  # 90th percentile
        sigma_min, sigma_max = estimate_sigma(x_train,x_test, n_samples=10_000)

        sigma_range = create_default_range_params(self.kernel_params.kernel_name, {'length_scale': (sigma_min, sigma_max)})
        sigma_list = generate_param_ranges_jax(self.kernel_params.hyperparams, sigma_range, num_values=20)
        sigma_list = [Params(kernel_name=self.kernel_params.kernel_name, hyperparams=p) for p in sigma_list]
        # sigma_list = jnp.logspace(jnp.log10(sigma_min), jnp.log10(sigma_max), num=10)

        lambda_min = self.lam
        lambda_max = 1.
        lambda_range = create_default_range_params(self.algo_params.kernel_name, {'lamda': (lambda_min, lambda_max)})
        lambda_list = generate_param_ranges_jax(self.algo_params.hyperparams, lambda_range, num_values=4)
        lambda_list = [Params(kernel_name=self.algo_params.kernel_name, hyperparams=p) for p in lambda_list]
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
        list_params = list(product(sigma_list, lambda_list))
        len_params = len(list_params)
        list_kernel_params,list_algo_params = ([i[0] for i in list_params], [i[1] for i in list_params])
        # batched_kernel_params = batch_params(list_kernel_params)
        # batched_algo_params = batch_params(list_algo_params)
        # list_params = ([i[0] for i in list_params], [i[1] for i in list_params])
        ## flatten params
        
        # list_flatten_kernels = [[tree_flatten(i[0])[0]] for i in list_params]
        # list_flatten_algos = [[tree_flatten(i[1])[0]] for i in list_params]
        # list_params = (np.concatenate(list_flatten_kernels), np.concatenate(list_flatten_algos))
        # pdb.set_trace()
        # sigma_chosen,lambda_chosen = self.kfold_tuning(x_test, x_train, self.__centers, (batched_kernel_params,batched_algo_params ),len_params=len_params )
        sigma_chosen,lambda_chosen = self.kfold_tuning(x_test, x_train, self.__centers, (list_kernel_params,list_algo_params ),len_params=len_params )

        # phi_numerator = create_covariance_v2(name=self.basis_func, x=x_test, y=self.__centers, params=sigma_chosen) @ jnp.eye(len(self.__centers)) 
        # phi_denominator = create_covariance_v2(name=self.basis_func, x=x_train, y=self.__centers, params=sigma_chosen) @ jnp.eye(len(self.__centers))
        phi_numerator = self.covar_func(x=x_test, y=self.__centers, params=sigma_chosen) 
        phi_denominator = self.covar_func(x=x_train, y=self.__centers, params=sigma_chosen)
        self.__coef = jnp.maximum(self.solve_qp(phi_numerator,phi_denominator,**lambda_chosen.hyperparams),0.)

        ## reassign hyperparameters
        self.kernel_params = sigma_chosen
        self.algo_params = lambda_chosen

        # # Choose centers by simulated annuealing
        # minbounds = jnp.min(jnp.concatenate((x_train, x_test),axis=0),axis=0,keepdims=True)
        # maxbounds = jnp.max(jnp.concatenate((x_train, x_test),axis=0),axis=0, keepdims=True)
        # ## tile to number of points
        # minbounds = jnp.tile(minbounds,(self.num_basis, 1))
        # maxbounds = jnp.tile(maxbounds,(self.num_basis, 1))

        # bounds = (minbounds, maxbounds)
        # def obj_func(chosen_centers):
        #     phi_x = create_covariance_v2(self.basis_func,x=x_test, y=chosen_centers, params = self.kernel_params) @ jnp.eye(len(chosen_centers))
        #     phi_y = create_covariance_v2(self.basis_func,x=x_train, y=chosen_centers, params = self.kernel_params) @ jnp.eye(len(chosen_centers))
        #     score = self.compute_score(self.__coef, phi_x, phi_y, **self.algo_params.hyperparams)
        #     return score[0]

        # sa = SimulatedAnnealingJAX(
        #                         obj_func,
        #                         bounds,
        #                         adaptive_perturb,
        #                         max_iter=3000,
        #                         initial_temp=100.0,
        #                         alpha=0.99,
        #                         temp_min=1e-3
        #                     )

        # key = jax.random.PRNGKey(42)
        # best_x, best_val, positions, values = sa.optimize(key)
        # self.__centers = best_x

        score = self.compute_score(self.__coef,phi_numerator,phi_denominator,**lambda_chosen.hyperparams)
        print(f'Finished training, the minimized function value is: lambda {lambda_chosen}, value: {score}')
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

class PearsonDensityratio(uLSIF):
    '''
    The weights will compute by the Pearson method
    '''
    def __init__(self, alpha=0.1,num_basis=500, k_fold=5, basis_func='squared_exponential',**kwargs):
        super().__init__(alpha=alpha,num_basis=num_basis, k_fold=k_fold, basis_func=basis_func,**kwargs)

    def compute_score(self, coef,phi_x,phi_y, alpha, lamda ):
        '''
        Helper fucntion to compute the score to evaluate learned parameters
        phi_x: numerator after projecting on the Hilbert space
        phi_y : denominator after projecting on the Hilbert space
        '''

        # H = (1. - alpha) * (jnp.dot(phi_y.T, phi_y) / len(phi_y)) + alpha * (jnp.dot(phi_x.T, phi_x) / len(phi_x)) 
        h1 = phi_x.mean(axis=0, keepdims=True).T 
        h2 = phi_y.mean(axis=0, keepdims=True).T
        score = -(1 -  alpha) * h2.T @ coef - alpha * h1.T @ coef
        return score.ravel()
    
    def solve_qp(self, phi_x, phi_y, alpha, lamda):
        '''
        Args:
            phi_x: numerator after projecting on the Hilbert space
            phi_y: denominator after projecting on the Hilbert space

            alpha: weights to merge denominator with numerator
            lambda: the regularization
        returns:
            coefficients
        '''
        # H = (1. - alpha) * (jnp.dot(phi_y.T, phi_y) / len(phi_y)) + alpha * (jnp.dot(phi_x.T, phi_x) / len(phi_x)) 
        # h = phi_x.mean(axis=0, keepdims=True).T 
        # b = len(H)
        P = 0 * np.eye(phi_x.shape[-1])
        h1 = phi_x.mean(axis=0, keepdims=True).T 
        h2 = phi_y.mean(axis=0, keepdims=True).T
        q = -(1 -  alpha) * h2  - alpha * h1
        G = -np.eye(phi_x.shape[-1])
        h_ = np.zeros((phi_x.shape[-1],1))
        qp = OSQP()
        constr = phi_y.mean(axis=0, keepdims=True)
        sol = qp.run(params_obj=(P, q), params_ineq=(G, h_),
                     params_eq=(constr, jnp.array([[1.]]))
                     ).params

        # sol = solvers.qp(P, q, G, h_, options={'show_progress': False}) ## turn off logging
        # coef = np.array(sol['x'])
        coef = jnp.atleast_2d(sol.primal)
        
        return coef

    
        

    
from jax import  value_and_grad
import optax
from covariance.params import Params
from jaxopt._src import tree_util

class GraduLSIF(DensityRatioEstimator):

    def __init__(self, alpha=0.1,num_basis=500, k_fold=5, basis_func='squared_exponential',_lambda=0.2,**kwargs):
        
        self.alpha = alpha
        self.num_basis = num_basis
        self.k_fold = k_fold
        self.fitting = False
        
        self.kernel_params = create_params(basis_func)
        self.basis_func = basis_func
        self._lambda = _lambda
        self.fitting = False

    
    
    def _choose_centers(self, x_test, n_centers,algo='kmeans',anchor_min=None,anchor_max=None):
        """
        Strategy to choose centers
        """

        return select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=anchor_min,anchor_max=anchor_max)
    
    
    
    def loss_function(self, params: Params,coef, x_nu, x_de, centers, alpha, lamda):
        """
        Loss function for gradient-based hyperparameter tuning.

        Args:
            params: Params object containing kernel-specific hyperparameters.
            coef : (jnp.ndarray): Trainable coefficients
            x_nu: Testing data (numerator).
            x_de: Training data (denominator).
            centers: chosen points to calcualte kernels
            _lambda: the coef for regularization

        Returns:
            Scalar loss value.
        """
       
        
        # Create kernel matrices for numerator and denominator
        K_nu = create_covariance_v2(name=params.kernel_name, x=x_nu, y=centers, params=params) @ jnp.eye(len(centers))
        K_de = create_covariance_v2(name=params.kernel_name, x=x_de, y=centers, params=params) @ jnp.eye(len(centers))

        # Compute QP matrices
        H = (alpha * (jnp.dot(K_nu.T, K_nu) / K_nu.shape[0]) +
            (1 - alpha) * (jnp.dot(K_de.T, K_de) / K_de.shape[0]))

        h = K_nu.mean(axis=0, keepdims=True).T  # Ensure correct shape (features, 1)
        b = H.shape[0]
        P = (H + lamda * jnp.eye(b))  # Regularization
        constr = jnp.mean(K_de,axis=0,keepdims=True).T
        # Define the loss function based on the coefficient parameter
        coef = projection_non_negative_and_hyperplane(coef,hyperparams=(constr, 1.))
        ## have to consider constraint for the params
        params = tree_util.tree_map(jax.nn.relu, params)

        loss = 0.5 * jnp.dot(coef.T, jnp.dot(P, coef)) - jnp.dot(h.T, coef)
        # loss += (jnp.mean(jnp.abs(1/K_nu @ coef -1)) - jnp.mean(jnp.abs(K_de @ coef -1)))**2  ## Total variation regularization
        
        return loss.squeeze()
    
    def tune_hyperparameters(self, x_nu, x_de, centers, init_params: Params,
                            learning_rate=0.01, num_steps=100, batch_size=64, seed=42):
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
        elif self.basis_func =='mixture_linear':
            hyperparams['weights'] = jnp.ones((5,))
            hyperparams['biases'] = jnp.ones((5, ))
            

        params = Params(init_params.kernel_name, hyperparams)
        init_coef = jnp.ones((len(centers),1))
        
        # Define optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init((params, init_coef,centers))

        # Define batched loss function using vmap

        loss_and_grad_fn = value_and_grad(self.loss_function, argnums=(0, 1, 4))

        # Function to process a single batch
        def step_fn(carry, batch):
            params, coef, ce, opt_state = carry
            x_nu_batch, x_de_batch = batch[0], batch[1]
            ## 
            # loss_value, (grad_params, grad_coef) = loss_and_grad_fn(params, x_nu_batch[0], x_de_batch[0])
            (loss_value, (grad_params, grad_coef,grad_ce)) = loss_and_grad_fn(params, coef, x_nu_batch[0], x_de_batch[0], ce, self.alpha, self._lambda)
        
            updates, opt_state = optimizer.update((grad_params, grad_coef,grad_ce), opt_state,(params, coef,ce))
            new_params, new_coef, new_ce = optax.apply_updates((params, coef, ce), updates)

            return (new_params, new_coef,new_ce, opt_state), loss_value
           

        n_steps = tqdm(range(num_steps), desc="Training Progress")
        min_length = min(len(x_nu), len(x_de))
        for step in n_steps:
            # Shuffle data at the start of each epoch
            key, subkey = random.split(key)
            shuffled_indices_nu = random.permutation(subkey, len(x_nu))
            shuffled_indices_de = random.permutation(subkey, len(x_de))
            x_nu = x_nu[shuffled_indices_nu]
            x_de = x_de[shuffled_indices_de]
            # Calculate the number of batches
            num_batches = int(np.ceil(min_length / batch_size))

            # Use np.array_split to divide arrays into approximately equal batches
            x_nu_batches = jnp.array_split(x_nu, num_batches)
            x_de_batches = jnp.array_split(x_de, num_batches)
            # check the first batch
            x_nu_batches[0] = x_nu_batches[0][:len(x_nu_batches[-1])]
            x_de_batches[0] = x_de_batches[0][:len(x_de_batches[-1])]

            ## increase dimension
            x_nu_batches = [i[jnp.newaxis,...] for i in x_nu_batches]
            x_de_batches = [i[jnp.newaxis,...] for i in x_de_batches]
            

            # Pair corresponding batches from both datasets
            batches = list(zip(x_nu_batches, x_de_batches))

            # # Prepare batches
            # x_nu_batches = [x_nu[i: i +batch_size][jnp.newaxis,...] for i in range(0, len(x_nu), batch_size)]
            # ## truncated the batch size
            # if len(x_nu_batches[-1][0]) < batch_size:
            #     pad_size = batch_size - x_nu_batches[-1].shape[1]
            #     x_nu_batches[-1] = jnp.pad(x_nu_batches[-1], ((0, 0), (0, pad_size), (0, 0)), mode='reflect')
            # # x_nu_batches[0] = x_nu_batches[0][:batch_size]
            # x_de_batches = x_de[jnp.newaxis,...]
            
            # batches = list(zip(x_nu_batches, [x_de_batches] * len(x_nu_batches)))
            # pdb.set_trace()

            # Scan over the mini-batches
            (params,init_coef,centers, opt_state), losses = lax.scan(step_fn, (params,init_coef,centers, opt_state), batches)

            # Calculate the average loss
            avg_loss = jnp.mean(losses)
            n_steps.set_postfix_str(f"Step {step}/{num_steps} - Avg Loss: {avg_loss:.6f}")

        return (params,init_coef,centers)

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
        archor_min, archor_max, valid = find_intersection_anchor(x_train,x_test)
        if not valid:
            archor_min = None 
            archor_max = None

        self.__centers = self._choose_centers(x_test, n_centers=self.num_basis, algo='kmeans',anchor_min=archor_min, anchor_max=archor_max)

        # Tune hyperparameters dynamically
        tuned_params = self.tune_hyperparameters(x_test, x_train, self.__centers, self.kernel_params,
                                                 learning_rate=learning_rate,num_steps=num_steps,
                                                 batch_size=batch_size)
        ## reassign hyperparameters
        self.kernel_params = tuned_params[0]
        self.__coef = jnp.maximum(tuned_params[1],0.)
        self.__centers = tuned_params[2]


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

class RandomProjection(DensityRatioEstimator):
    '''
    The weights will compute by therandom projection method
    
    '''

    def __init__(self, num_projection:int=100,method='KLIEP'):
        self.num_projection = num_projection
        if method == 'uLSIF':
            self.method = uLSIF(alpha=0.3,num_basis=50,k_fold=5,basis_func='radial')
        elif method == 'KLIEP':
            self.method = KLIEP(num_basis=50,basis_func='radial', learning_rate=1e-7,a_tol=1e-6,cv=5)
        pass

    def fit(self, x_train,x_test):
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
        n_train = len(x_train)
        n_test = len(x_test)
        n_total = n_train + n_test
        n_features = x_train.shape[1]
        self.__projection_matrix = np.random.randn(n_features,self.num_projection)
        self.__projection_matrix /= np.linalg.norm(self.__projection_matrix,axis=0)
        ## weights will minimize the KL divergence between two distributions after projection
        ## using the random projection method
        ## first project x train and x test
        x_total = np.vstack((x_train,x_test))
        x_total = x_total @ self.__projection_matrix
        x_train = x_total[:n_train]
        x_test = x_total[n_train:]
        ## compute the weights using defined method
        self.method.fit(x_train,x_test)        
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
        x_train = x_train @ self.__projection_matrix
        return self.method.compute_weights(x_train)
    
    def kfold_tuning(self, *args, **kwargs):
        pass 


import pdb
from scipy.stats import norm
def experiment():
    x_nu = 10 + 2*np.random.normal(size=(1000,1))
    x_de = 3 + 1 * np.random.normal(size=(4000,1))
    mean_ = x_de.mean()
    std_ = x_de.std()
    x_nu = (x_nu - mean_)/std_
    x_de = (x_de - mean_)/std_

    # sigma_list = np.arange(0.1,0.5,0.1)
    # lambda_list = np.arange(0.1,0.5,0.1)
    alpha = 0.4
    b = 200
    model = uLSIF(alpha=alpha,num_basis=b,k_fold=5,basis_func='radial')
    # model = KLIEP(num_basis=20,basis_func='radial', learning_rate=1e-7,a_tol=1e-6,cv=5)
    model.fit(x_de,x_nu)

    # coef_chosen,sigma_chosen,centers = uLSIF(x_nu, x_de, alpha, sigma_list, lambda_list, b, fold=5)
    # pdb.set_trace()
    
    fig,ax = plt.subplots()
    x = np.linspace(-2,2,10000)
    # phi_de =np.exp(-cdist(x.reshape(-1,1), centers, 'sqeuclidean') / (2 * sigma_chosen ** 2))
    # estimate_weights = phi_de @ coef_chosen
    estimate_weights = model.compute_weights(x.reshape(-1,1))

    model = uLSIF(alpha=alpha,num_basis=b,k_fold=5,basis_func='radial')
    model.fit(x_de,x_nu)
    estimate_weights_2 = model.compute_weights(x.reshape(-1,1))
    ax.plot(x,norm.pdf(x, loc=10, scale=2),label='numerator')
    ax.plot(x,norm.pdf(x, loc=3, scale=1),label='denominator')
    ax.plot(x,norm.pdf(x, loc=10, scale=2)/norm.pdf(x, loc=3, scale=1),label='weight')
    ax.plot(x,estimate_weights.flatten(),label='KLIEP weight')
    ax.plot(x,estimate_weights_2.flatten(),label='uLSIF weight')
    fig.legend()
    fig.savefig('covariateshift_code/fig_KLIEP_2.png')
    # fig.savefig('covariateshift_code/fig.png')

# experiment()

