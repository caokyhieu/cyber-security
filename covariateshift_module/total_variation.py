import numpy as np
import random
from tqdm import tqdm
from covariateshift_module.utils import  projection_non_negative_and_hyperplane
import warnings

from tqdm import tqdm
from covariance.covariance_matrix import  create_params,create_covariance_v2
from covariance.params import Params
from jax import random
import jax.numpy as jnp
import jax
from covariateshift_module.utils import DensityRatioEstimator, select_centers
from jax import  value_and_grad
import optax
import pdb
# Set warnings to be treated as errors
warnings.filterwarnings("error")
    
from jaxopt import OSQP
from jax import lax
from covariateshift_module.utils import find_intersection_anchor
from jaxopt._src import tree_util


class TotalVariationEstimator(DensityRatioEstimator):
    # """
    # This estimator will use Total Variation to estimate the density ratio.
    # TV(p, q) = 1/2\int_{\mathcal{X}} |p(x) - q(x)| dx = 1/2\int_{\mathcal{X}} |q(x)/p(x) - 1| p(x) dx
    #          = 1/2\int_{\mathcal{X}} |r(x) - 1| p(x) dx = 1/2\int_{\mathcal{X}} |1/r(x) - 1| q(x) dx
    #          = 1/4 E_{p(x)}[|r(x) - 1|] + 1/4 E_{q(x)}[|1/r(x) - 1|] 
    #          with r(x) = q(x)/p(x) p(x) = w(x) p(x)
    # """
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
       
        ## have to consider constraint for the params
        params = tree_util.tree_map(jax.nn.relu, params)
        # Create kernel matrices for numerator and denominator
        K_nu = create_covariance_v2(name=params.kernel_name, x=x_nu, y=centers, params=params) @ jnp.eye(len(centers))
        K_de = create_covariance_v2(name=params.kernel_name, x=x_de, y=centers, params=params) @ jnp.eye(len(centers))


        constr = jnp.mean(K_de,axis=0,keepdims=True).T
        # Define the loss function based on the coefficient parameter
        coef = projection_non_negative_and_hyperplane(coef,hyperparams=(constr, 1.))
       

        ## compute the loss
        loss = (jnp.mean(jnp.abs(1/(K_nu @ coef) -1)) - jnp.mean(jnp.abs(K_de @ coef -1)))**2
        
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
        opt_state = optimizer.init((params, init_coef))

        # Define batched loss function using vmap

        loss_and_grad_fn = value_and_grad(self.loss_function, argnums=(0, 1))

        # Function to process a single batch
        def step_fn(carry, batch):
            params, coef, opt_state = carry
            x_nu_batch, x_de_batch = batch[0], batch[1]
            ## 
            # loss_value, (grad_params, grad_coef) = loss_and_grad_fn(params, x_nu_batch[0], x_de_batch[0])
            (loss_value, (grad_params, grad_coef)) = loss_and_grad_fn(params, coef, x_nu_batch[0], x_de_batch[0], centers, self.alpha, self._lambda)
        
            updates, opt_state = optimizer.update((grad_params, grad_coef), opt_state,(params, coef))
            new_params, new_coef = optax.apply_updates((params, coef), updates)

            return (new_params, new_coef, opt_state), loss_value
           

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
        # self.__centers = tuned_params[2]


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