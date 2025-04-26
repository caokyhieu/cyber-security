
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training.train_state import TrainState
from optax import adam
from covariance.covariance_matrix import create_covariance
import jax.numpy as jnp
from jax import jit, vmap,lax,grad
from scipy.spatial.distance import cdist
from covariateshift_module.utils import DensityRatioEstimator
import jax
import optax
from tqdm import tqdm
from covariateshift_module.utils import  select_centers, pairwise_distances, estimate_sigma
from covariateshift_module.utils import find_intersection_anchor, projection_non_negative_and_hyperplane
from covariance.params import Params, generate_param_ranges_jax, create_default_range_params, batch_params
from covariance.covariance_matrix import create_params,create_covariance_v2



def cdist_jax(XA, XB):
    return vmap(lambda x: jnp.sqrt(jnp.sum((XB - x)**2, axis=-1)))(XA)

# @jit
def sliced_wasserstein_distance(samples_p, samples_q, weights_p, num_projections=10):
    # Initialize the random number generator
    keys = random.split(random.PRNGKey(0), num=num_projections)
    keys_gumbel = random.split(random.PRNGKey(2), num=samples_q.shape[0])

    # Initialize the SWD
    swd = 0

    def body_fun(i, swd):
        # Generate a random direction
        direction = random.normal(keys[i], (samples_p.shape[1],))
        direction = direction / jnp.linalg.norm(direction)

        # Project the samples onto the direction
        projections_p = jnp.dot(samples_p, direction)
        projections_q = jnp.dot(samples_q, direction)

        # Vectorize the draw_samples function
        draw_samples_vmap = vmap(draw_samples, in_axes=(None, None, None, 0))

        # Use vmap to replace the inner loop
        projections_p = draw_samples_vmap(projections_p, weights_p, 0.1, keys_gumbel)
        projections_p = projections_p.reshape(-1, 1)

        # Sort the projections
        projections_p_sorted = jnp.sort(projections_p)
        projections_q_sorted = jnp.sort(projections_q)

        # Add the 1D Wasserstein distance between the projections to the SWD
        swd += jnp.mean(jnp.abs(projections_p_sorted - projections_q_sorted))

        return swd

    swd = lax.fori_loop(0, num_projections, body_fun, swd)

    # Average the SWD over the number of projections
    swd /= num_projections

    return swd

from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud
from ott.solvers.linear import implicit_differentiation as imp_diff

def sinkhorn_loss_v2(x, y, weight_x, weight_y, implicit: bool = True):
    
    return sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x,
        y,  # this part defines geometry
        a=weight_x,
        b=weight_y,  # this sets weights
        sinkhorn_kwargs={
            "implicit_diff": imp_diff.ImplicitDiff() if implicit else None,
            "use_danskin": False,
        },
    ).divergence 
def sinkhorn_loss(x, y, weights_x, reg=1., num_iters=100):
    # Calculate the pairwise distances
    C = cdist_jax(x, y)

    # Initialize the Sinkhorn algorithm
    a = weights_x / jnp.sum(weights_x)
    b = jnp.ones(y.shape[0]) / y.shape[0]
    K = jnp.exp(-C / reg)

    # Sinkhorn iterations
    for _ in range(num_iters):
        a = jnp.divide(1.0, jnp.dot(K, b))
        b = jnp.divide(1.0, jnp.dot(K.T, a))

    # Calculate the Sinkhorn loss
    P = jnp.outer(a, b) * K
    loss = jnp.sum(P * C)

    return loss


class TrainableParamNetwork(nn.Module):
    """A network that returns trainable parameters of shape (len(x), 1)."""
    data_size: int  # Fixed size of x

    def setup(self):
        """Initialize trainable parameters with shape (data_size, 1)."""
        self.params = self.param("trainable_params", nn.initializers.lecun_normal(), (self.data_size, 1))

    def __call__(self, x):
        """Return the trainable parameters directly."""
        return jax.nn.relu(self.params) +1e-4 # Shape (data_size, 1)

class WeightNetwork(nn.Module):
    layers_config: list

    def setup(self):
        self.layers = []
        self.residuals = []
        for nodes in self.layers_config:
            self.layers+= (nn.Dense(nodes),nn.relu,nn.Dense(nodes),nn.relu)
            self.residuals+= (nn.Dense(nodes),)
         
        self.layers+= (nn.Dense(1),)

    def __call__(self, x):
        for i,res in enumerate(self.residuals):
            residual = res(x)
            x = self.layers[i*4](x) 
            x = self.layers[i*4+1](x)
            x = self.layers[i*4+2](x) 
            x = self.layers[i*4+3](x) + residual
        # for i,layer in enumerate(self.layers):
        #     residual = x
        #     x = layer(x)
        x = self.layers[-1](x)
        return jnp.exp(x)

def total_variation_distance(samples_p, samples_q, weights_p, num_bins=30):
    # Calculate the total weight for normalization
    total_weight_p = jnp.sum(weights_p)

    # Initialize the TV distance
    tv_distance = 0

    # Loop over each dimension
    for dim in range(samples_p.shape[1]):
        # Extract the samples for this dimension
        samples_p_dim = samples_p[:, dim]
        samples_q_dim = samples_q[:, dim]

        # Define the edges of the bins
        bin_edges = jnp.linspace(jnp.min(jnp.concatenate([samples_p_dim, samples_q_dim])), 
                                 jnp.max(jnp.concatenate([samples_p_dim, samples_q_dim])), 
                                 num_bins+1)

        # Estimate the weighted PMF of p by creating a histogram
        hist_p, _ = jnp.histogram(samples_p_dim, bins=bin_edges, weights=weights_p/total_weight_p)

        # Estimate the PMF of q by creating a histogram
        hist_q, _ = jnp.histogram(samples_q_dim, bins=bin_edges, density=True)

        # Add the TV distance for this dimension to the total TV distance
        tv_distance += 0.5 * jnp.sum(jnp.abs(hist_p - hist_q))

    # Average the TV distance over the dimensions
    tv_distance /= samples_p.shape[1]

    return tv_distance
@jit
def weighted_MMD(samples_p, samples_q, weights_p, length_scale:list=[0.1,0.2,0.3]):
    def body_fun(i, mmd):
        scale = length_scale[i]
        # Compute the Gram matrices
        K_pp = create_covariance('squared_exponential_kernel',samples_p, samples_p, scale) @ jnp.eye(len(samples_p))
        K_pq = create_covariance('squared_exponential_kernel',samples_p, samples_q, scale) @ jnp.eye(len(samples_q))

        # Compute the weighted means of the Gram matrices
        mean_K_pp = weights_p.T @ K_pp @ weights_p / len(samples_p)**2
        mean_K_pq = jnp.sum(weights_p.T @ K_pq)  / (len(samples_p) * len(samples_q))

        # Compute the weighted MMD
        mmd = mean_K_pp - 2 * mean_K_pq

        return mmd.squeeze()

    mmd_values = lax.fori_loop(0, len(length_scale), body_fun, 0.0)

    return mmd_values

def update_fn(state, model, x_train, x_test,num_projections=1000):  # accept model as an argument
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params, model, x_train, x_test,num_projections=num_projections)  # pass model to loss_fn
    return state.apply_gradients(grads=grad), loss

def loss_fn(params, model, x_train, x_test,num_projections=10000):  # accept model as an argument
    weight_p = model.apply(params, x_train)  # use model.apply instead of calling model
    # q = model.apply(params, x_test)  # use model.apply instead of calling model
    # return weighted_MMD(x_train, x_test, weight_p)  + jnp.mean((q-1.)**2)
    # return sinkhorn_loss(x_train, x_test, weight_p.flatten(), reg=1., num_iters=100)
    # return sliced_wasserstein_distance(x_train, x_test, weight_p.flatten(),num_projections=num_projections)
    
    return sinkhorn_loss_v2(x_train, x_test, weight_p.flatten()/weight_p.sum(), 
                            jnp.ones(x_test.shape[0])/len(x_test), implicit=False)
def train(model, x_train, x_test, learning_rate=0.01, num_epochs=100,num_projections=100):
    params = model.init(jax.random.PRNGKey(0), x_train)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=adam(learning_rate))

    for epoch in range(num_epochs):
        state, loss = update_fn(state, model, x_train, x_test,num_projections=num_projections)  # pass model as an argument
        print(f"Epoch {epoch+1}, Loss: {loss}")

    return state.params

def sample_gumbel(shape, key):
    """Sample from Gumbel(0, 1) distribution."""
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U))

def gumbel_softmax(logits, tau=1.0, key=None):
    """Sample from the Gumbel-Softmax distribution."""
    gumbel_noise = sample_gumbel(logits.shape, key)
    y = logits + gumbel_noise
    return jax.nn.softmax(y / tau)

def draw_samples(samples, weights, tau=1.0, key=None):
    """Draw samples from a given set of samples using the provided weights."""
    # Convert weights to logits
    logits = jnp.log(weights)

    # Get the Gumbel-Softmax sample
    softmax_sample = gumbel_softmax(logits, tau, key)
    
    # Use the softmax sample to draw from the original samples
    sampled_value = jnp.dot(softmax_sample, samples)
    return sampled_value


class WassersteinRatio(DensityRatioEstimator):

    def __init__(self, num_projection:int=60,layers_config:list=[128,64,32],
                 learning_rate:float=1e-3,num_epochs:int=100 ):
        self.num_projection = num_projection
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.layers_config = layers_config
        # self.model = WeightNetwork(layers_config=self.layers_config)
        self.model = TrainableParamNetwork(data_size=layers_config[0]) ## now first layer size is the size of  data array
        self.fitting = False
        self.params = None
        pass
    def _choose_centers(self, x_test, n_centers,algo='kmeans',archor_min=None, archor_max=None):
        """
        Strategy to choose centers
        """

        points = select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=archor_min, anchor_max=archor_max)
        ## set center to true
        self.set_centers = True
        return points

    def fit(self, x_denom, x_num):
        self.params = train(self.model, x_denom, x_num, num_epochs=self.num_epochs,learning_rate=self.learning_rate,num_projections=self.num_projection)
        self.fitting = True

        pass

    def compute_weights(self, *args, **kwargs):
        if not self.fitting:
            raise ValueError('Model is not fitted yet')
        return self.model.apply(self.params, *args, **kwargs)
    
    def kfold_tuning(self, *args, **kwargs):
        pass

from tqdm import tqdm
from ott.geometry.costs import PNormP,Euclidean

class WassersteinRatiov2(DensityRatioEstimator):

    def __init__(self, num_projection:int=60,layers_config:list=[128,64,32],
                 learning_rate:float=1e-3,num_epochs:int=100,k_fold=5, **kwargs ):
        
        self.num_basis = num_projection
        self.basis_func = 'squared_exponential'
        self.kernel_params = create_params(self.basis_func)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.k_fold = k_fold
        self.covar_func = lambda x, y, params: create_covariance_v2(name=self.basis_func, x=x, y=y, params=params) @ jnp.eye(len(y))
        self.batchsize= kwargs.get('batch_size', 128)

        self.fitting = False
        pass

    def loss_func(self,theta, x, y, phi_x, weight_y):
        
        weight_x = jax.nn.relu(phi_x @ theta ) + 1e-3
        weight_x = weight_x.flatten()/jnp.sum(weight_x)
        weight_y = weight_y.flatten()/ jnp.sum(weight_y)
        d = sinkhorn_divergence.sinkhorn_divergence(
                                            pointcloud.PointCloud,
                                            x,
                                            y,  # this part defines geometry
                                            a=weight_x,
                                            b=weight_y,  # this sets weights
                                            sinkhorn_kwargs={
                                                "implicit_diff": imp_diff.ImplicitDiff(),
                                                "use_danskin": True,
                                            },
                                            # cost_fn= PNormP(1),
                                            cost_fn= Euclidean(),
                                            batch_size= self.batchsize
                                        ).divergence 
        return d
    
    def compute_score(self, x, y, phi_x, weight_y, theta):
        return self.loss_func(theta, x, y, phi_x, weight_y)
    
    
    
    def solve_params(self, x, y, phi_x, weight_y, num_steps=1000):
        theta = jnp.ones((self.num_basis, 1), dtype=jnp.float32) ## change to float64
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(theta)

        @jax.jit
        def step(carry, _):
            theta, opt_state = carry
            loss, grads = jax.value_and_grad(self.loss_func)(theta, x, y, phi_x, weight_y)
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            # theta = jax.nn.relu(theta) + 1e-6
            ## add constraint intergrate equal 1
            theta = projection_non_negative_and_hyperplane(theta,(jnp.mean(phi_x,axis=0,keepdims=True).T, 1.))
            return (theta, opt_state), loss

        (theta, _), loss_vals = lax.scan(step, (theta, opt_state), None, length=num_steps)
        return theta, loss_vals[-1]



    def _choose_centers(self, x_test, n_centers,algo='kmeans',archor_min=None, archor_max=None):
        """
        Strategy to choose centers
        """

        points = select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=archor_min, anchor_max=archor_max)
        ## set center to true
        self.set_centers = True
        return points

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
        sigma_list = generate_param_ranges_jax(self.kernel_params.hyperparams, sigma_range, num_values=self.k_fold)
        sigma_list = [Params(kernel_name=self.kernel_params.kernel_name, hyperparams=p) for p in sigma_list]

        print('We are here')

        # need strategy to chose centers
        archor_min, archor_max, valid = find_intersection_anchor(x_train,x_test)
        if not valid:
            archor_min = None 
            archor_max = None

        self.__centers = self._choose_centers(x_test,n_centers=self.num_basis,algo='kmeans',archor_min=archor_min,archor_max=archor_max)

        ## product params
        list_params = sigma_list
        len_params = len(list_params)
        # batched_kernel_params = batch_params(list_params)
        
        # sigma_chosen = self.kfold_tuning(x_test, x_train, self.__centers, batched_kernel_params,len_params=len_params )
        sigma_chosen = self.kfold_tuning(x_test, x_train, self.__centers, list_params,len_params=len_params )
        
       
        # phi_numerator = self.covar_func(x=x_test, y=self.__centers, params=sigma_chosen) 
        phi_denominator = self.covar_func(x=x_train, y=self.__centers, params=sigma_chosen)
        weight_y = jnp.ones((len(x_test),1))/ len(x_test)
        coef, loss  = self.solve_params(x_train, x_test, phi_denominator,weight_y,num_steps=self.num_epochs )
        self.__coef = jnp.maximum(coef,0.)

        ## reassign hyperparameters
        self.kernel_params = sigma_chosen

        score = loss
        print(f'Finished training, the minimized function value: {score}')
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
    
    # def kfold_tuning(self, x_nu, x_de, centers,list_params, len_params=100):

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
    #         sigma = par.slice_at_index(idx)
    #         # unflatten parameters
    #         # compute score
    #         K =  self.covar_func(x=x, y=centers, params=sigma)
    #         def cv_loop(carry, params):
    #             # sigma, lambda_ = params
    #             cv_index = carry 
                
    #             index_te_nu = lax.dynamic_slice(cv_index_nu, (cv_index * fold_size_nu,),(fold_size_nu,))
    #             index_tr_nu = jnp.setdiff1d(cv_index_nu, index_te_nu,size=n_nu - fold_size_nu)

    #             index_te_de = lax.dynamic_slice(cv_index_de, (cv_index * fold_size_de,),(fold_size_de,))
    #             index_tr_de = jnp.setdiff1d(cv_index_de, index_te_de,size=n_de - fold_size_de)

    #             # Ktmp = K[jnp.hstack((index_tr_nu, index_tr_de)), :]

    #             # phi_y = Ktmp[:len(index_tr_nu), :]
    #             weight_y = jnp.ones((len(index_tr_nu),1))/len(index_tr_nu)
    #             phi_x = K[index_tr_de, :]
                
    #             sub_data = x[jnp.hstack((index_tr_nu, index_tr_de)), :]
    #             sub_y = sub_data[:len(index_tr_nu), :]
    #             sub_x = sub_data[len(index_tr_nu):, :]
                
                
    #             coef, loss  = self.solve_params(sub_x, sub_y, phi_x,weight_y,num_steps=self.num_epochs )
    #             # Ktmp_ = K[jnp.hstack((index_te_nu, index_te_de)), :]
    #             # phi_x_1 = K[:len(index_te_nu), :]
    #             phi_x_1 = K[index_te_de, :]
    #             sub_data_1 = x[jnp.hstack(((index_te_nu, index_te_de))), :]
    #             sub_y_1 = sub_data_1[:len(index_te_nu), :]
    #             sub_x_1 = sub_data_1[len(index_te_nu):, :]
    #             weight_y_1 = jnp.ones((len(index_te_nu),1))/len(index_te_nu)
                

    #             score = self.compute_score(sub_x_1, sub_y_1, phi_x_1, weight_y_1, coef)

    #             return cv_index +1 , score
    #         ## first, run cv
    #         _, cv_score = lax.scan(cv_loop, 0, jnp.arange(self.k_fold))
    #         return par, jnp.nanmean(cv_score)
    #     ## second, run outer loop
    #     _, final_score = lax.scan(params_loop, list_params, jnp.arange(len_params))
    #     ## find min score
    #     score_cv_min_index = jnp.nanargmin(final_score)
    #     sigma_chosen = list_params.slice_at_index(score_cv_min_index)
    #     print('Finish K fold')
    #     return sigma_chosen

  

    def kfold_tuning(self, x_nu, x_de, centers, list_params, len_params=100):
        """
        K-Fold tuning with small memory (compute K slices on the fly per fold).
        """

        n_nu = x_nu.shape[0]
        n_de = x_de.shape[0]
        key = jax.random.PRNGKey(123)

        cv_index_nu = jax.random.permutation(key, n_nu)
        cv_index_de = jax.random.permutation(key, jnp.arange(n_nu, n_nu + n_de))
        x = jnp.vstack((x_nu, x_de))

        fold_size_nu = int(jnp.ceil(n_nu / self.k_fold))
        fold_size_de = int(jnp.ceil(n_de / self.k_fold))
        all_cv_scores = []

        for idx in tqdm(range(len_params), desc="Param tuning"):
            sigma = list_params[idx]  # assume list_params is a jnp array

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

                # Extract training and validation data
                sub_data = x[jnp.hstack((index_tr_nu, index_tr_de)), :]
                sub_y = sub_data[:len(index_tr_nu), :]
                sub_x = sub_data[len(index_tr_nu):, :]

                sub_data_val = x[jnp.hstack((index_te_nu, index_te_de)), :]
                sub_y_val = sub_data_val[:len(index_te_nu), :]
                sub_x_val = sub_data_val[len(index_te_nu):, :]

                # Compute kernel slices on the fly
                phi_x = self.covar_func(x=x[index_tr_de], y=centers, params=sigma)
                phi_x_1 = self.covar_func(x=x[index_te_de], y=centers, params=sigma)

                weight_y = jnp.ones((len(index_tr_nu), 1)) / len(index_tr_nu)
                weight_y_val = jnp.ones((len(index_te_nu), 1)) / len(index_te_nu)

                # Solve and score
                coef, loss = self.solve_params(sub_x, sub_y, phi_x, weight_y, num_steps=self.num_epochs)
                score = self.compute_score(sub_x_val, sub_y_val, phi_x_1, weight_y_val, coef)

                fold_scores.append(score)

            fold_scores = jnp.stack(fold_scores)
            avg_cv_score = jnp.nanmean(fold_scores)
            all_cv_scores.append(avg_cv_score)

        all_cv_scores = jnp.stack(all_cv_scores)

        best_idx = jnp.nanargmin(all_cv_scores)
        # best_idx = int(best_idx)
        list_params = batch_params(list_params)

        sigma_chosen = list_params.slice_at_index(best_idx)

        print('Finish K-fold tuning.')

        return sigma_chosen


    


# class WassersteinRatiov2(DensityRatioEstimator):

#     def __init__(self, num_projection=60, layers_config=[128, 64, 32],
#                  learning_rate=1e-3, num_epochs=100, k_fold=5):

#         self.num_basis = num_projection
#         self.basis_func = 'squared_exponential'
#         self.kernel_params = create_params(self.basis_func)
#         self.num_epochs = num_epochs
#         self.learning_rate = learning_rate
#         self.k_fold = k_fold
#         self.covar_func = lambda x, y, params: create_covariance_v2(
#             name=self.basis_func, x=x, y=y, params=params
#         ) @ jnp.eye(len(y))

#         self.fitting = False

#     def _choose_centers(self, x_test, n_centers,algo='kmeans',archor_min=None, archor_max=None):
#         """
#         Strategy to choose centers
#         """

#         points = select_centers(x_test,n_centers=n_centers, algo=algo,anchor_min=archor_min, anchor_max=archor_max)
#         ## set center to true
#         self.set_centers = True
#         return points
    
#     def compute_score(self, x, y, phi_x, weight_y, theta):
#         return self.loss_func(theta, x, y, phi_x, weight_y)
    
#     @jax.jit
#     def loss_func(self, theta, x, y, phi_x, weight_y):
#         weight_x = jax.nn.relu(phi_x @ theta) + 1e-3
#         return sinkhorn_loss_v2(x, y, weight_x.squeeze(), weight_y.squeeze(), implicit=True)

#     def solve_params(self, x, y, phi_x, weight_y, num_steps=1000):
#         theta = jnp.ones((self.num_basis, 1), dtype=jnp.float32)
#         optimizer = optax.adam(self.learning_rate)
#         opt_state = optimizer.init(theta)

#         @jax.jit
#         def step(carry, _):
#             theta, opt_state = carry
#             loss, grads = jax.value_and_grad(self.loss_func)(theta, x, y, phi_x, weight_y)
#             updates, opt_state = optimizer.update(grads, opt_state)
#             theta = optax.apply_updates(theta, updates)
#             theta = jax.nn.relu(theta) + 1e-6
#             return (theta, opt_state), loss

#         (theta, _), loss_vals = lax.scan(step, (theta, opt_state), None, length=num_steps)
#         return theta, loss_vals[-1]

#     def fit(self, x_train, x_test):
#         p_dist = pairwise_distances(x_train, x_test).flatten()
#         sigma_min, sigma_max = jnp.percentile(p_dist, [10, 90])
#         sigma_range = create_default_range_params(self.kernel_params.kernel_name, {'length_scale': (sigma_min, sigma_max)})
#         sigma_list = generate_param_ranges_jax(self.kernel_params.hyperparams, sigma_range, num_values=self.k_fold)
#         sigma_list = [Params(kernel_name=self.kernel_params.kernel_name, hyperparams=p) for p in sigma_list]

#         archor_min, archor_max, valid = find_intersection_anchor(x_train, x_test)
#         archor_min, archor_max = (archor_min, archor_max) if valid else (None, None)

#         self.__centers = self._choose_centers(x_test, self.num_basis, algo='kmeans', archor_min=archor_min, archor_max=archor_max)

#         sigma_chosen = self.kfold_tuning(x_test, x_train, self.__centers, batch_params(sigma_list), len_params=len(sigma_list))

#         phi_denominator = self.covar_func(x_train, self.__centers, sigma_chosen)
#         weight_y = jnp.ones((len(x_test), 1)) / len(x_test)

#         coef, loss = self.solve_params(x_train, x_test, phi_denominator, weight_y, num_steps=self.num_epochs)
#         self.__coef = jnp.maximum(coef, 0.)

#         self.kernel_params = sigma_chosen
#         self.fitting = True
#         print(f'Finished training, minimized function value: {loss}')

#     def kfold_tuning(self, x_nu, x_de, centers, list_params, len_params=100):
#         n_nu, n_de = x_nu.shape[0], x_de.shape[0]
#         key = random.PRNGKey(123)
#         cv_index_nu = random.permutation(key, n_nu)
#         cv_index_de = random.permutation(key, jnp.arange(n_nu, n_nu + n_de))

#         x = jnp.vstack((x_nu, x_de))
#         fold_size_nu, fold_size_de = int(jnp.ceil(n_nu / self.k_fold)), int(jnp.ceil(n_de / self.k_fold))

#         def params_loop(param_carry, idx):
#             sigma = param_carry.slice_at_index(idx)
#             K = self.covar_func(x, centers, sigma)

#             def cv_loop(carry, params):
#                 cv_index = carry
#                 index_te_nu = lax.dynamic_slice(cv_index_nu, (cv_index * fold_size_nu,), (fold_size_nu,))
#                 index_tr_nu = jnp.setdiff1d(cv_index_nu, index_te_nu, size=n_nu - fold_size_nu)
#                 index_te_de = lax.dynamic_slice(cv_index_de, (cv_index * fold_size_de,), (fold_size_de,))
#                 index_tr_de = jnp.setdiff1d(cv_index_de, index_te_de, size=n_de - fold_size_de)

#                 weight_y = jnp.ones((len(index_tr_nu), 1)) / len(index_tr_nu)
#                 phi_x = K[index_tr_de, :]
#                 sub_data = x[jnp.hstack((index_tr_nu, index_tr_de)), :]
#                 sub_y, sub_x = sub_data[:len(index_tr_nu), :], sub_data[len(index_tr_nu):, :]

#                 coef, loss = self.solve_params(sub_x, sub_y, phi_x, weight_y, num_steps=self.num_epochs)

#                 phi_x_1 = K[index_te_de, :]
#                 sub_data_1 = x[jnp.hstack((index_te_nu, index_te_de)), :]
#                 sub_y_1, sub_x_1 = sub_data_1[:len(index_te_nu), :], sub_data_1[len(index_te_nu):, :]
#                 weight_y_1 = jnp.ones((len(index_te_nu), 1)) / len(index_te_nu)

#                 score = self.compute_score(sub_x_1, sub_y_1, phi_x_1, weight_y_1, coef)
#                 return cv_index + 1, score

#             _, cv_score = lax.scan(cv_loop, 0, jnp.arange(self.k_fold))
#             return param_carry, jnp.nanmean(cv_score)

#         _, final_score = lax.scan(params_loop, list_params, jnp.arange(len_params))
#         best_idx = jnp.nanargmin(final_score)
#         return list_params.slice_at_index(best_idx)
    
#     def compute_weights(self, x_train):
#         '''
#         Compute the weights for the training
#         Args:
#             x_train: training covariates
#             centers: 
#         returns:
#             weights: the weights for the training
#         '''
#         assert self.fitting, 'Need to fit the model before computing the weights'
#         jax.debug.print("Compute weight ") 
#         # phi_x = create_covariance_v2(name=self.basis_func, x=x_train, y=self.__centers, params=self.kernel_params) @ jnp.eye(len(self.__centers))
#         phi_x = self.covar_func(x=x_train, y=self.__centers, params=self.kernel_params)
#         weights =  phi_x @ self.__coef
        
#         return weights
