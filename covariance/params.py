from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten

# @register_pytree_node_class
# class Params:
#     """
#     General class to manage kernel hyperparameters dynamically.
#     """
#     def __init__(self, kernel_name: str, hyperparams: dict):
#         """
#         Initialize Params.

#         Args:
#             kernel_name (str): Name of the kernel (e.g., 'matern', 'squared_exponential').
#             hyperparams (dict): Dictionary of kernel-specific hyperparameters.
#         """
#         self.kernel_name = kernel_name
#         self.hyperparams = hyperparams
        
        

#     def update(self, **kwargs):
#         """
#         Update hyperparameters immutably (returns a new Params instance).

#         Args:
#             kwargs: Updated hyperparameter values.

#         Returns:
#             Params: A new Params instance with updated hyperparameters.
#         """
#         updated_hyperparams = {**self.hyperparams, **kwargs}
#         return Params(self.kernel_name, updated_hyperparams)

#     def get(self, param_name, default=None):
#         """
#         Get the value of a specific hyperparameter.

#         Args:
#             param_name (str): Name of the hyperparameter.
#             default: Default value if the parameter is not found.

#         Returns:
#             The value of the hyperparameter or the default value.
#         """
#         return self.hyperparams.get(param_name, default)

#     def tree_flatten(self):
#         flat_hyperparams = []
#         hyperparam_shapes = {}

#         for key, value in self.hyperparams.items():
#             value_array = jnp.array(value)
#             flat_hyperparams.extend(value_array.ravel())
#             hyperparam_shapes[key] = value_array.shape

#         aux_data = (self.kernel_name, list(self.hyperparams.keys()), hyperparam_shapes)
#         return flat_hyperparams, aux_data


#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         """
#         Custom unflattening logic for the Params object.

#         Args:
#             aux_data: A tuple containing kernel_name, hyperparameter keys, and their shapes.
#             children: A flattened list of hyperparameter values.

#         Returns:
#             Params: A reconstructed Params instance.
#         """
#         kernel_name, hyperparam_keys, hyperparam_shapes = aux_data

#         hyperparams = {}
#         current_index = 0
#         for key in hyperparam_keys:
#             # Get the shape of the current hyperparameter
#             shape = hyperparam_shapes[key]

#             # Calculate the number of elements in this hyperparameter
#             size = jnp.prod(jnp.array(shape).astype(int)).item() # Convert to Python integer

#             # Slice the flattened array and reshape it to the original shape
#             hyperparams[key] = jnp.array(children[current_index:current_index + size]).reshape(shape)

#             # Move the index forward
#             current_index += size

#         return cls(kernel_name, hyperparams)

#     def __repr__(self):
#         return f"Params(kernel_name={self.kernel_name}, hyperparams={self.hyperparams})"



@register_pytree_node_class
class Params:
    """
    General class to manage kernel hyperparameters dynamically.
    """
    def __init__(self, kernel_name: str, hyperparams: dict):
        self.kernel_name = kernel_name
        self.hyperparams = hyperparams

    def update(self, **kwargs):
        updated_hyperparams = {**self.hyperparams, **kwargs}
        return Params(self.kernel_name, updated_hyperparams)

    def get(self, param_name, default=None):
        return self.hyperparams.get(param_name, default)

    def tree_flatten(self):
        # Collect hyperparameters in insertion order to maintain consistency
        children = []
        keys = []
        for key, value in self.hyperparams.items():
            keys.append(key)
            children.append(jnp.asarray(value))
        aux_data = (self.kernel_name, keys)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel_name, keys = aux_data
        hyperparams = dict(zip(keys, children))
        return cls(kernel_name, hyperparams)
    
    def slice_at_index(self, idx):
        """Slice batched parameters at a specific index using JAX-safe operations."""
        sliced_hyperparams = {
            k: v[idx] if isinstance(v, jnp.ndarray) else v
            for k, v in self.hyperparams.items()
        }
        return Params(self.kernel_name, sliced_hyperparams)
    def __repr__(self):
        return f"Params(kernel_name={self.kernel_name}, hyperparams={self.hyperparams})"

def batch_params(params_list):
    """Convert a list of Params into a batched Params object."""
    kernel_name = params_list[0].kernel_name
    batched_hyperparams = {}

    # Assume all Params have the same hyperparameter keys
    for key in params_list[0].hyperparams.keys():
        # Stack values across folds/batches
        batched_values = jnp.stack([p.hyperparams[key] for p in params_list], axis=0)
        batched_hyperparams[key] = batched_values

    return Params(kernel_name, batched_hyperparams)

    

import jax.numpy as jnp
from sklearn.model_selection import ParameterGrid

def create_default_range_params(kernel_name, dict_range=None):
    """
    Generate a range of values for each parameter.

    Args:
        kernel_name (str): Kernel name.
        dict_range (dict): Dictionary with min and max values for each parameter. Default is None.
    Returns:
        dict: A dictionary with each parameter mapped to a  range (min, max)
    """
    ## check dict range is valid
    if dict_range is not None:
        if not isinstance(dict_range, dict):
            raise ValueError("dict_range must be a dictionary.")
    else:
        ## check dict range in format {params: (min,max)}
        for param in dict_range:
        
            if not isinstance(dict_range[param], tuple):
                raise ValueError("dict_range must be in format {params: (min,max)}.")
            ## check length of dict range
            if len(dict_range[param]) != 2:
                raise ValueError("dict_range must be in format {params: (min,max)}.")
            
            if not (isinstance(dict_range[param][0], (int, float)) and isinstance(dict_range[param][1], (int, float))):
                raise ValueError("dict_range values must be numbers.")
            if dict_range[param][0] > dict_range[param][1]:
                raise ValueError("dict_range values must be in ascending order.")
            
    ## check name of params
    param_ranges = {}
    if kernel_name == 'squared_exponential':
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'mixture_squared_exponential':
        param_ranges['length_scales'] = (0.1, 10.)
        param_ranges['mixture_weights'] = (0., 1.)
        
    elif kernel_name == 'squared_exponential_spatial':
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'rational_quadratic':
        param_ranges['length_scale'] = (0.1, 10.)
        param_ranges['alpha'] = (0.1, 2.)
    elif kernel_name == 'matern':
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'periodic':
        param_ranges['length_scale'] = (0.1, 10.)
        param_ranges['period'] = (0.1, 10.)
    elif kernel_name == 'linear':
        param_ranges['c'] = (1e-6, 10.)
    elif kernel_name == 'polynomial':
        param_ranges['degree'] = (1, 5)
        param_ranges['coef0'] = (1e-6, 10.)
    elif kernel_name =='matern_spatial':
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'locally_periodic':
        param_ranges['length_scale_se'] = (0.1, 10.)
        param_ranges['length_scale_pk'] = (0.1, 10.)
        param_ranges['period'] = (0.1, 10.)
    elif kernel_name =='sm':
        param_ranges['weights'] = (0., 1.)
        param_ranges['scales'] = (0., 1.)
        param_ranges['means'] = (-10., 10.)
    elif kernel_name =='ard':
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name =='vonMisesFisher':
        param_ranges['kappa'] = (1., 10.)
    elif kernel_name == 'linear_trend_multiply_matern':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'linear_trend_multiply_periodicity':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale_se'] = (0.1, 10.)
        param_ranges['length_scale_pk'] = (0.1, 10.)
        param_ranges['period'] = (0.1, 10.)
    elif kernel_name == 'linear_trend_multiply_rational_quadratic':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
        param_ranges['alpha'] = (0.1, 2.)
    elif kernel_name == 'linear_trend_multiply_square_exponential':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'linear_trend_with_matern':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name == 'linear_trend_with_rational_quadratic':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
        param_ranges['alpha'] = (0.1, 2.)
    elif kernel_name == 'linear_trend_with_square_exponential':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
    elif kernel_name =='linear_trend_with_periodicity':
        param_ranges['c'] = (1e-6, 10.)
        param_ranges['length_scale'] = (0.1, 10.)
        param_ranges['period'] = (0.1, 10.)
    elif kernel_name == 'mixture_linear':
        param_ranges['weights'] = (0., 1.)
        param_ranges['biases'] = (-10., 10.)
    elif kernel_name == 'uLSIF':
        param_ranges['alpha'] = (0.01, 1.)
        param_ranges['lamda'] = (0.01, 2.)
    else:
        raise ValueError(f"Kernel {kernel_name} not supported.")

    ## update param_ranges with dict_range if dict_range is not None
    if dict_range is not None:
        for param in param_ranges:
            if param in dict_range:
                param_ranges[param] = dict_range[param]
    return param_ranges
def generate_param_ranges_jax(params, param_ranges, num_values=5):
    """
    Generate a range of values for each parameter based on specified min/max constraints using JAX.

    Args:
        params (dict): Dictionary of parameters with their current values.
        param_ranges (dict): Dictionary with min and max values for each parameter.
        num_values (int): Number of values to generate within the range (default is 5).

    Returns:
        dict: A dictionary with each parameter mapped to a list of values within the given range.
    """
    param_grid = {}

    for param, value in params.items():
        if param in param_ranges:
            min_val, max_val = param_ranges[param]
            if isinstance(value, int):
                param_grid[param] = jnp.linspace(min_val, max_val, num_values).astype(jnp.int32).tolist()
            elif isinstance(value, float):
                param_grid[param] = jnp.linspace(min_val, max_val, num_values).tolist()
            else:
                param_grid[param] = [value]  # Keep fixed value if not numeric
        else:
            param_grid[param] = [value]  # Keep original value if no range is provided
    list_grid =  list(ParameterGrid(param_grid))
    return list_grid
