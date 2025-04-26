import jax
import jax.numpy as jnp
from flax.training import train_state
from optax import adam
from covariateshift_module.utils import DensityRatioEstimator
from covariateshift_module.wasserstein_ratio import WeightNetwork

def nce_loss(weight_p, weight_q,k):
    '''
    weight_p: the ratio of p(x)/q(x) for each x in the training set
    weight_q: the ratio of p(x)/q(x) for each x in the test set
    k: the ratio of test set data points with the train set data points
    '''
    positive = jnp.sum(jnp.log(weight_p/(weight_p + k)))
    negative = jnp.sum(jnp.log(k/(weight_q + k)))
    return -positive - negative

def update_fn(state, model, x_train, x_test, k):  # accept model as an argument
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params, model, x_train, x_test, k)  # pass model to loss_fn
    return state.apply_gradients(grads=grad), loss

def loss_fn(params, model, x_train, x_test, k):  # accept model as an argument
    weight_p = model.apply(params, x_train)  # use model.apply instead of calling model
    weight_q = model.apply(params, x_test) # use model.apply instead of calling model
    
    return nce_loss(weight_p, weight_q, k)
def train(model, x_train, x_test, learning_rate=0.01, num_epochs=100,k=1.):
    params = model.init(jax.random.PRNGKey(0), x_train)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adam(learning_rate))

    for epoch in range(num_epochs):
        state, loss = update_fn(state, model, x_train, x_test, k)  # pass model as an argument
        print(f"Epoch {epoch+1}, Loss: {loss}")

    return state.params


class NCEEstimator(DensityRatioEstimator):

    def __init__(self, layers_config:list=[128,64,32],
                 learning_rate:float=1e-3,num_epochs:int=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.layers_config = layers_config
        self.model = WeightNetwork(layers_config=self.layers_config)
        self.fitting = False
        self.params = None
        pass

    def fit(self, x_denom, x_num):
        k = len(x_num)/len(x_denom)
        self.params = train(self.model, x_denom, x_num, num_epochs=self.num_epochs,learning_rate=self.learning_rate,k=k)
        self.fitting = True

        pass

    def compute_weights(self, *args, **kwargs):
        if not self.fitting:
            raise ValueError('Model is not fitted yet')
        return self.model.apply(self.params, *args, **kwargs)
    
    def kfold_tuning(self, *args, **kwargs):
        pass


