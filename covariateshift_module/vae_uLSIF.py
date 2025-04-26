'''
Define VAE uSLIF model
'''

from experiment_code.numpyro_model import NumpyroModel
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from covariateshift_module.neural_net import MLP

class VAE_uLSIF(NumpyroModel):

    @staticmethod
    def model( X_numerator, X_denominator, **kwargs):
        '''
        X_numerator: jnp.array, shape (n_samples, n_features). The data from the numerator (test) distribution
        X_denominator: jnp.array, shape (n_samples, n_features). The data from the denominator (train) distribution
        '''
        ## prior for ratios, have the same length as X_denominator. the value must >0 and go to infinity
        ratios = numpyro.sample('ratios', dist.HalfNormal(10.).expand([X_denominator.shape[0]]))
        ## now I want to compute the distance between two distributions, p(x_numerator) and p(x_denominator) * r(x)
        ## and this distance will follow some distribution
        ## first I will use a neural network to compute the distance
        
        pass


    @staticmethod
    def guide(X_numerator, X_denominator, **kwargs):
        '''
        X_numerator: jnp.array, shape (n_samples, n_features). The data from the numerator (test) distribution
        X_denominator: jnp.array, shape (n_samples, n_features). The data from the denominator (train) distribution
        '''


        pass 


    


