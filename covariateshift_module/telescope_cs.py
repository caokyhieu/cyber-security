from covariateshift_module.utils import DensityRatioEstimator
from covariateshift_module.KLIEP import KLIEP
from covariateshift_module.uLSIF import uLSIF
from covariateshift_module.transformation import SequenceTransformer, SequenceRegularizedPCAWhitening
from covariateshift_module.KMM import KernelMeanMatching
from covariateshift_module.wasserstein_ratio import WassersteinRatio
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
class TelescopeCS(DensityRatioEstimator):
    def __init__(self, method='uLSIF',n_estimators=5,transformer_name='normal', **kwargs):
        self.n_estimators = n_estimators
        self.transformer_name = transformer_name
        
        if method == 'KLIEP':
            self.estimators = [KLIEP(**kwargs) for _ in range(n_estimators + 1)]
        elif method == 'uLSIF':
            self.estimators = [uLSIF(**kwargs) for _ in range(n_estimators + 1)]
        elif method == 'KMM':
            self.estimators = [KernelMeanMatching(**kwargs) for _ in range(n_estimators + 1)]
        elif method == 'Wasserstein':
            self.estimators = [WassersteinRatio(**kwargs) for _ in range(n_estimators + 1)]
        else:
            raise ValueError('Invalid method')

    def fit(self, x_denom, x_num):
        '''Fit the model to the data.'''
        num_means = jnp.mean(x_num, axis=0)
        denom_means = jnp.mean(x_denom, axis=0)
        
        if self.transformer_name == 'normal':
            # first, need to fit the transformer
            num_sds = jnp.std(x_num, axis=0)
            denom_sds = jnp.std(x_denom, axis=0)
            ## create the transformer
            self.trasformer = SequenceTransformer(n_transforms=self.n_estimators, 
                                                start_sd=num_sds, end_sd=denom_sds,
                                                start_mean=num_means, end_mean=denom_means)
        else:
            dest_cov = jnp.cov(x_num, rowvar=False)
            self.trasformer = SequenceRegularizedPCAWhitening(n_transforms=self.n_estimators, 
                                                              start_mean=num_means, end_mean=denom_means,
                                                              dest_cov=dest_cov)

        self.trasformer.fit(x_denom)
        for i in tqdm(range(self.n_estimators), desc=f'Fitting estimator'):
            if i==0:
                self.estimators[i].fit(self.trasformer.transform(x_denom, to_idx=i), x_num)
            else:
                self.estimators[i].fit(self.trasformer.transform(x_denom, to_idx=i), self.trasformer.transform(x_denom, to_idx=i-1))
        print(f'Fitting final estimator {len(self.estimators)}')
        self.estimators[-1].fit(x_denom, self.trasformer.transform(x_denom, to_idx=self.n_estimators-1))

    def compute_weights(self, x_denom):
        results = []
        for i, estimator in enumerate(self.estimators):
            if i< self.n_estimators:
                results.append(estimator.compute_weights(self.trasformer.transform(x_denom, to_idx=i)).reshape(-1,1))
            else:
                results.append(estimator.compute_weights(x_denom).reshape(-1,1))
        results = jnp.concatenate(results, axis=1)
        return jnp.prod(results, axis=1, keepdims=True)
        
    def kfold_tuning(self, x_denom, x_num):
        pass
