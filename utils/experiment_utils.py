import os 
import json
import yaml
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
from itertools import combinations
from torch.utils.data import DataLoader
from utils.dataset import PhotoDataset, jax_collate_fn

from scipy.stats import f_oneway, ttest_ind

from utils.pyro_utils import get_data
from utils.read_config import read_config_from_path
from utils.metric import MetricMileometer
from visualization import visualize as viz
import numpyro
from models.base_model import LinearModel, SpatialModelFast, compute_inverse_weights
from experiment_code.inference_model import SpatialVariationalInference, RegressionVariationalInference
from sklearn.preprocessing import StandardScaler
import pdb

def load_data(exp_path):
    '''
    Helper function to load data
    '''
    SI_train_path = os.path.join(exp_path, 'SI_train.npy')
    SI_test_path = os.path.join(exp_path, 'SI_test.npy')
    X_train_path = os.path.join(exp_path, 'X_train.npy')
    X_test_path = os.path.join(exp_path, 'X_test.npy')
    y_train_path = os.path.join(exp_path, 'y_train.npy')
    y_test_path = os.path.join(exp_path, 'y_test.npy')

    ## check exist
    if os.path.isfile(SI_train_path):
        SI_train = np.load(SI_train_path)
    else:
        raise ValueError(f'{SI_train_path} file not exist')

    if os.path.isfile(SI_test_path):
        SI_test = np.load(SI_test_path)
    else:
        raise ValueError(f'{SI_test_path} file not exist')

    ## check exist
    if os.path.isfile(X_train_path):
        X_train = np.load(X_train_path)
    else:
        raise ValueError(f'{X_train_path} file not exist')

    if os.path.isfile(X_test_path):
        X_test = np.load(X_test_path)
    else:
        raise ValueError(f'{X_test_path} file not exist')

    ## check exist
    if os.path.isfile(y_train_path):
        y_train = np.load(y_train_path)
    else:
        raise ValueError(f'{y_train_path} file not exist')

    if os.path.isfile(y_test_path):
        y_test = np.load(y_test_path)
    else:
        raise ValueError(f'{y_test_path} file not exist')

    return (X_train, SI_train, y_train),(X_test, SI_test, y_test)


def save_numpy_data(folder, data_dict):
    os.makedirs(folder, exist_ok=True)
    for name, data in data_dict.items():
        np.save(os.path.join(folder, f'{name}.npy'), data)

def run_and_save_linear_model(key,model_name,folder,mcmc_args, X_train, Y_train, X_test,method ,log_likelihood=True):
    name = f'BRM{"_" + method if method else ""}'
    model = LinearModel(key, name=model_name, distance_method=method, scale_method='', 
                            sigma=True, 
                            log_likelihood=log_likelihood,)
    model.setting_mcmc(mcmc_args)
    model.inference(X_train, Y_train, X_test)
    pred = model.predict(X_test)
    model.plot_posterior(f"{folder}/posterior_{name}.png")
    model.save_model(f"{folder}/{name}.pkl")
    assert np.isnan(pred).sum() == 0, "Prediction contains NaN"
    return pred

def run_and_save_gp_model(key, model_name, folder , mcmc_args, X_train, Y_train, X_test, SI_train=None, SI_test=None, **kwargs):
    model = SpatialModelFast(key, name=model_name, **kwargs)
    model.setting_mcmc(mcmc_args)
    distance_method = kwargs.get('distance_method')
    model.inference(X_train, Y_train, X_test, SI_train, SI_test)
    pred = model.predict(X_test, SI_train=SI_train, SI_test=SI_test)
    
    name = f'GP{"_" + distance_method if distance_method else ""}'
    model.plot_posterior(f'{folder}/{name}.png')
    model.save_model(f'{folder}/{name}.pkl')
    ## check prediction nan validation
    assert np.isnan(pred).sum() == 0, "Prediction contains NaN"
    return pred

def run_and_save_svgp_model(key, model_name,folder , X_train, Y_train, X_test, SI_train, SI_test, method, 
                            n_epochs=500, vi_method='SVI', num_particles=7, 
                            n_basis=2, stepsize=0.05, n_inducing_points=10, n_centers=20,log_likelihood=True):
    train_dataset = PhotoDataset(X_train, SI_train, Y_train)
    
    
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=jax_collate_fn, shuffle=True)
    test_dataset = PhotoDataset(X_test, SI_test, jnp.ones((len(X_test,)))) ## Y test is non-empty
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=jax_collate_fn, shuffle=False)
    test_loader.dataset.add_weights(jnp.ones(len(test_loader.dataset)))  # for consistency


    # Compute and apply importance weights
    iw = compute_inverse_weights(X_train, X_test, method, '', n_centers)
    train_loader.dataset.add_weights(iw)

    name = f'SVGP{"_" + method if method else ""}'
    svgp = SpatialVariationalInference(key, model_name, sigma=True, log_likelihood=log_likelihood)

    svgp.inference(train_loader, test_loader,
                   n_steps=n_epochs,
                   vi_method=vi_method,
                   num_particles=num_particles,
                   n_data=len(train_loader.dataset),
                   n_basis=n_basis,
                   stepsize=stepsize,
                   n_inducing_points=n_inducing_points)

    preds = svgp.predict(test_loader)[0]
    assert np.isnan(preds).sum() == 0, "Prediction contains NaN"

    svgp.plot_posterior(f'{folder}/posterior_model_{name}.png', keys=['weights', 'p', 'gamma', 'noise'])
    svgp.save_model(f'{folder}/{name}.pkl')

    return preds

def load_and_predict_linear(folder, method, X_test, return_model=False):
    name = f'BRM{"_" + method if method else ""}'
    path = os.path.join(folder, f"{name}.pkl")
    model = LinearModel.load_model(path)
    pred = model.predict(X_test)
    if return_model:
        return pred, model
    return pred

def load_and_predict_gp(folder, method, X_test, SI_train, SI_test, return_model=False):
    name = f'GP{"_" + method if method else ""}'
    path = os.path.join(folder, f"{name}.pkl")
    model = SpatialModelFast.load_model(path)
    pred = model.predict(X_test, SI_train=SI_train, SI_test=SI_test)
    if return_model:
        return pred, model
    return pred


def load_and_predict_svgp(folder, method, X_test, SI_test, return_model=False):
    name = f'SVGP{"_" + method if method else ""}'
    path = os.path.join(folder, f"{name}.pkl")
    model = SpatialVariationalInference.load_model(path)
    
    test_dataset = PhotoDataset(X_test, SI_test, jnp.ones((len(X_test),)))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=jax_collate_fn, shuffle=False)
    test_loader.dataset.add_weights(jnp.ones(len(test_loader.dataset)))  # dummy weights

    pred = model.predict(test_loader)[0]
    
    if return_model:
        return pred, model
    return pred
