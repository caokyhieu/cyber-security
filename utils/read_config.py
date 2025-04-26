import yaml
import numpy as np
from collections import defaultdict

def _get_subdict(dictionary,listkeys=[]):
    results = defaultdict(None)
    for key in listkeys:
        results[key] = dictionary.get(key)
    return results


def _get_data_params(config_dict:dict):
    m1= np.array(config_dict.get('m1',None)).reshape(-1,)
    m2= np.array(config_dict.get('m2',None)).reshape(-1,)
    sig1= np.array(config_dict.get('sig1',None)).reshape(-1,)
    sig2= np.array(config_dict.get('sig2',None)).reshape(-1,)
    n_l= config_dict.get('n_l',None)
    n_u= config_dict.get('n_u',None)
    p =np.array(config_dict.get('p',None))
    gamma = np.array(config_dict.get('gamma',None))
    theta = np.array(config_dict.get('theta',None))
    alpha = config_dict.get('alpha',None)
    sigma_sq = config_dict.get('sigma_sq',None)

    data_params = {
                   'm1':m1,
                   'm2':m2,
                   'sig1':sig1,
                   'sig2':sig2,
                   'n_l':n_l,
                   'n_u':n_u,
                   'p':p,
                   'gamma':gamma,
                   'theta':theta,
                   'alpha':alpha,
                   'sigma_sq':sigma_sq,
                   }
    return data_params

def _get_mcmc_params(config_dict:dict):

    n_samples=config_dict.get('n_samples',1000)
    burn_in = config_dict.get('burn_in',1000)
    num_chains = config_dict.get('num_chains',4)
    thinning = config_dict.get('thinning',1)
    device = config_dict.get('device','cpu')
    init_strategy = config_dict.get('init_strategy','median')
    dense_mass = config_dict.get('dense_mass', False)
    trajectory_length = config_dict.get('trajectory_length', None)
    max_tree_depth = config_dict.get('max_tree_depth',10)
    adapt_step_size = config_dict.get('adapt_step_size', True)
    target_accept_prob = config_dict.get('target_accept_prob',0.8)
    chain_method = config_dict.get('chain_method','parallel')
    kernel = config_dict.get('kernel','NUTS')
    
    mcmc_args = {
                'num_samples':n_samples,
                'num_warmup': burn_in,
                'num_chains': num_chains,
                'thinning': thinning,
                'device':device,
                'init_strategy':init_strategy,
                'dense_mass':dense_mass,
                'trajectory_length':trajectory_length,
                'max_tree_depth':max_tree_depth,
                'adapt_step_size':adapt_step_size,
                'target_accept_prob':target_accept_prob,
                'chain_method':chain_method,
                'kernel': kernel,
                }
    
    return mcmc_args


def read_config_from_path(path:str):
    with open(path, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    
    num_run = config_dict['num_run']
    parent_folder = config_dict['parent_folder']
    random_seeds =  np.random.choice(100000000,num_run*10,replace=False)
    config_dict['seed'] = random_seeds.tolist()
    mcmc_args = _get_mcmc_params(config_dict=config_dict)
    data_params = _get_data_params(config_dict=config_dict)
    
    return random_seeds,parent_folder,num_run,data_params,mcmc_args,config_dict
