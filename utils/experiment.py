from abc import ABCMeta, abstractmethod
from utils.read_config import read_config_from_path
from utils.metric import MetricMileometer
import pandas as pd
## implement abstract class for Experiment
import os
import json
import wandb
import jax.numpy as jnp
from jax import random
# from jaxlib.xla_extension import DeviceArray
from numpy import int64, ndarray
from torch.utils.data.dataloader import DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.wandb_run import Run
import pdb

## login wandb
# api_key = os.environ.get("WANDB_API_KEY")
api_key = "3580656f3774a8365c2cd036147cb3e5b6383a30"

wandb.login(key=api_key)
from jax import config
config.update('jax_debug_nans', False)
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', False)
## set seed for numpy
import numpy as np
# np.random.seed(0)
class BaseExperiment(metaclass=ABCMeta):

    def __init__(self,models:list=[]) -> None:
        '''
        add model names and metric obj to measure some statistics
        '''
        self.model_name = models
        self.mileometer_dict:dict = {name:MetricMileometer() for name in models}
        pass 
    
    @abstractmethod
    def one_experiment(self,randomseed=None,parent_folder=None,
                        data_params=None,mcmc_args=None,config_dict=None,iteration=0):
        '''
        This is the heart of this class
        In this method, we need to evaluate all methods we use and 
        return a ditionary object, with key is the model name and value is a list (dict,labels),
        with dict is the dictionary of samples from a group of models that we want to measure
        together.
        
        '''
        return 
    
    
    def run(self,config_file_path: str,ignore_errors: bool=True) -> None:
        '''
        Run the experiments with params extract from config file
        '''
        
        random_seeds,parent_folder,num_run,data_params,mcmc_args,config_dict = read_config_from_path(config_file_path)
        if not os.path.isdir(parent_folder):
            os.makedirs(parent_folder)
        wandb.init()
        run = wandb.init(
                        # Set the project where this run will be logged
                        project="local-variational",
                        notes="My first experiment",
                        tags=["baseline", "paper1"],
                        # Track hyper
                        config=config_dict)
        
        iteration = 0
        for i in range(len(random_seeds)):
            print(f"Time {iteration}")
            # try:
                ## do something
            self.one_experiment(random_seeds[i],parent_folder,data_params,mcmc_args,config_dict,iteration=iteration,run=run)
                
            
            iteration += 1
            
            if iteration>=num_run:
                break
        
        ## print the result
        for model in self.model_name:
            self.mileometer_dict[model].train_test_compare()
            self.mileometer_dict[model].count()
            self.mileometer_dict[model].plot_accumuate_error(f'{parent_folder}/accumulated_{model}_error.png')
        
    
        #  save config file to reproduce
        with open(f"{parent_folder}/config.json", "w") as output_file:
            json.dump(config_dict, output_file, indent = 6)
        
        # Finish the run (useful in notebooks)
        run.finish()
        pass

from utils.pyro_utils import read_real_data
from jax import random
from models.variational_model import VariationalLinearModel
import os

class VariationalExperiment(BaseExperiment):

    def __init__(self, models=['linear'],photo_path=None,
                 spectral_path=None,normalize_data=True,
                 batchsize=128,n_epochs=20,vi_method='SVI',
                 num_particles=20):
        super().__init__(models)
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.normalize_data = normalize_data
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.vi_method = vi_method
        self.num_particles = num_particles
        
    
    def one_experiment(self,randomseed=None,parent_folder=None,
                        data_params=None,mcmc_args=None,config_dict=None,iteration=0,run=None):
        ''' In this experiment, we just compare the linear variaitonal models'''

        ## read data
        key = random.PRNGKey(randomseed)
        data  = read_real_data(self.photo_path,self.spectral_path,random_state=randomseed,normalize=self.normalize_data)
        
        X_train, SI_train, Y_train = data[0]
        X_test, SI_test, Y_test = data[1]
        # if len(data)==3:
        #     stats_data = data[2]

        ## check dim data, should change to atleast_2d func
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
        
        ############################## Linear model #################################
        # model 0, Linear model
        BRM = VariationalLinearModel(key,name='multivariate_regression_model' ,
                                    distance_method='',
                                    scale_method='concentration',
                                    sigma=None,
                                    log_likelihood=False,
                                    batchsize=self.batchsize
                                    )

        # BRM.setting_mcmc(mcmc_args)
        # BRM.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        BRM.inference(X_train, Y_train, X_test, SI_train, SI_test,
                    n_steps=self.n_epochs,
                    vi_method=self.vi_method,
                    num_particles=self.num_particles,
                    n_data=len(X_train))
        predictive0 = BRM.predict(X_test)
        predictive0_0 = BRM.predict(X_train)
        ## create folder
        folder_name = f"{parent_folder}/exp_{iteration}"
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        BRM.plot_posterior(f'{folder_name}/BRM.png')

        ### NRM NCA model        
        BRM_NCA = VariationalLinearModel(key,name='multivariate_regression_model',
                                            distance_method='NCA',
                                            scale_method='concentration',
                                            sigma=None,
                                            log_likelihood=False,
                                            propensity=False,
                                            batchsize=self.batchsize
                                            )
        # BRM_NCA.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        BRM_NCA.inference(X_train, Y_train, X_test, SI_train, SI_test,
                        n_steps=self.n_epochs,
                        vi_method=self.vi_method,
                        num_particles=self.num_particles,
                        n_data=len(X_train))
        predictive1 = BRM_NCA.predict(X_test)
        predictive1_0 = BRM_NCA.predict(X_train)
        BRM_NCA.plot_posterior(f'{folder_name}/BRM_NCA.png')
        BRM_NCA.plot_covariates(X_train,X_test,root_path=f'{folder_name}/BRM_NCA_')

        ################################################################
        BRM_OT = VariationalLinearModel(key, name='multivariate_regression_model',
                                        distance_method='OT',
                                        scale_method='concentration',
                                        sigma=None,
                                        log_likelihood=False,
                                        propensity=False,
                                        batchsize=self.batchsize
                                        )
        # BRM_OT.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        BRM_OT.inference(X_train, Y_train, X_test, SI_train, SI_test,
                        n_steps=self.n_epochs,
                        vi_method=self.vi_method,
                        num_particles=self.num_particles,
                        n_data=len(X_train))
        predictive2 = BRM_OT.predict(X_test)
        predictive2_0 = BRM_OT.predict(X_train)
        BRM_OT.plot_posterior(f'{folder_name}/BRM_OT.png')
        BRM_OT.plot_covariates(X_train,X_test,root_path=f'{folder_name}/BRM_OT_')

        ################################################################
        BRM_KMM = VariationalLinearModel(key, name='multivariate_regression_model',
                                        distance_method='KMM',
                                        scale_method='concentration',
                                        sigma=None,
                                        log_likelihood=False,
                                        propensity=False,
                                        batchsize=self.batchsize
                                        )
        # BRM_KMM.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        BRM_KMM.inference(X_train, Y_train, X_test, SI_train, SI_test,
                        n_steps=self.n_epochs,
                            vi_method=self.vi_method,
                            num_particles=self.num_particles,
                            n_data=len(X_train))
        predictive3 = BRM_KMM.predict(X_test)
        predictive3_0 = BRM_KMM.predict(X_train)
        BRM_KMM.plot_posterior(f'{folder_name}/BRM_KMM.png')
        BRM_KMM.plot_covariates(X_train,X_test,root_path=f'{folder_name}/BRM_KMM_')

        ################################################################
        linear_samples = {
                        'test BRM': predictive0,
                        'test BRM + NCA':predictive1,
                        'test BRM + OT':predictive2,
                        'test BRM + KMM':predictive3,
                        }
        train_linear_samples = {
                        'train BRM': predictive0_0,
                        'train BRM + NCA':predictive1_0,
                        'train BRM + OT':predictive2_0,
                        'train BRM + KMM':predictive3_0,

        }
        ### one_experiment does nt need to return any thing
        err1 = self.mileometer_dict[self.model_name[0]].measure(linear_samples,Y_test)
        err2 = self.mileometer_dict[self.model_name[0]].measure(train_linear_samples,Y_train)
        err1.update(err2)
        run.log({i:v.R2 for i,v in err1.items()})

        pass

class VariationalExperimentTuning(BaseExperiment):

    def __init__(self, models=['linear'],photo_path=None,
                 spectral_path=None,normalize_data=True,
                 batchsize=128,n_epochs=20,vi_method='SVI',
                 num_particles=20,max_n_basis=5):
        super().__init__(models)
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.normalize_data = normalize_data
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.vi_method = vi_method
        self.num_particles = num_particles
        self.max_n_basis = max_n_basis
        
    
    def one_experiment(self,randomseed=None,parent_folder=None,
                        data_params=None,mcmc_args=None,config_dict=None,iteration=0,run=None):
        ''' In this experiment, we just compare the linear variaitonal models'''

        ## read data
        key = random.PRNGKey(randomseed)
        data  = read_real_data(self.photo_path,self.spectral_path,random_state=randomseed,normalize=self.normalize_data)
        
        X_train, SI_train, Y_train = data[0]
        X_test, SI_test, Y_test = data[1]
        # if len(data)==3:
        #     stats_data = data[2]

        ## check dim data, should change to atleast_2d func
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
        # linear_samples = {}
        # train_linear_samples = {}
        # shuffle_feats = jnp.arange(X_train.shape[-1])
        # random.permutation(key,shuffle_feats,independent=True)
        # X_train = X_train[:, shuffle_feats]
        # X_test = X_test[:, shuffle_feats]
        total_err = {}
        for d in [1,3,5]:
            for n_basis in [1,3,5]:
                selected_features = [i for i in range(d)]
                predictive0, predictive0_0 = self.sub_experiment(key, X_train, X_test, Y_train, SI_train, SI_test, n_basis,selected_features )
               
            ################################################################
                err1 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'test BRM {d}-{n_basis}': predictive0,
                                    },Y_test)
                
                err2 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'train BRM {d}-{n_basis}': predictive0_0,
                                    },Y_train)
                total_err.append(err1)
                total_err.append(err2)
                # linear_samples.update({
                #                 f'test BRM {d}-{n_basis}': predictive0,
                #                 })
                # train_linear_samples.update({
                #                 f'train BRM {d}-{n_basis}': predictive0_0,
                #                 })

        # return {self.model_name[0]:[(linear_samples,Y_test),(train_linear_samples,Y_train)]}
        run.log({i:v.R2 for i,v in total_err.items()})
        pass
    

    def sub_experiment(self,key, X_train, X_test, Y_train, SI_train, SI_test, n_basis,selected_features ):

        ############################## Linear model #################################
        # model 0, Linear model
        BRM = VariationalLinearModel(key,name='multivariate_regression_model' ,
                                    distance_method='',
                                    scale_method='concentration',
                                    sigma=None,
                                    log_likelihood=False,
                                    batchsize=self.batchsize
                                    )

        # BRM.setting_mcmc(mcmc_args)
        # BRM.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        BRM.inference(X_train[:,selected_features], Y_train, X_test[selected_features], SI_train, SI_test,
                    n_steps=self.n_epochs,
                    vi_method=self.vi_method,
                    num_particles=self.num_particles,
                    n_data=len(X_train),
                    n_basis=n_basis)
        predictive0 = BRM.predict(X_test[:,selected_features])
        predictive0_0 = BRM.predict(X_train[:,selected_features])
       
        return predictive0, predictive0_0


from utils.dataset import ProcessPhotoData
from models.variational_model import VariationalLinearModelv2
from covariateshift_code.base_cs import CovariateShiftMethod


class VariationalExperimentv2(BaseExperiment):

    def __init__(self, models: List[str]=['linear'],photo_path: Optional[Union[str,List[str]]]=None,
                 spectral_path: Optional[str]=None,normalize_data: bool=True,
                 batchsize: int=128,n_epochs: int=20,vi_method: str='SVI',
                 num_particles: int=20,max_n_basis: int=5,feature_selection: bool=False,
                 distance_method: str='uLSIF',scale_method: str='concentration',beta_a:float=1.,beta_b:float=1.,
                 metric='MAPE',dataset:str='',step_basis=1,stepsize=1e-3) -> None:
        super().__init__(models)
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.normalize_data = normalize_data
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.vi_method = vi_method
        self.num_particles = num_particles
        self.max_n_basis = max_n_basis
        self.feature_selection = feature_selection
        self.distance_method = distance_method
        self.scale_method = scale_method
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.metric = metric
        self.dataset = dataset
        self.step_basis = step_basis
        self.stepsize = stepsize
        self.plot_posterior = False

       
    def one_experiment(self,randomseed: Optional[int64]=None,parent_folder: Optional[str]=None,
                        data_params: Optional[Dict[str, Union[ndarray, int, float]]]=None,mcmc_args: Optional[Dict[str, Optional[Union[int, str, bool, float]]]]=None,config_dict: Optional[Dict[str, Any]]=None,iteration: int=0,run: Optional[Run]=None) -> None:
        ''' In this experiment, we just compare the linear variaitonal models'''

        ## read data
        key = random.PRNGKey(randomseed)
        # data  = read_real_data(self.photo_path,self.spectral_path,random_state=randomseed,normalize=self.normalize_data)
         ### add some process needed
        process_data = ProcessPhotoData(randomseed, self.photo_path, self.spectral_path,
                                        batchsize=self.batchsize,normalize=self.normalize_data,
                                        feature_selection=self.feature_selection,
                                        beta_a=self.beta_a,beta_b=self.beta_b,dataset=self.dataset)
        data = process_data.load_data()
        
        train_loader = data[0]
        test_loader = data[1]
        # pdb.set_trace()
        ##
        cv_method = CovariateShiftMethod(key ,distance_method=self.distance_method,scale_method=self.scale_method)
        ## compute weights
        train_weights = cv_method.compute_similarity(train_loader,test_loader)
        # pdb.set_trace()
        ## add weights to ds
        train_loader.dataset.add_weights(train_weights)
        ## add weights to test ds
        test_loader.dataset.add_weights(jnp.ones(len(test_loader.dataset)))
        # linear_samples = {}
        # linear_bic = {}
        # train_linear_samples = {}
        # shuffle_feats = jnp.arange(X_train.shape[-1])
        # random.permutation(key,shuffle_feats,independent=True)
        # X_train = X_train[:, shuffle_feats]
        # X_test = X_test[:, shuffle_feats]
        total_err = {}
        training_loss = {}
        _eval_loss = {}
        Y_test = process_data.postprocess_data(test_loader.dataset.get_labels())
        Y_train = process_data.postprocess_data(train_loader.dataset.get_labels())
       
        for n_basis in range(1,self.max_n_basis+1,self.step_basis):
            predictive0, predictive0_0, bic_score , log_loss, eval_loss ,fig_posterior= self.sub_experiment(key, train_loader,test_loader, n_basis )
            ## post-process for output
            predictive0 = process_data.postprocess_data(predictive0)
            predictive0_0 = process_data.postprocess_data(predictive0_0)
            if fig_posterior is not None:
                run.log({f"chart pair posterior num basis:{n_basis} iteration:{iteration}": fig_posterior[0]},commit=False)
                run.log({f"chart trace posterior num basis:{n_basis} iteration:{iteration}": fig_posterior[1]},commit=False)
            
        ################################################################

            err1 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'test BRM {n_basis}-basis': predictive0,
                                                        },Y_test)
                
            err2 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'train BRM {n_basis}-basis': predictive0_0,
                                    },Y_train)
            err3 = self.mileometer_dict[self.model_name[0]].measure({
                                     f'BRM {n_basis}-basis bic': bic_score,
                                    }, None)
            total_err.update(err1)
            total_err.update(err2)
            total_err.update(err3)
            training_loss.update({f'training loss {n_basis}-basis': log_loss})
            _eval_loss.update({f'eval loss {n_basis}-basis': eval_loss})
        
        # # log table
        # df1 = pd.DataFrame(training_loss)
        # df1 = df1.reset_index()
        # df1 = pd.melt(df1,id_vars=['index'])
        # run.log({f"custom_data_table_training loss - {iteration}": wandb.Table(data=df1.values.tolist(),
        #             columns = df1.columns.tolist())},commit=False)
        
        # df1 = pd.DataFrame(_eval_loss)
        # df1 = df1.reset_index()
        # df1 = pd.melt(df1,id_vars=['index'])
        # run.log({f"custom_data_table_eval loss - {iteration}": wandb.Table(data=df1.values.tolist(),
        #             columns = df1.columns.tolist())},commit=False)
       
        ## log all err
        run.log({i:getattr(v,self.metric) for i,v in total_err.items()})
        pass

    def sub_experiment(self,key, train_loader: DataLoader, test_loader: DataLoader, n_basis: int ) :
        # ## alter features to 
        # train_loader.dataset.update_covariates(selected_features)
        # test_loader.dataset.update_covariates(selected_features)
       

        ############################## Linear model #################################
        # model 0, Linear model
        BRM = VariationalLinearModelv2(key,name='multivariate_regression_model' ,
                                    sigma=None,
                                    log_likelihood=False
                                    )

        # BRM.setting_mcmc(mcmc_args)
        # BRM.inference(X_train, Y_train, X_test,mcmc_args=mcmc_args)
        log_loss,eval_loss =  BRM.inference(train_loader,
                                            test_loader,
                                            n_steps=self.n_epochs,
                                            vi_method=self.vi_method,
                                            num_particles=self.num_particles,
                                            n_data=len(train_loader.dataset),
                                            n_basis=n_basis,
                                            stepsize=self.stepsize)
        fig_posterior = None
        if self.plot_posterior:
            fig_posterior = BRM.plot_posterior(None)
        train_covariates = train_loader.dataset.get_covariates()
        # predictive0 = BRM.predict(test_loader.dataset.get_covariates())
        # predictive0_0 = BRM.predict(train_covariates)
        predictive0 = BRM.predictv2(test_loader.dataset.get_covariates(),n_basis=n_basis)
        predictive0_0 = BRM.predictv2(train_covariates,n_basis=n_basis)
        
        # # create folder
        # folder_name = f"{parent_folder}/exp_{iteration}"
        # if not os.path.isdir(folder_name):
        #     os.makedirs(folder_name)

        # BRM.plot_posterior(f'{folder_name}/BRM.png')

        return predictive0, predictive0_0, BRM.bic_score, log_loss, eval_loss, fig_posterior


from models.variational_model import VariationalSpatialModelv2,VariationalSpatialModelv3
from utils.pyro_utils import calculate_sparse_matrix
from visualization.process_file_wandb import save_point_animation
class VariationalExperimentv3(BaseExperiment):

    def __init__(self, models: List[str]=['linear'],photo_path: Optional[Union[str,List[str]]]=None,
                 spectral_path: Optional[str]=None,normalize_data: bool=False,
                 batchsize: int=128,n_epochs: int=20,vi_method: str='SVI',
                 num_particles: int=20,max_n_basis: int=5,feature_selection: bool=False,
                 distance_method: str='uLSIF',scale_method: str='concentration',beta_a:float=1.,beta_b:float=1.,
                 metric='MAPE',dataset:str='',step_basis=1,stepsize=1e-3) -> None:
        super().__init__(models)
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.normalize_data = normalize_data
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.vi_method = vi_method
        self.num_particles = num_particles
        self.max_n_basis = max_n_basis
        self.feature_selection = feature_selection
        self.distance_method = distance_method
        self.scale_method = scale_method
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.metric = metric
        self.dataset = dataset
        self.step_basis = step_basis
        self.stepsize = stepsize
        self.plot_posterior = False

       
    def one_experiment(self,randomseed: Optional[int64]=None,parent_folder: Optional[str]=None,
                        data_params: Optional[Dict[str, Union[ndarray, int, float]]]=None,mcmc_args: Optional[Dict[str, Optional[Union[int, str, bool, float]]]]=None,config_dict: Optional[Dict[str, Any]]=None,iteration: int=0,run: Optional[Run]=None) -> None:
        ''' In this experiment, we just compare the linear variaitonal models'''

        ## read data
        key = random.PRNGKey(randomseed)
        # data  = read_real_data(self.photo_path,self.spectral_path,random_state=randomseed,normalize=self.normalize_data)
         ### add some process needed
        process_data = ProcessPhotoData(randomseed, self.photo_path, self.spectral_path,
                                        batchsize=self.batchsize,normalize=self.normalize_data,
                                        feature_selection=self.feature_selection,
                                        beta_a=self.beta_a,beta_b=self.beta_b,dataset=self.dataset)
        data = process_data.load_data()
        
        train_loader = data[0]
        test_loader = data[1]
        print(f"Length of train data:{len(train_loader.dataset)}")
        print(f"Length of test data:{len(test_loader.dataset)}")
        # pdb.set_trace()
        ##
        cv_method = CovariateShiftMethod(key ,distance_method=self.distance_method,scale_method=self.scale_method)
        ## compute weights
        train_weights = cv_method.compute_similarity(train_loader,test_loader)
        ## add weights to ds
        train_loader.dataset.add_weights(train_weights)
        ## add weights to test ds
        test_loader.dataset.add_weights(jnp.ones(len(test_loader.dataset)))
       
        total_err = {}
        training_loss = {}
        _eval_loss = {}
        
        for n_basis in range(1,self.max_n_basis+1,self.step_basis):
            predictive0, predictive0_0, inference_results ,fig_posterior= self.sub_experiment(key, train_loader,test_loader, n_basis )
            predictive0,label0 = predictive0
            predictive0_0,label0_0 = predictive0_0
            ## post-process for output
            predictive0 = process_data.postprocess_data(predictive0)
            predictive0_0 = process_data.postprocess_data(predictive0_0)
            if fig_posterior is not None:
                run.log({f"chart pair posterior num basis:{n_basis} iteration:{iteration}": fig_posterior[0]},commit=False)
                run.log({f"chart trace posterior num basis:{n_basis} iteration:{iteration}": fig_posterior[1]},commit=False)
            
        ################################################################

            err1 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'test BRM {n_basis}-basis': predictive0,
                                                        },process_data.postprocess_data(label0))
                
            err2 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'train BRM {n_basis}-basis': predictive0_0,
                                    },process_data.postprocess_data(label0_0))
            
            err3 = self.mileometer_dict[self.model_name[0]].measure({
                                     f'BRM {n_basis}-basis bic': inference_results.BIC,
                                    }, None)
            total_err.update(err1)
            total_err.update(err2)
            total_err.update(err3)
            training_loss.update({f'training loss {n_basis}-basis': inference_results.train_loss})
            _eval_loss.update({f'eval loss {n_basis}-basis': inference_results.test_loss})
        
        ## log all err
        metrics = ['MSE','R2','MAPE']
        for metric in metrics:
            run.log({i + metric:getattr(v,metric) for i,v in total_err.items()})
        pass
        # run.log({i:getattr(v,self.metric) for i,v in total_err.items()})
        # pass
        ## plot inducing points
        save_point_animation(inference_results.inducing_points,f"{parent_folder}/inducing_points_{iteration}.gif")
        ## save to wandb
        run.log({f"inducing points {iteration}": wandb.Video(f"{parent_folder}/inducing_points_{iteration}.gif", fps=4, format="gif")})

    def sub_experiment(self,key, train_loader: DataLoader, test_loader: DataLoader, n_basis: int ) :
      

        ############################## Linear model #################################
        # model 0, Linear model, change to new model svgp_model_v2
        SM = VariationalSpatialModelv3(key,name='HenSManSVGP' ,
                                    sigma=None,
                                    log_likelihood=False
                                    )
        # pdb.set_trace()
        inference_results =  SM.inference(train_loader,
                                            test_loader,
                                            n_steps=self.n_epochs,
                                            vi_method=self.vi_method,
                                            num_particles=self.num_particles,
                                            n_data=len(train_loader.dataset),
                                            n_basis=n_basis,
                                            stepsize=self.stepsize,
                                            covar_matrix= None)
        fig_posterior = None
        if self.plot_posterior:
            fig_posterior = SM.plot_posterior(None)
       
        predictive0 = SM.predictv2(test_loader,return_label=True,n_basis=n_basis)
        predictive0_0 = SM.predictv2(train_loader,return_label=True,n_basis=n_basis)
        
    
        return predictive0, predictive0_0, inference_results, fig_posterior
    

import numpy as np
from joblib import Parallel, delayed
from itertools import product
from visualization.contour_plot import plot_contour
class FullVariationalExperiment(BaseExperiment):

    def __init__(self, models: List[str]=['linear'],photo_path: Optional[Union[str,List[str]]]=None,
                 spectral_path: Optional[str]=None,normalize_data: bool=False,
                 batchsize: int=128,n_epochs: int=20,vi_method: str='SVI',
                 num_particles: int=20,max_n_basis: int=5,feature_selection: bool=False,
                 distance_method: str='uLSIF',scale_method: str='concentration',beta_a:float=1.,beta_b:float=1.,
                 metric='MAPE',dataset:str='',step_basis=1,stepsize=1e-3) -> None:
        super().__init__(models)
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.normalize_data = normalize_data
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.vi_method = vi_method
        self.num_particles = num_particles
        self.max_n_basis = max_n_basis
        self.feature_selection = feature_selection
        self.distance_method = distance_method
        self.scale_method = scale_method
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.metric = metric
        self.dataset = dataset
        self.step_basis = step_basis
        self.stepsize = stepsize
        self.plot_posterior = False

       
    def one_experiment(self,randomseed: Optional[int64]=None,parent_folder: Optional[str]=None,
                        data_params: Optional[Dict[str, Union[ndarray, int, float]]]=None,
                        mcmc_args: Optional[Dict[str, Optional[Union[int, str, bool, float]]]]=None,
                        config_dict: Optional[Dict[str, Any]]=None,iteration: int=0,run: Optional[Run]=None) -> None:
        ''' In this experiment, we just compare the linear variaitonal models'''

        ## read data
        key = random.PRNGKey(randomseed)
        # data  = read_real_data(self.photo_path,self.spectral_path,random_state=randomseed,normalize=self.normalize_data)
         ### add some process needed
        process_data = ProcessPhotoData(randomseed, self.photo_path, self.spectral_path,
                                        batchsize=self.batchsize,normalize=self.normalize_data,
                                        feature_selection=self.feature_selection,
                                        beta_a=self.beta_a,beta_b=self.beta_b,dataset=self.dataset)
        
       
        total_err = {}
        _training_loss = {}
        _eval_loss = {}
        inducing_points=  {}
        weights = {}
        ## create _alpha rray
        _alpha = np.linspace(0.1,0.3,3)
        n_basis=1
        model_names = ['multivariate_linear_regression_model','HenSManSVGP']
        if self.distance_method == 'all':
            # _distance_methods = ['','neuralNCA','uLSIF','KMM','OT','NCA','euclidean']
            _distance_methods = ['','neuralNCA','uLSIF','KMM','OT','NCA']
        else:
            _distance_methods = [self.distance_method]
        
        
            ## using joblib
        def subprocess(a,d_method,model):
            data = process_data.load_data(alpha=a)
            train_loader = data[0]
            test_loader = data[1]
            print(f"Length of train data:{len(train_loader.dataset)}")
            print(f"Length of test data:{len(test_loader.dataset)}")

            cv_method = CovariateShiftMethod(key ,distance_method=d_method,scale_method=self.scale_method)
            ## compute weights
            train_weights = cv_method.compute_similarity(train_loader,test_loader)
            training = True
            ## add weights to ds
            train_loader.dataset.add_weights(train_weights)
            ## add weights to test ds
            test_loader.dataset.add_weights(jnp.ones(len(test_loader.dataset)))

            predictive0, predictive0_0, inference_results ,fig_posterior= self.sub_experiment(key, train_loader,test_loader, n_basis,training=training, model_name=model )
            predictive0,label0 = predictive0
            predictive0_0,label0_0 = predictive0_0
            # pdb.set_trace()
            ## post-process for output
            predictive0 = process_data.postprocess_data(predictive0)
            predictive0_0 = process_data.postprocess_data(predictive0_0)
            err1 = self.mileometer_dict[self.model_name[0]].measure({
                                        f'test GP :alpha {a}, distance method {d_method}, model {model} ': predictive0,
                                                            },process_data.postprocess_data(label0))
                
            err2 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'train GP :alpha {a}, distance method {d_method}, model {model} ': predictive0_0,
                                    },process_data.postprocess_data(label0_0))
            
            err3 = self.mileometer_dict[self.model_name[0]].measure({
                                    f'train GP BIC :alpha {a}, distance method {d_method}, model {model} ': inference_results.BIC,
                                    }, None)
            
            return err1, err2, err3, inference_results
            

        ## outside subprocess, parallel codes
        parallel_params = product(_alpha,_distance_methods,model_names)
        parallel_params = [i for i in parallel_params]
        # print(f"parallel params: {parallel_params}")
        results = Parallel(n_jobs=12)(delayed(subprocess)(a,d_method,model) for a,d_method,model in parallel_params)
        # results = [subprocess(a,d_method,model) for a,d_method,model in parallel_params]
        
        ## finish parallel code
        ## update result to the error dict
        for (err1,err2,err3,inference_results),config in zip(results,parallel_params):
            total_err.update(err1)
            total_err.update(err2)
            total_err.update(err3)
            name = ' '.join([str(i) for i in config])
            _training_loss[name + ' train loss'] = np.array(inference_results.train_loss).flatten()
            _eval_loss[name + ' validation loss'] = np.array(inference_results.test_loss).flatten()
            if inference_results.inducing_points is not None:
                inducing_points[name] = inference_results.inducing_points
            if inference_results.weights is not None:
                weights[name] = inference_results.weights
       ## log all err
        metrics = ['MSE','R2','MAPE']
        # plot_data = []
        for metric in metrics:
      
            run.log({i + metric:getattr(v,metric) for i,v in total_err.items()})

        ## plot inducing points
        for name, data in inducing_points.items():
            save_point_animation(data,f"{parent_folder}/inducing_points_{name}_{iteration}.gif")
            run.log({f"inducing points {name} {iteration}": wandb.Video(f"{parent_folder}/inducing_points_{name}_{iteration}.gif", fps=4, format="gif")})

        ## plot the weights
        for name, data in weights.items():
            ## plot the contour
            data = [np.array(a).squeeze() for a in data]
            data = np.vstack(data)
            prepare_data = {f"feature_{i}": data[:,i] for i in range(data.shape[1]) }
            plot_contour(f"{parent_folder}/weights_{name}_{iteration}.png",**prepare_data)
            run.log({f"weights {name} {iteration}": wandb.Image(f"{parent_folder}/weights_{name}_{iteration}.png")})



        xs = range(len(list(_training_loss.values())[0]))
        run.log({f"train loss {iteration}"  : wandb.plot.line_series(
                                                                xs=[i for i in xs],
                                                                ys=[[j for j in i] for i in _training_loss.values()],
                                                                keys=[name for name in _training_loss],
                                                                title="Train loss")})
        
        xs = range(len(list(_eval_loss.values())[0]))
        run.log({f"validation loss {iteration}" : wandb.plot.line_series(
                                                                xs=[i for i in xs],
                                                                ys=[[j for j in i] for i in _eval_loss.values()],
                                                                keys=[name for name in _eval_loss],
                                                                title="Validation loss")})

        pass

    def sub_experiment(self,key, train_loader: DataLoader, test_loader: DataLoader, n_basis: int , training=True, model_name='HenSManSVGP') :
      

        ############################## Linear model #################################
        # model 0, Linear model, change to new model svgp_model_v2
        if model_name=='HenSManSVGP':
            SM = VariationalSpatialModelv3(key,name= model_name ,
                                            sigma=None,
                                            log_likelihood=False
                                            )
            n_epochs = self.n_epochs
        elif model_name=='multivariate_linear_regression_model':
            SM = VariationalLinearModelv2(key,name=model_name,
                                    sigma=None,
                                    log_likelihood=False
                                    )
            n_epochs = self.n_epochs
        # pdb.set_trace()
        
        inference_results =  SM.inference( 
                                            train_loader,
                                            test_loader,
                                            n_steps=n_epochs,
                                            vi_method=self.vi_method,
                                            num_particles=self.num_particles,
                                            n_data=len(train_loader.dataset),
                                            n_basis=n_basis,
                                            stepsize=self.stepsize,
                                            covar_matrix= None,
                                            training=training
                                            )
        ## save model
        config = {
            'n_basis':n_basis,
            'vi_method':self.vi_method,
            'num_particles':self.num_particles,
            'n_data':len(train_loader.dataset),
            'stepsize':self.stepsize,
            'covar_matrix': None,
            'model_name': model_name,
            
        }
        
        # ## save model
        # saved_model_path =f"saved_models/{model_name}.pkl"
        # SM.save_model(saved_model_path)
        # if run is not None:
        #     model_artifact = wandb.Artifact(
        #                                 model_name, type="model",
        #                                 description="Simple AlexNet style CNN",
        #                                 metadata=dict(config))
        #     ## add to wandb
        #     model_artifact.add_file(saved_model_path)
        #     wandb.save(f"{model_name}.pkl")
        #     run.log_artifact(model_artifact)

        # log_loss,eval_loss = inference_results.train_loss, inference_results.test_loss
        ## make animation for inducing points

        fig_posterior = None
        if self.plot_posterior:
            fig_posterior = SM.plot_posterior(None)
       
        predictive0 = SM.predictv2(test_loader,return_label=True,n_basis=n_basis)
        predictive0_0 = SM.predictv2(train_loader,return_label=True,n_basis=n_basis)
        
    
        return predictive0, predictive0_0, inference_results, fig_posterior
