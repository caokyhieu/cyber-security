from torch.utils.data import Dataset,DataLoader
import numpy as np
import jax.numpy as jnp
## Need higher class to hold data and processing
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Any, List, Tuple, Optional, Union 
import torch.utils.data.dataloader
from jax.interpreters.partial_eval import DynamicJaxprTracer
# from jaxlib.xla_extension import DeviceArray
from numpy import float64, ndarray
import pdb
from sklearn.model_selection import train_test_split

Stats = namedtuple('Stats', ['mean', 'std'])
EPS = 1e-5

class RunningStats:
    '''
    This stats can be updated realtime 
    '''

    def __init__(self):
        self.mean:Optional[float] = None
        self.s:Optional[float] = None 
        self.count:int = 0
        pass 
    
    def update(self, obs):
        '''
        This function to update the mean and std of the data
        Args:
            obs: numpy array (n x d)
                data
        '''
        obs = np.asleast_2d(obs)
        if self.mean is None:
            self.mean = np.zeros((1,) + obs.shape[1:])
            self.std = np.zeros((1,) + obs.shape[1:])
        self.count += 1
        delta = obs - self.mean
        self.mean += delta/self.count
        delta2 = obs - self.mean
        self.s += delta*delta2
    
    def get_mean(self):
        return self.mean
    
    def get_std(self):
        return np.sqrt(self.s/(self.count-1))
    
    def get_variance(self):
        return self.s/(self.count-1)
    
    def set_mean(self, mean):
        self.mean = mean
        pass 
    
    def set_s(self, s):
        self.s = s
        pass

class ProcessingData(metaclass=ABCMeta):

    def __init__(self, randomseed:int, path:str, batchsize:Optional[int]):

        self.path = path
        self.batchsize = batchsize
        self.randomseed = randomseed
    @abstractmethod
    def load_data(self,path:str):
        '''
        This function to load data from the path
        Args:
            path: str
                path of the data
        Return:
            data: array
                data of the path
        '''
        
        pass
    @abstractmethod
    def preprocess_data(self, train_data:tuple,test_data:tuple):
        '''
        This method helps to preprocess data before loading to the model
        '''

        pass     
    
    @abstractmethod
    def postprocess_data(self, data:ndarray):
        '''
        This method helps to process the output from the models before 
        saving to the disk
        '''

        pass 


def convert_ra_dec_to_certesian(ra_dec):
    '''
    This function helps to convert ra/dec to cartesian coordinates
    Args:
        ra_dec: numpy array (n x 2)
            ra and dec
    Return:
        cartesian: numpy array (n x 3)
            cartesian coordinates
    '''
    ra = ra_dec[:,0]
    dec = ra_dec[:,1]
    x = np.cos(np.radians(ra))*np.cos(np.radians(dec))
    y = np.sin(np.radians(ra))*np.cos(np.radians(dec))
    z = np.sin(np.radians(dec))
    return np.stack((x,y,z),axis=-1)

class ProcessPhotoData(ProcessingData):

    def __init__(self, randomseed:int, 
                 photo_path:str, 
                 spectral_path:str, 
                 batchsize:Optional[int]=None, 
                 normalize:bool=False,
                 feature_selection:str='',
                 beta_a:float=1.,
                 beta_b:float=1.,
                 dataset:str='freeman') -> None:
        
        self.photo_path = photo_path
        self.spectral_path = spectral_path
        self.batchsize = batchsize
        self.randomseed = randomseed
        self.normalize = normalize
        self.stats:dict = {}
        self.feature_selection = feature_selection
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.dataset = dataset
        pass

    def load_data(self, alpha=0.2) -> Tuple[    torch.utils.data.dataloader.DataLoader,     torch.utils.data.dataloader.DataLoader]:
        '''
        This method with prepocessing data and return train and test data
        It can be the tuple, dataloader or a numpy array etc.
        '''
        ## import here to make the code more nested
        from utils.pyro_utils import read_real_data
        from utils.split_data import read_data_selection_bias
        from utils.uci_dataset import (
                                        WineDataset, CrimeDataset,
                                        ConcreteDataset, ParkinsonDataset,
                                        AirfoilDataset, IndoorLocationDataset,
                                        SkyServerDataset,SimulatedDataset,
                                        SkyServerDatasetv2,
                                        sync_biased_data,sync_biased_data_skyserver,
                                        sync_biased_data_smooth, sync_biased_data_adjusted,
                                        sync_biased_data_peak_reduce, sync_biased_data_with_kmeans
                                       )
    
        ## helper function should return two tuples, one for train and one for test data
        # data = read_real_data(self.photo_path,self.spectral_path,
        #                       random_state=self.randomseed,normalize=False)
        
        ## new selection bias
        ## select dataset
        if self.dataset == 'freeman':
            data = read_data_selection_bias(self.photo_path,self.spectral_path,a=self.beta_a,b=self.beta_b,normalize=False)
        elif self.dataset == 'wine':
            data_obj = WineDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'crime':
            data_obj = CrimeDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'concrete':
            data_obj = ConcreteDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'parkinson':
            data_obj = ParkinsonDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'airfoil':
            data_obj = AirfoilDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'indoor':
            data_obj = IndoorLocationDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha)
        elif self.dataset == 'skyserver':
            data_obj = SkyServerDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data(spatial=True)
            ## set hyper parameters for split data
            # data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha,spatial_column=data_obj.features[-2:],drop=True)
            # data = sync_biased_data_adjusted(data,label=data_obj.label,alpha=alpha,spatial_column=data_obj.features[-2:],drop=True)
            
            # data = sync_biased_data_peak_reduce(data,label=data_obj.label,alpha=30.,spatial_column=data_obj.features[-2:],drop=True)
            # data = sync_biased_data_skyserver(data,label=data_obj.label,n_source=None,alpha=0.3,spatial_column=['dec','ra'],drop=True,labelshift=False)
            
            data = sync_biased_data_skyserver(data,label=data_obj.label,n_source=int(0.5 * len(data)),alpha=3.5,
                                  spatial_column=['dec','ra'],drop=True,labelshift=False,
                                  min_fraction=0.5, initial_buffer=0.4)  ## alpha=2.,5. ## al

            # data = sync_biased_data_skyserver(data,label=data_obj.label,n_source=10000,alpha=alpha,spatial_column=['dec','ra'],drop=True,labelshift=False)
            ## convert to carterisan
            spatial_train = convert_ra_dec_to_certesian(data[0][1])
            spatial_test = convert_ra_dec_to_certesian(data[1][1])
            ## recreate the tuple
            data = (data[0][0],spatial_train,data[0][2]), (data[1][0],spatial_test,data[1][2])

        elif self.dataset == 'skyserver_2':
            data_obj = SkyServerDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data(spatial=True)
            data = sync_biased_data_with_kmeans(data,label=data_obj.label,alpha=alpha,spatial_column=data_obj.features[-2:],drop=True)
            ## change from ra/dec to cartesian
            ## spatial train
        elif self.dataset == 'simulated':
            data_obj = SimulatedDataset(self.photo_path,self.spectral_path)
            data = data_obj.load_data()
            ## set hyper parameters for split data
            # data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha,spatial_column=data_obj.features[-2:],drop=True)
            data = sync_biased_data(data,label=data_obj.label,n_source=None,alpha=alpha,spatial_column=['dec','ra'],drop=True)

        else:
            raise ValueError('Not implemented for this dataset')
        

        ## get train and test data, here they are tuples of (covariates, spatial coordinates, labels)
        train_data = data[0]
        test_data = data[1]
        ## this one for preprocessing
        train_data, test_data = self.preprocess_data(train_data,test_data)
        
        ## check to put data to dataloader
        ## first, put these data to Photodataset
        if self.batchsize is not None:
            train_data = PhotoDataset(*train_data)
            test_data = PhotoDataset(*test_data)
            ## then, put these data to dataloader
            train_data = DataLoader(train_data,batch_size=self.batchsize,shuffle=True,collate_fn=jax_collate_fn,drop_last=False)
            test_data = DataLoader(test_data,batch_size=self.batchsize,shuffle=False,collate_fn=jax_collate_fn)
            # test_data = DataLoader(test_data,batch_size=len(test_data),shuffle=False,collate_fn=jax_collate_fn)
        
        ## assign these data to the class
        
        return train_data,test_data

    def calculate_stats(self,data:Optional[ndarray]) -> Optional[Stats]:
        '''
        This function to calculate the mean and std of the data
        Args:
            data: numpy array (n x d)
                data
        Return:
            mean: numpy array (1 x d)
                mean of the data
            std: numpy array (1 x d)
                std of the data
        '''
        if data is not None:
            mean = np.mean(data,axis=0)
            std = np.std(data,axis=0)
            return Stats(mean=mean, std=std)
        else:
            return None
        
    def apply_stats(self,data:ndarray,stats:Optional[Stats]) -> ndarray:
        '''
        This function to apply the mean and std to the data
        Args:
            data: numpy array (n x d)
                data
            stats: namedtuple
                mean and std of the data
        Return:
            data: numpy array (n x d)
                data after applying mean and std
        '''
        if stats is not None:
            data = (data - stats.mean)/(stats.std + EPS)
        
        return data
       
    
    def inverse_stats(self,data:ndarray,stats:Optional[Stats]) -> ndarray:
        '''
        This method helps to inverse the data to the original coordinate
        '''
        if stats is not None:
            data = data*(stats.std + EPS) + stats.mean
        
        return data
        
    
    def preprocess_data(self, train_data:tuple, test_data:tuple,split_val:bool=False) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        '''
        This method helps to preprocess data before loading to the model
        by calculating the mean and the standard deviation of training data and 
        applying to both train and test data

        Args:
            train_data: (tuples): calcuating mean and std for any arr in this tuple 
            test_data: (tuples): apply mean and std to the test data
            These tuples have specific meanings (covariates, spatial coordinates, labels)
        '''
        ## check dim data, should change to atleast_2d func
        if isinstance(train_data,tuple) and isinstance(test_data,tuple):
            train_data = list(train_data)
            test_data = list(test_data)
        train_data[0] = np.atleast_2d(train_data[0])
        test_data[0] = np.atleast_2d(test_data[0])
        
        ## TODO: complete this function 
        if split_val:
            temp_data = train_test_split(*train_data, test_size=0.2)
            train_data = temp_data[::2]
            val_data = temp_data[1::2]


        ## feature selection
        if self.feature_selection:
            if self.feature_selection == 'MMD':
                from utils.feature_selection import MMDBasedFeatureSelection
                feature_selection = MMDBasedFeatureSelection(projected_dim=train_data[0].shape[-1])
                feature_selection.fit(train_data[0], test_data[0])
            elif self.feature_selection == 'propensity':
                from propensity.propensity_score_matching import PropensityScoreMaching
                feature_selection = PropensityScoreMaching()
                source_label = np.zeros((len(train_data[0]),))
                target_label = np.ones((len(test_data[0]),))
                indexes = feature_selection.get_similar_index(train_data[0],source_label,test_data[0],target_label)
            else:
                from utils.feature_selection import _get_feature_selection
                feature_selection = _get_feature_selection(self.feature_selection)
                chosen_features,score = feature_selection.fit(train_data[0], train_data[2])
                print(f'Best features: {chosen_features}, score: {score:.2f}')
            ## asign for best features
            if self.feature_selection != 'propensity':
                train_data[0] = feature_selection.transform(train_data[0])
                test_data[0] = feature_selection.transform(test_data[0])
            else:
                train_data = [t[indexes] for t in train_data]

        if self.normalize:
            ## get stats for data
            self.stats['covariate'] = self.calculate_stats(train_data[0])
            self.stats['spatial'] = self.calculate_stats(train_data[1])
            # self.stats['label'] = self.calculate_stats(train_data[-1])
            ## normalize data
            train_data[0] = self.apply_stats(train_data[0],self.stats['covariate'])
            train_data[1] = self.apply_stats(train_data[1],self.stats['spatial'])
            # train_data[-1] = self.apply_stats(train_data[-1],self.stats['label'])
            test_data[0] = self.apply_stats(test_data[0],self.stats['covariate'])
            test_data[1] = self.apply_stats(test_data[1],self.stats['spatial'])
            # test_data[-1] = self.apply_stats(test_data[-1],self.stats['label'])
        
        if train_data[1] is not None:
            return_data =  (train_data[0],train_data[1],train_data[2]), (test_data[0],test_data[1],test_data[2])
        else:
            return_data = (train_data[0],train_data[2]), (test_data[0],test_data[2])
        
        if split_val:
            return return_data + (split_val,)
        else:
            return return_data
    
    def postprocess_data(self, prediction:ndarray):
        '''Postpreprocess for the output of the model trained'''
        if self.normalize:
            if 'label' in self.stats:
                prediction = self.inverse_stats(prediction,self.stats['label'])
        return prediction
    

class PhotoDataset(Dataset):

    def __init__(self,*args,**kwargs) -> None:

        super().__init__()
        self.args = args
        self.kwargs = kwargs
        if len(self.args)>=1:
            self.n_data = len(self.args[0])
            self.features = [i for i in range(self.args[0].shape[-1])]
        elif len(self.kwargs)>=1:
            self.n_data = len(self.kwargs[list(self.kwargs.keys())[0]])
            self.features = [i for i in range(self.kwargs[list(self.kwargs.keys())[0]].shape[-1])]
        else:
            raise ValueError("At least one dataset must be provided")
        
        for dataset in self.args:
            assert len(dataset)==self.n_data,"All datasets must have the same length"
        for key in self.kwargs:
            assert len(self.kwargs[key])==self.n_data,"All datasets must have the same length"

        self.return_index= False
    
    def __len__(self) -> int:
        return self.n_data
    
    def set_return_index(self,return_index:bool) -> None:
        self.return_index = return_index
        pass
    
    def __getitem__(self,idx: int) -> Union[DynamicJaxprTracer, List[Union[ndarray, DynamicJaxprTracer, float64]], List[Union[ndarray, float64]], List[DynamicJaxprTracer]]:
        args = None
        kwargs = None
        if len(self.args)==1:
            args =  self.args[0][idx]
        elif len(self.args)>1:
            args = [self.args[0][idx][self.features,...]] + [ self.args[i][idx] for i in range(1,len(self.args))]
        if len(self.kwargs)>=1:
            kwargs = {list(self.kwargs.keys())[0]:self.kwargs[list(self.kwargs.keys())[0]][idx][self.features,...] }
            kwargs.update({key:self.kwargs[key][idx] for key in list(self.kwargs.keys())[1:]})
        
        if args is None:
            if self.return_index:
                kwargs.update({'index':idx})
            return kwargs
        elif kwargs is None:
            if self.return_index:
                args = args + [idx]
            return args
        else:
            if self.return_index:
                kwargs.update({'index':idx})
            return args,kwargs
        
    def add_weights(self, weights) -> None:
        '''
        Need to change the argument from weights to distance
        '''
        if len(self.kwargs) ==0:
            assert len(weights)==len(self.args[0]), f'weight should have same length as data, but {len(weights)}!={len(self.args[0])}'
            self.args = (self.args[0],) + (weights,) + self.args[1:] ## hardcode ordering
        elif len(self.args)==0:
            assert len(weights) == len(self.kwargs[len(self.kwargs.keys())[0]]), f'weight should have same length as data, but {len(weights)}!={len(self.kwargs[len(self.kwargs.keys())[0]])}'
            self.kwargs['weights'] = weights 
        else:
            raise ValueError('Not implemented')
        
    def remove_weights(self):
        """Remove weights"""
        if len(self.kwargs)==0:
            self.args = (self.args[0],) + self.args[2:] ## hardcode ordering
        elif len(self.args)==0:
            self.kwargs.remove('weights')
        else:
            raise ValueError('Not implemented')

    def update_covariates(self,features):
        '''
        Selecgted features
        '''
        self.features = features
        pass
        
    def get_covariates(self):
        if len(self.kwargs)==0:
            return jnp.array(self.args[0][:,self.features,...])
        elif len(self.args)==0:
            return jnp.array(self.kwargs[len(self.kwargs.keys())[0]][:,self.features,...])
        else:
            raise ValueError('Not implemented')
    
    def get_labels(self) -> ndarray:
        if len(self.kwargs)==0:
            return self.args[-1]
        elif len(self.args)==0:
            return self.kwargs[len(self.kwargs.keys())[-1]]
        else:
            raise ValueError('Not implemented')
    
    def get_weights(self):
        if len(self.kwargs)==0:
            return self.args[1]
        elif len(self.args)==0:
            return self.kwargs[len(self.kwargs.keys())[1]]
        else:
            raise ValueError('Not implemented')
        
    def get_spatial(self):
        if len(self.kwargs)==0:
            return self.args[2]
        elif len(self.args)==0:
            return self.kwargs[len(self.kwargs.keys())[2]]
        else:
            raise ValueError('Not implemented')
        
    def add_data(self, data):
        '''
        This method help to add more data to the dataset, this will be after the last index(spatial at [2]) and before the labels [-1]
        '''
        ## first check length of added data, if smaller than the length of current data, circle to equal
        len_data = len(data)
        if len_data < self.n_data:
            n_time = self.n_data // len_data
            idx_data = np.tile(np.arange(len_data), n_time)
            remained = self.n_data - len_data * n_time 
            if remained > 0:
                idx_data = np.concatenate((idx_data, np.random.choice(np.arange(len_data), remained, replace=False)))
            data = data[idx_data]
        elif len_data > self.n_data:
            data = data[:self.n_data]
            
        if len(self.kwargs) ==0:
            self.args = self.args[0:-1] + (data,) + (self.args[-1],) ## hardcode ordering
        elif len(self.args)==0:
            self.kwargs['added_data'] = data
        else:
            raise ValueError('Not implemented')
        
    def remove_data(self):
        """Remove added data"""
        if len(self.kwargs)==0:
            self.args = self.args[:-2] + (self.args[-1],) ## hardcode ordering
        elif len(self.args)==0:
            self.kwargs.remove('added_data')
        else:
            raise ValueError('Not implemented')

    
def jax_collate_fn(batch: Any):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [jax_collate_fn(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: jax_collate_fn([d[key] for d in batch]) for key in batch[0]}
    else:
        return jnp.array(batch)