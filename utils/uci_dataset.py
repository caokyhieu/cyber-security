'''
This module used to load UCI datasets: use paper "Robust Covariate Shift Regression" as referece.

This module helps to construct the biased data
'''

import pandas as pd
import numpy as np
from typing import List, Tuple, Union,Dict
from scipy.stats import multivariate_normal,dirichlet,multinomial
import re
import pandas as pd
from abc import ABCMeta,abstractmethod
import pdb
import logging as logger  # Handling warnings
logger.getLogger().setLevel(logger.INFO)
def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    
    # 📏 Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # 🔄 Iterate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    # ℹ️ Provide memory optimization information if 'verbose' is True
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    # 🔄 Return the DataFrame with optimized memory usage
    return df
class UCIDataset(metaclass=ABCMeta):
    '''
    Abstract class to load UCI dataset into a pandas DataFrame.

    Args:
        data_path (str): The path to the data file.
        names_path (str): The path to the names file.
    '''

    def __init__(self, data_path:str, names_path:str):
        '''
        Initializes the UCIDataset object with the paths to the data file and the names file.

        Args:
            data_path (str): The path to the data file.
            names_path (str): The path to the names file.
        '''
        self.data_path = data_path
        self.names_path = names_path

    @abstractmethod
    def get_feature_names(self):
        '''
        Abstract method that should be implemented in the subclass to define the feature names.
        '''
        pass 

    @abstractmethod
    def load_data(self):
        '''
        Abstract method that should be implemented in the subclass to load the data into a DataFrame.
        '''
        pass 
    
    def preprocess_data(self, data:pd.DataFrame, features:List[str]):
        '''
        Preprocesses the data by filling missing values and removing non-numeric features.

        Args:
            data (pd.DataFrame): The data to be preprocessed.
            features (List[str]): The list of feature names.

        Returns:
            Tuple[pd.DataFrame, List[str]]: The preprocessed data and the selected feature names.
        '''
        threshold = 30
        drop_cols = data[features].isna().apply(lambda x: sum(x)/len(x) *100)>threshold
        selected_features = drop_cols[~drop_cols].index.tolist()
        ## sorted according old features order
        selected_features = sorted(selected_features, key=features.index)
        
        data[selected_features] = data[selected_features].fillna(method='ffill')

        selected_features = [col for col in selected_features if data[col].dtype in ['float32','int32','int8','float','float32']]
        
        selected_features = list(set(selected_features).difference(set(['communityname','state','countyCode','communityCode','fold'])))
        
        return data[selected_features + [self.label]], selected_features



class CrimeDataset(UCIDataset):

    def get_feature_names(self):
        '''
        function to get the feature names
        '''
        with open(self.names_path,'r') as f:
            doc = f.readlines()
        
        feature_names:Dict[str,str] = {}
        for line in doc:
            if re.match("^@attribute.*",line):
                line = line.strip().split()
                if line[2] == 'numeric':
                    feature_names[line[1]] = np.float32
                elif line[2] == 'string':
                    feature_names[line[1]] = str
                else:
                    raise ValueError('Unknown data type')
        
        self.label = 'ViolentCrimesPerPop'
        return feature_names
    
    def load_data(self):
        '''
        function to load data
        '''
        ## get features
        feature_names = self.get_feature_names()
        ## load data
        df = pd.read_csv(self.data_path,names=list(feature_names.keys()),na_values='?',dtype=feature_names)
        ## fill na
        df,self.features = self.preprocess_data(df,features=list(set(feature_names.keys()).difference(set([self.label]))))
                
        return df
    
class SkyServerDataset(UCIDataset):

    def get_feature_names(self):
        self.label = 'redshift'
    
    def load_data(self,spatial=True):
        df = pd.read_csv(self.data_path,skiprows=1)
        # pdb.set_trace()
        ## here we need to filter the data  where the redshift in range (0,1)
        original_len = len(df)
        ## if filter data by this condition, the linear regression become too good
        columns_to_check = ['u', 'g', 'r', 'i', 'z'] ## chek non negative magnitude
        df = df[(df[columns_to_check] >= 0).all(axis=1)] 

        df = df[(df['redshift'] > 0) & (df['redshift'] < 1.3)]
        # ## sample 4 %
        df = df.sample(frac=.006)
        ## open index
        # pdb.set_trace()
        # with open('index.npy', 'rb') as f:
        #     index = np.load(f,allow_pickle=True)
        
        # df = df[df.index.isin(index.tolist())]
        ## reduce length
        reduced_len = len(df)
        print(f"length of data reduced from {original_len} to {reduced_len} after filtering redshift in range 0-1.3")
        # df = df.drop(columns=['ra','dec'])
        ## reduce memory usage
        # df = reduce_mem_usage(df,verbose=True)
        ## get label
        self.get_feature_names()
        feature_names = df.columns.tolist()
        ## remove label
        feature_names.remove(self.label)
        ## remove 'objid'
        feature_names.remove('objid')
        if not spatial:
            feature_names.remove('ra')
            feature_names.remove('dec')
        ## fill na
        df,self.features = self.preprocess_data(df,features=feature_names)
        # ## now change to colors (difference magnitude between bands) and take 'r' as reference
        # df[['u-g', 'g-r', 'r-i',  'i-z']] = df[['u', 'g', 'r', 'i' ]].values - df[[ 'g', 'r', 'i', 'z']].values
        # ## remove u, g, z, i from feature names and add 'u-g', 'g-r', 'r-i', 'i-z'
        # feature_names.remove('u')
        # feature_names.remove('g')
        # feature_names.remove('i')
        # feature_names.remove('z')
        # feature_names.append('u-g')
        # feature_names.append('g-r')
        # feature_names.append('r-i')
        # feature_names.append('i-z')
        
        ## reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
       
    
from sklearn.covariance import EllipticEnvelope
class SkyServerDatasetv2(UCIDataset):

    def get_feature_names(self):
        self.label = 'redshift'
    
    def load_data(self,spatial=True):
        df = pd.read_csv(self.data_path,skiprows=1)
        ## first oulier elimination
        clf = EllipticEnvelope().fit(df[['u','g','r','i','z']])
        labels = clf.predict(df[['u','g','r','i','z']])
        df = df[labels == 1]
        
        ## get label
        self.get_feature_names()
        feature_names = df.columns.tolist()
        ## remove label
        feature_names.remove(self.label)
        ## remove 'objid'
        feature_names.remove('objid')
        if not spatial:
            feature_names.remove('ra')
            feature_names.remove('dec')
        ## fill na
        df,self.features = self.preprocess_data(df,features=feature_names)
        return df

class ParkinsonDataset(UCIDataset):

    def get_feature_names(self):
        self.label='total_UPDRS'
        return None
    
    def load_data(self):
        df = pd.read_csv(self.data_path)
        ## get label
        self.get_feature_names()
        feature_names = df.columns.tolist()
        ## fill na
        df,self.features = self.preprocess_data(df,features=list(set(feature_names).difference(set([self.label]))))
        return df
    
class WineDataset(UCIDataset):

    def __init__(self, data_paths:List[str], names_path:str):
        self.data_paths = data_paths
        self.names_path = names_path
    
    def get_feature_names(self):
        self.label='quality'
        return None
    
    def load_data(self):
        return [pd.read_csv(path,sep=';') for path in self.data_paths]
    
class ConcreteDataset(UCIDataset):
    
    def get_feature_names(self):
        self.label = 'Concrete compressive strength(MPa, megapascals) '
        return None
    
    def load_data(self):
        self.get_feature_names()
        df = pd.read_excel(self.data_path)
        return df

class AirfoilDataset(UCIDataset):

    def get_feature_names(self):
        '''
        1. Frequency, in Hertzs. 
        2. Angle of attack, in degrees. 
        3. Chord length, in meters.
        4. Free-stream velocity, in meters per second. 
        5. Suction side displacement thickness, in meters. 

        The only output is:
        6. Scaled sound pressure level, in decibels. 
        
        '''
        self.label = 'sound_pressure(db)'
        return  ['freq(Hz)','angle_attack(deg)','chord_length(m)','velocity(m/s)','suction_displacement(m)','sound_pressure(db)']
    
    def load_data(self):
        df = pd.read_csv(self.data_path,sep='\t',names=self.get_feature_names())
        df,self.features = self.preprocess_data(df,features=list(set(self.get_feature_names()).difference(set([self.label]))))
        return df

from sklearn.gaussian_process.kernels import Matern
from sklearn.mixture import GaussianMixture

class SimulatedDataset(UCIDataset):
    N1 = 500
    N2 = 500
    dim = 5 
    m1 = np.array([19.,22.5,22.,19.,23.])
    m2 = np.array([17.5,19.,17.5,15.,17.])
    sig1 = 2.
    sig2 = 1.
    spatial_dim= 2
    spatial_mean = np.array([5.,6.])
    weights = np.array([5.,2.,3.,-2,-4]).reshape(5,1)

    def get_feature_names(self):
        pass
    
    def load_data(self,spatial=True):
        ## simulation 
        source_data = np.random.normal(SimulatedDataset.m1,SimulatedDataset.sig1,size=(SimulatedDataset.N1,SimulatedDataset.dim))
        target_data = np.random.normal(SimulatedDataset.m2,SimulatedDataset.sig2,size=(SimulatedDataset.N2,SimulatedDataset.dim))
        # spatial =  np.random.normal(SimulatedDataset.spatial_mean,3.,size=(SimulatedDataset.N2+SimulatedDataset.N1 ,SimulatedDataset.spatial_dim))
        mixture = GaussianMixture(n_components=2,covariance_type='full')
        mixture.fit(np.random.rand(10, 2))
        mixture.weights_ = np.array([0.5, 0.5])
        mixture.means_ = np.array([[-1,2.],[-1.2,2.5]])
        # mixture.weights_ = np.array([0.5,0.5])
        mixture.covariances_ = np.array([np.diag([1.,1.]),np.diag([1.,1.])])
        
        spatial,label_spatial = mixture.sample(n_samples=SimulatedDataset.N2+SimulatedDataset.N1)
        # print(set(label_spatial))
        # pdb.set_trace()
        kernel =  2. * Matern(length_scale=.8, nu=1.5)
        # pdb.set_trace()
        covar =  kernel(spatial,spatial)
        mean_func = lambda x: (x @ SimulatedDataset.weights).flatten()
        data = np.concatenate((source_data,target_data),axis=0)
        mean = mean_func(data)
        label = np.random.multivariate_normal(mean, covar) 
        # diag = np.diag(np.sqrt(1/np.diag(covar)))
        # corr = diag @ covar @ diag

        noise = 0.2 * np.random.normal(size=(SimulatedDataset.N2+SimulatedDataset.N1))
        label +=noise
        data = np.concatenate((data,spatial,label.reshape(-1,1)),axis=-1)

        df = pd.DataFrame(data,columns=['1','2','3','4','5','ra','dec','redshift'])
        self.label = 'redshift'
        self.features = ['1','2','3','4','5','ra','dec']
        # plt.hist(covar.flatten(),bins=50)
        
        return df

class IndoorLocationDataset(UCIDataset):

    def get_feature_names(self):
        return ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','room']
    
    def load_data(self):
        return pd.read_csv(self.data_path,sep='\t',names=self.get_feature_names())
    
## need some modification
from collections import namedtuple
fields = ('train','val','test')
SyncBiasData = namedtuple('SyncBiasData',fields,defaults=((None,None,None),) * len(fields))
# def sync_biased_data(df:pd.DataFrame,label:str='label',n_source:int=1000,alpha:float=0.1,spatial_column=[],drop=True, labelshift=False):
#     '''
#     function to create biased source and target dataset

#     Args:
#         df: pandas dataframe, the dataframe contains the source and target data
#         label: str, the label column name
#         n_source: int, the number of source data
#         alpha: float, the bias parameter
#     returns:
#         source_df: pandas dataframe, the biased source data
#         target_df: pandas dataframe, the biased target data
#     ## todo: need to split to get the validation set
#     '''

#     ## feature columns
#     if drop:
#         feature_cols = [col for col in df.columns if col not in  [label] + spatial_column]
#     else:
#         feature_cols = [col for col in df.columns if col not in  [label]]

#     if labelshift:
#         feature_cols += [label]
    
#     ## split df randomly and evenly into two disjoint datasets
#     df1,df2 = np.array_split(df.sample(frac=1),[int(0.3 * len(df))])
#     ## just select 50% of test data
#     if n_source is None:
#         n_source = int(min(len(df1),len(df1) * 2 * alpha))

#     if n_source > len(df1):
#         n_source = int(min(len(df1),len(df1) * 2 * alpha))
    
#     # #     df2 = df2.sample(frac=0.5 * n_source/len(df1))
#     # # else:
#     # #     percent = n_source/len(df1)
#     # #     df2 = df2.sample(frac=0.5 * percent)

#     # ## sample mean of df1
#     # df1_mean = df1[feature_cols].mean().values
#     # df1_cov = df1[feature_cols].cov().values
#     # ## sample  x_seed
#     # ## we can change the way to sample x_seed to create bias
#     # ## by sampling uniform from the mean to the max
#     # # x_max = df1[feature_cols].max().values
#     # # x_min = df1[feature_cols].min().values
#     # ## sample x_seed from the mean to the max values
#     # ## select quantile 0.75 for x_min
#     # # x_min = np.array([np.quantile(df1[col].values,0.75) for col in feature_cols])
#     # # x_seed = np.random.uniform(x_min , x_max)
#     # # print(f"x max: {x_max}")
#     # # print(f"x min: {x_min}")
#     # # print(f"x seed: {x_seed}")
#     # # print(f"df1 cov: {df1_cov}")
#     # # x_seed = np.random.multivariate_normal(df1_mean,df1_cov,size=1)
#     # ## fix seed
#     # x_seed = np.array([19.,22.5,22.,19.,23., 0.2]) ## this is magic number
#     # # x_seed = np.array([19.,22.5,22.]) ## this is magic number
#     # # pdb.set_trace()
    
#     # ## sample  x_source from N(x_seed,lapha*Q)
#     # x_source_pdf = multivariate_normal.pdf(df1[feature_cols].values, mean=x_seed.flatten(), cov=alpha*df1_cov)
#     # ## deal with zeros entries
#     # eps = 1e-10
#     # x_source_pdf = x_source_pdf + eps 
#     # x_source_pdf = x_source_pdf/x_source_pdf.sum()
#     # ## using multinoimal distributions
#     # x_source_index = np.random.choice(len(df1),size=n_source,p=x_source_pdf,replace=True)
#     # x_source_index = list(set(x_source_index))

#     # print(f"{len(set(x_source_index))}/{len(df1)} datapoints selected in sources")
#     # ## fixed number of test set = 2000
#     # # df2 = df2.sample(n=2000)
#     # test_data = df1.iloc[x_source_index].sample(n=min(1000,len(x_source_index)))

#     ## with df1 we do that with the lower values according lower quantile, we will drop with higher prob
    
#     test_data = df1.copy()
#     for col in feature_cols:
#         # Calculate quantile values
#         quantiles = test_data[col].quantile([0.01, 0.05, 0.08])
        
#         # Iterate through each row in the column
#         for i in test_data.index:
#             value = test_data.at[i, col]
#             # Determine the drop probability based on quantiles
#             if value <= quantiles[0.01]:
#                 drop_prob = 0.99
#             elif value <= quantiles[0.05]:
#                 drop_prob = 0.95
#             elif value <= quantiles[0.08]:
#                 drop_prob = 0.9
#             # elif value <= quantiles[0.15]:
#             #     drop_prob = 0.8
#             else:
#                 drop_prob = 0

#             # Drop the sample with the specified probability
#             if np.random.rand() < drop_prob:
#                 test_data.drop(i, inplace=True)
    
#     # Reset index after dropping rows
#     test_data.reset_index(drop=True, inplace=True)
#     test_data = test_data.sample(frac=0.2)
    
#     print(f"{len(test_data)}/{len(df1)} datapoints selected in sources")


#     ## sample x_target from df2
#     ## supsample df2
#     ## for df2, i want to lop through the cols and drop the samples have high values
#     ## I want to drop with the prob that higher values will hinger percentage to drop
#     ## for example, if the value is 0.8 quantile, the prob to drop is 0.8
#     ## if the value is 0.9 quantile, the prob to drop is 0.9
#     ## if the value is 0.95 quantile, the prob to drop is 0.95
#     ## if the value is 0.99 quantile, the prob to drop is 0.99


#     for col in feature_cols:
#         # Calculate quantile values
#         quantiles = df2[col].quantile([0.9, 0.95, 0.99])
#         # Iterate through each row in the column
#         for i in df2.index:
#             value = df2.at[i, col]
#             # Determine the drop probability based on quantiles
#             if value >= quantiles[0.99]:
#                 drop_prob = 0.99
#             elif value >= quantiles[0.95]:
#                 drop_prob = 0.95
#             elif value >= quantiles[0.9]:
#                 drop_prob = 0.9
#             else:
#                 drop_prob = 0

#             if np.random.rand() < drop_prob:
#                 df2.drop(i, inplace=True)

#     df2.reset_index(drop=True, inplace=True)

#     ## truncated df2 and test_data in order to get the value in same range
    

#     ## pop label from feature_cols
#     feature_cols = [col for col in feature_cols if col != label]
#     ## sub sample both train and test data
#     if len(spatial_column) >0:
#         return (df2[feature_cols].values,df2[spatial_column].values,df2[label].values),\
#                 (test_data[feature_cols].values,
#                 test_data[spatial_column].values,
#                 test_data[label].values)
        
#         # return (np.random.normal(size=(2000,5)), np.random.normal(size=(2000,2)),np.random.normal(size=(2000,))),\
#         #         (np.random.normal(size=(1000,5)), np.random.normal(size=(1000,2)),np.random.normal(size=(1000,)))
#     else:
#         return (df2[feature_cols].values,None,df2[label].values),\
#                 (test_data[feature_cols].values,None,test_data[label].values)
    

from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.stats import norm, halfnorm, beta
def vectorized_stable_sigmoid( x):
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = np.zeros_like(x, dtype=np.float64)
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        
        return result
def sync_biased_data_skyserver_v2(df:pd.DataFrame,label: str = 'label', n_source: int = 1000, alpha: float = 0.1, 
                     spatial_column: list = [], drop: bool = True, labelshift: bool = False):
    '''
    This function just apply for df Skyserver
    Args:
    df: DataFrame with label and possibly spatial columns
    label: str, the column name of label
    n_source: int, the number of source data points
    alpha: float in (0,1) quantile to take
    '''
     # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]

    
    ## mean and std of data
    mean_ = np.mean(df[feature_cols].values,axis=0,keepdims=True)
    std_ = np.std(df[feature_cols].values,axis=0,keepdims=True)
    ## normalize the data
    normalize_data = (df[feature_cols].values - mean_)/std_
    ## create two means and two standard deviations for two clusters
    ## inorder that the weighted of these two distributions will be the normalized data
    ## with alpha will be the controled weighted
    ## init lower mean for cluster 0
    # mean_cluster0 = beta.rvs(3,1,loc=np.abs(np.mean(normalize_data,axis=0,keepdims=True)),scale=1., size=mean_.shape)
    mean_cluster0 = (1 - alpha) * np.mean(df[feature_cols].values,axis=0,keepdims=True) + alpha *  0.3 * np.ones_like(mean_)
    std_cluster0 = alpha * np.ones_like(std_)
    ## mean cluster 1 can be calculated by weighted sum
    mean_cluster1 = (np.mean(normalize_data,axis=0,keepdims=True) - alpha * mean_cluster0)/(1 - alpha)
    # std_cluster1 = np.sqrt((np.std(normalize_data,axis=0,keepdims=True) \
    #                             - alpha * std_cluster0**2 \
    #                         - alpha * mean_cluster0**2\
    #                         - (1 - alpha) * mean_cluster1**2\
    #                         + (alpha * mean_cluster0 + (1 - alpha) * mean_cluster1)**2)/ (1 - alpha))
    
    std_cluster1 = 1. *  np.ones_like(std_)
    
    ## now calculate the prob for each point of normalized data belong to the cluster
    prob_cluster0 = np.sum(norm.logpdf(normalize_data, mean_cluster0, std_cluster0),axis=-1)
    prob_cluster1 = np.sum(norm.logpdf(normalize_data, mean_cluster1, std_cluster1),axis=-1)
    ## use binomial distribution to seelct label
    prob = vectorized_stable_sigmoid(prob_cluster0 - prob_cluster1)

    cluster_labels = np.random.binomial(1, prob)
    # pdb.set_trace()

    
    # create two datasets
    df1_cluster0 = df[cluster_labels == 0]
    df1_cluster1 = df[cluster_labels == 1]

    ## the cluster 1 will be the source, and cluster 0 will be the target
    if n_source is None:
        n_source = len(df1_cluster1)
    if len(df1_cluster1) < n_source:
        n_source = len(df1_cluster1)

    
    # create two datasets
    df1_cluster0 = df[cluster_labels == 0]
    df1_cluster1 = df[cluster_labels == 1]

    source_data = df1_cluster1.sample(n=n_source)
    target_data = df1_cluster0.sample(n=len(df1_cluster0))

    # Determine the overlapping range for truncation
    for col in feature_cols:
        min_val = max(source_data[col].min(), target_data[col].min())
        max_val = min(source_data[col].max(), target_data[col].max())
        
        source_data = source_data[(source_data[col] >= min_val) & (source_data[col] <= max_val)]
        target_data = target_data[(target_data[col] >= min_val) & (target_data[col] <= max_val)]

    # Prepare the return values
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data
from sklearn.metrics import pairwise_distances
from scipy.special import softmax
def soft_kmeans_membership(X, n_clusters=2, temperature=1.0, random_state=0):
    """
    Perform KMeans clustering and return soft membership probabilities.
    
    Args:
        X: ndarray, feature matrix
        n_clusters: int, number of clusters
        temperature: float, controls softness of assignment
        random_state: int
    
    Returns:
        soft_probs: ndarray, shape (n_samples, n_clusters)
        labels: ndarray, hard cluster assignments
        model: fitted KMeans model
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=512, n_init='auto')
    kmeans.fit(X)
    distances = pairwise_distances(X, kmeans.cluster_centers_)  # shape (n_samples, n_clusters)
    soft_probs = softmax(-distances / temperature, axis=1)
    return soft_probs, kmeans.labels_, kmeans
def sync_biased_data_skyserver(df: pd.DataFrame, label: str = 'label', n_source: int = 1000, alpha: float = 0.1, 
                               spatial_column: list = [], drop: bool = True, labelshift: bool = False,
                               min_fraction: float = 0.5, initial_buffer: float = 0.05,temperature: float = 1.0):
    '''
    This function applies to Skyserver dataset.
    It splits the data using KMeans clustering and then truncates features based on overlapping ranges.
    Additional parameters:
      - min_fraction: minimum fraction of samples that must remain after truncation (default: 0.5)
      - initial_buffer: initial buffer percentage for expanding the overlapping range (default: 0.05)
    '''
    df = df.copy()
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]

    # # Define KMeans, use MiniBatchKMeans for stable
    # kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=512,n_init='auto')
    # kmeans.fit(df[feature_cols])
    # cluster_labels = kmeans.labels_

    # # Determine source and target clusters based on median values
    # median_mags = df.groupby(kmeans.labels_).median()
    # source_label = median_mags.mean(axis=1).idxmin()
    # target_label = 1 - source_label

    # # Create two datasets based on clusters
    # df1_cluster0 = df[cluster_labels == target_label]
    # df1_cluster1 = df[cluster_labels == source_label]
    # print(f"length source data: {len(df1_cluster0)}")
    # print(f"length target data: {len(df1_cluster1)}")
    # # Dynamically adjust n_source if needed
    # if n_source is None or n_source > len(df1_cluster1):
    #     n_source = min(len(df1_cluster1), int(0.7 * len(df1_cluster1)))

    # source_data = df1_cluster1.sample(n=n_source, replace=True)
    # target_data = df1_cluster0.sample(n=len(df1_cluster0))

    # Step 1: Soft KMeans clustering
    soft_probs, _, kmeans_model = soft_kmeans_membership(df[feature_cols].values, temperature=temperature)
    df['cluster_0_prob'] = soft_probs[:, 0]
    df['cluster_1_prob'] = soft_probs[:, 1]

    # Step 2: Choose source = brighter cluster (based on mean magnitude)
    mean_mags = df[['r', 'cluster_0_prob', 'cluster_1_prob']].copy()
    cluster0_mean_r = (mean_mags['r'] * mean_mags['cluster_0_prob']).sum() / mean_mags['cluster_0_prob'].sum()
    cluster1_mean_r = (mean_mags['r'] * mean_mags['cluster_1_prob']).sum() / mean_mags['cluster_1_prob'].sum()

    source_cluster = 0 if cluster0_mean_r < cluster1_mean_r else 1
    target_cluster = 1 - source_cluster

    # Step 3: Compute sampling weights
    source_weights = soft_probs[:, source_cluster] ** (1 / alpha)
    target_weights = soft_probs[:, target_cluster] ** (1 / alpha)

    source_weights /= source_weights.sum()
    target_weights /= target_weights.sum()

    # Step 4: Sample
    if n_source > len(df):
        n_source = int(0.5 * len(df))
    source_data = df.sample(n=n_source, weights=source_weights, random_state=42)
    remaining_df = df.drop(source_data.index)
    target_data = remaining_df.sample(frac=1.0, weights=target_weights[remaining_df.index], random_state=42)
    

    # Set minimum samples threshold
    min_samples_threshold = int(min_fraction * min(len(source_data), len(target_data)))

    # Truncation with dynamic range expansion
    # For each feature, try to filter the data but check if it would lead to an empty set
    for col in feature_cols:
        # Start with an initial buffer
        buffer = initial_buffer * (source_data[col].max() - source_data[col].min())
        # Compute overlapping range with buffer
        min_val = max(source_data[col].min(), target_data[col].min()) - buffer
        max_val = min(source_data[col].max(), target_data[col].max()) + buffer

        new_source = source_data[(source_data[col] >= min_val) & (source_data[col] <= max_val)]
        new_target = target_data[(target_data[col] >= min_val) & (target_data[col] <= max_val)]
        
        # If the new subsets have too few samples, try increasing the buffer
        while (len(new_source) < min_samples_threshold or len(new_target) < min_samples_threshold) and buffer < 0.5 * (source_data[col].max() - source_data[col].min()):
            buffer *= 1.5  # Increase the buffer by 50%
            min_val = max(source_data[col].min(), target_data[col].min()) - buffer
            max_val = min(source_data[col].max(), target_data[col].max()) + buffer
            new_source = source_data[(source_data[col] >= min_val) & (source_data[col] <= max_val)]
            new_target = target_data[(target_data[col] >= min_val) & (target_data[col] <= max_val)]
        
        # If after buffer expansion we still get too few samples, skip truncation for that feature
        if len(new_source) < min_samples_threshold or len(new_target) < min_samples_threshold:
            print(f"Warning: Skipping truncation for feature '{col}' due to insufficient overlap.")
        else:
            source_data, target_data = new_source, new_target

    # Prepare the return values by excluding the label from feature columns
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data




def sync_biased_data(df: pd.DataFrame, label: str = 'label', n_source: int = 1000, alpha: float = 0.1, 
                     spatial_column: list = [], drop: bool = True, labelshift: bool = False):
    '''
    Function to create biased source and target datasets with range truncation

    Args:
        df: pandas DataFrame, the dataframe contains the source and target data
        label: str, the label column name
        n_source: int, the number of source data
        alpha: float, the bias parameter
        spatial_column: list, list of spatial column names
        drop: bool, whether to drop spatial columns from features
        labelshift: bool, whether to include label in features

    Returns:
        source_data: tuple, biased source data (features, spatial columns, labels)
        target_data: tuple, biased target data (features, spatial columns, labels)
    '''
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]
    
    # Split df randomly into two disjoint datasets
    # df = df.sample(frac=0.5)
    df1, df2 = np.array_split(df.sample(frac=1), [int(0.5 * len(df))])

    # Adjust n_source if necessary
    if n_source is None or n_source > len(df1):
        n_source = int(min(len(df1), len(df1) * 2 * alpha))

    # Bias in the target dataset
    biased_target_data = df1.copy()
    for col in feature_cols:
        quantiles = biased_target_data[col].quantile([0.1, 0.4, 0.6])
        drop_probabilities = {0.1: 0.6, 0.4: 0.4, 0.6: 0.3}

        biased_target_data = biased_target_data.drop(
            biased_target_data.index[
                biased_target_data[col].apply(lambda x: np.random.rand() < drop_probabilities.get(
                    next((q for q in quantiles.index if x <= quantiles[q]), 0), 0))
            ]
        )

    biased_target_data.reset_index(drop=True, inplace=True)
    ## dont need
    # biased_target_data = biased_target_data.sample(frac=0.2)

    print(f"{len(biased_target_data)}/{len(df1)} datapoints selected in target")

    # Bias in the target dataset
    biased_source_data = df2.copy()
    for col in feature_cols:
        quantiles = biased_source_data[col].quantile([0.7, 0.8, 0.9])
        drop_probabilities = {0.7: 0.4, 0.8: 0.5,0.9:0.6}

        biased_source_data = biased_source_data.drop(
            biased_source_data.index[
                biased_source_data[col].apply(lambda x: np.random.rand() < drop_probabilities.get(
                    next((q for q in quantiles.index if x >= quantiles[q]), 0), 0))
            ]
        )
    
    ## sample bias source data
    biased_source_data = biased_source_data.sample(n=min(n_source, len(biased_source_data)))

    biased_source_data.reset_index(drop=True, inplace=True)

    # Determine the overlapping range for truncation
    for col in feature_cols:
        min_val = biased_source_data[col].min()
        max_val =biased_source_data[col].max()
        
        # biased_source_data = biased_source_data[(biased_source_data[col] >= min_val) & (biased_source_data[col] <= max_val)]
        biased_target_data = biased_target_data[(biased_target_data[col] >= min_val) & (biased_target_data[col] <= max_val)]

    # Prepare the return values
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (biased_source_data[feature_cols].values, 
                       biased_source_data[spatial_column].values, 
                       biased_source_data[label].values)
        
        target_data = (biased_target_data[feature_cols].values, 
                       biased_target_data[spatial_column].values, 
                       biased_target_data[label].values)
    else:
        source_data = (biased_source_data[feature_cols].values, None, biased_source_data[label].values)
        target_data = (biased_target_data[feature_cols].values, None, biased_target_data[label].values)

    return source_data, target_data



from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np


def sync_biased_data_smooth(df: pd.DataFrame, label: str = 'label', alpha: float = 4.0,
                           spatial_column: list = [], drop: bool = True, labelshift: bool = False,
                           kde_bandwidth: float = None):
    '''
    Function to create clearly differentiated source and target datasets.
    
    Args:
        df: pandas DataFrame, the dataframe contains the source and target data
        label: str, the label column name
        alpha: float, the bias parameter (controls drop probability based on KDE)
        spatial_column: list, list of spatial column names
        drop: bool, whether to drop spatial columns from features
        labelshift: bool, whether to include label in features
        kde_bandwidth: float, bandwidth for the KDE (None for automatic selection)
    
    Returns:
        source_data: tuple, biased source data (features, spatial columns, labels)
        target_data: tuple, biased target data (features, spatial columns, labels)
    '''
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]
    
    # Fit KDE for each feature
    kde_models = {}
    for col in feature_cols:
        kde = gaussian_kde(df[col], bw_method=kde_bandwidth)
        kde_models[col] = kde

    # Compute KDE probabilities for each row and feature
    source_probabilities = []
    target_probabilities = []
    for col in feature_cols:
        kde = kde_models[col]

        # Compute KDE probabilities for the entire column
        probabilities = kde.evaluate(df[col].values)

        # Normalize probabilities for consistent scaling
        probabilities /= probabilities.max()

        # Define disjoint quantile thresholds
        low_threshold = df[col].quantile(0.2)
        high_threshold = df[col].quantile(0.8)

        # Define source and target probabilities
        source_prob = ((df[col] <= low_threshold) * (1 - probabilities)) ** alpha
        target_prob = ((df[col] >= high_threshold) * probabilities) ** alpha

        source_probabilities.append(source_prob)
        target_probabilities.append(target_prob)

    # Combine probabilities across all features
    source_bias = np.mean(source_probabilities, axis=0)
    target_bias = np.mean(target_probabilities, axis=0)

    # Clip probabilities to avoid negatives
    source_bias = np.clip(source_bias, 1e-6, 1)
    target_bias = np.clip(target_bias, 1e-6, 1)

    # Convert biases to Pandas Series for compatibility with .reindex
    source_bias = pd.Series(source_bias, index=df.index)
    target_bias = pd.Series(target_bias, index=df.index)

    # Sample source data
    source_data = df.sample(frac=0.5, weights=source_bias, random_state=42)

    # Ensure alignment of weights with remaining_data index
    remaining_data = df.drop(source_data.index)
    aligned_target_bias = target_bias.reindex(remaining_data.index)

    # Sample target data
    target_data = remaining_data.sample(frac=1, weights=aligned_target_bias, random_state=42)

    # Reset indices
    source_data.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    # Prepare the return values
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data



def sync_biased_data_adjusted(df: pd.DataFrame, label: str = 'label', alpha: float = 4.0,
                              spatial_column: list = [], drop: bool = True, labelshift: bool = False,
                              kde_bandwidth: float = None, smooth_factor: float = 0.05):
    '''
    Function to create differentiated source and target datasets with minimal data filtering.
    
    Args:
        df: pandas DataFrame, the dataframe contains the source and target data
        label: str, the label column name
        alpha: float, the bias parameter (controls drop probability based on KDE)
        spatial_column: list, list of spatial column names
        drop: bool, whether to drop spatial columns from features
        labelshift: bool, whether to include label in features
        kde_bandwidth: float, bandwidth for the KDE (None for automatic selection)
        smooth_factor: float, standard deviation for Gaussian noise to smoothen the distribution
    
    Returns:
        source_data: tuple, biased source data (features, spatial columns, labels)
        target_data: tuple, biased target data (features, spatial columns, labels)
    '''
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]
    
    # Fit KDE for each feature
    kde_models = {}
    for col in feature_cols:
        kde = gaussian_kde(df[col], bw_method=kde_bandwidth)
        kde_models[col] = kde

    # Initialize weights
    source_probabilities = []
    target_probabilities = []

    for col in feature_cols:
        kde = kde_models[col]

        # Compute KDE probabilities for the entire column
        probabilities = kde.evaluate(df[col].values)
        probabilities /= probabilities.max()  # Normalize probabilities

        # Relaxed quantile thresholds
        low_threshold = df[col].quantile(0.3)
        high_threshold = df[col].quantile(0.7)

        # Compute weights for source and target datasets
        source_weight = ((df[col] <= high_threshold) * (1 - probabilities)) ** alpha
        target_weight = ((df[col] >= low_threshold) * probabilities) ** alpha

        # Penalize rows outside the relaxed quantile ranges
        source_weight[df[col] > high_threshold] *= 0.5
        target_weight[df[col] < low_threshold] *= 0.5

        source_probabilities.append(source_weight)
        target_probabilities.append(target_weight)

    # Combine weights across all features
    source_bias = np.mean(source_probabilities, axis=0)
    target_bias = np.mean(target_probabilities, axis=0)

    # Add small noise for smoother transitions
    source_bias += np.random.normal(0, smooth_factor, size=len(source_bias))
    target_bias += np.random.normal(0, smooth_factor, size=len(target_bias))

    # Clip weights to avoid negatives
    source_bias = np.clip(source_bias, 1e-6, 1)
    target_bias = np.clip(target_bias, 1e-6, 1)

    # Convert weights to Pandas Series
    source_bias = pd.Series(source_bias, index=df.index)
    target_bias = pd.Series(target_bias, index=df.index)

    # Sample source and target data with adjusted weights
    source_data = df.sample(frac=0.5, weights=source_bias, random_state=42)
    remaining_data = df.drop(source_data.index)
    aligned_target_bias = target_bias.reindex(remaining_data.index)
    target_data = remaining_data.sample(frac=1, weights=aligned_target_bias, random_state=42)

    # Reset indices
    source_data.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data



def sync_biased_data_peak_reduce(df: pd.DataFrame, label: str = 'label', alpha: float = 4.0,
                                 spatial_column: list = [], drop: bool = True, labelshift: bool = False,
                                 kde_bandwidth: float = None):
    '''
    Function to create source and target datasets that match a target distribution shape.
    
    Args:
        df: pandas DataFrame, the dataframe contains the source and target data
        label: str, the label column name
        alpha: float, the bias parameter (controls how sharply the desired PDF reduces)
        spatial_column: list, list of spatial column names
        drop: bool, whether to drop spatial columns from features
        labelshift: bool, whether to include label in features
        kde_bandwidth: float, bandwidth for the KDE (None for automatic selection)
    
    Returns:
        source_data: tuple, biased source data (features, spatial columns, labels)
        target_data: tuple, biased target data (features, spatial columns, labels)
    '''
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]
    
    # Initialize weights
    source_probabilities = []
    target_probabilities = []

    for col in feature_cols:
        # Define peaks for source and target
        low_threshold = df[col].quantile(0.25)  # Source peak
        high_threshold = df[col].quantile(0.75)  # Target peak
        value_range = df[col].max() - df[col].min()

        # Fit KDE for the feature
        kde = gaussian_kde(df[col], bw_method=kde_bandwidth)
        probabilities = kde.evaluate(df[col].values)

        # Define desired target PDFs
        def norm_pdf(x, mean, std):
            return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        def train_pdf(x):
            std = np.std(x)
            return np.where(x<low_threshold, norm_pdf(x, low_threshold, std), norm_pdf(x, low_threshold, alpha * std))

        def test_pdf(x):
            std = np.std(x)
            return np.where(x<high_threshold, norm_pdf(x, high_threshold, alpha * std), norm_pdf(x, high_threshold, std))

        # Compute desired PDFs
        desired_train_pdf = train_pdf(df[col].values)
        desired_test_pdf = test_pdf(df[col].values)

        # Compute adjusted weights using importance sampling
        source_weight = desired_train_pdf / probabilities
        target_weight = desired_test_pdf / probabilities

        # Clip and normalize weights
        source_weight = np.clip(source_weight, 1e-6, None)
        target_weight = np.clip(target_weight, 1e-6, None)

        source_weight /= source_weight.sum()
        target_weight /= target_weight.sum()

        source_probabilities.append(source_weight)
        target_probabilities.append(target_weight)

    # Combine weights across all features
    source_bias = np.mean(source_probabilities, axis=0)
    target_bias = np.mean(target_probabilities, axis=0)

    # Convert weights to Pandas Series
    source_bias = pd.Series(source_bias, index=df.index)
    target_bias = pd.Series(target_bias, index=df.index)

    # Sample source and target data
    source_data = df.sample(frac=0.5, weights=source_bias, random_state=42)
    remaining_data = df.drop(source_data.index)
    aligned_target_bias = target_bias.reindex(remaining_data.index)
    target_data = remaining_data.sample(frac=1, weights=aligned_target_bias, random_state=42)

    # Reset indices
    source_data.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    # Prepare the return values
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data

from sklearn.cluster import KMeans
def sync_biased_data_with_kmeans(df: pd.DataFrame, label: str = 'label', n_clusters: int = 2, alpha: float = 4.0,
                                 spatial_column: list = [], drop: bool = True, labelshift: bool = False,
                                 kde_bandwidth: float = None):
    '''
    Function to create train and test datasets using KMeans clustering and control the distribution difference with alpha.
    
    Args:
        df: pandas DataFrame, the dataframe contains the data
        label: str, the label column name
        n_clusters: int, number of clusters for KMeans
        alpha: float, parameter controlling how different train and test distributions are
        spatial_column: list, list of spatial column names
        drop: bool, whether to drop spatial columns from features
        labelshift: bool, whether to include label in features
        kde_bandwidth: float, bandwidth for the KDE (None for automatic selection)
    
    Returns:
        source_data: tuple, train data (features, spatial columns, labels)
        target_data: tuple, test data (features, spatial columns, labels)
    '''
    # Determine feature columns
    if drop:
        feature_cols = [col for col in df.columns if col not in [label] + spatial_column]
    else:
        feature_cols = [col for col in df.columns if col not in [label]]

    if labelshift:
        feature_cols += [label]

    # Extract features
    X = df[feature_cols].values

    # Cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,init='random',n_init=100)
    df['cluster'] = kmeans.fit_predict(X)

    # Get cluster centers and assign probabilities based on clusters
    cluster_centers = kmeans.cluster_centers_

    # Initialize weights for train and test
    train_weights = np.zeros(len(df))
    test_weights = np.zeros(len(df))
    all_kde = gaussian_kde(df[feature_cols].values.T, bw_method=kde_bandwidth)

    for cluster_id in range(n_clusters):
        # Filter data points belonging to the current cluster
        cluster_points = df[df['cluster'] == cluster_id]

        # Fit KDE within the cluster (optional, for smooth distributions)
        kde = gaussian_kde(cluster_points[feature_cols].values.T, bw_method=kde_bandwidth)
        
        
        probabilities = kde.evaluate(cluster_points[feature_cols].values.T)
        ## normalize
        normalize_probs = all_kde.evaluate(cluster_points[feature_cols].values.T)
        # Normalize probabilities
        probabilities /= normalize_probs

        # Train weight: Favor low probability for alpha > 1
        train_cluster_weight = probabilities ** (1 / alpha)

        # Test weight: Favor high probability for alpha > 1
        test_cluster_weight = probabilities ** alpha

        # Normalize cluster-specific weights
        train_cluster_weight /= train_cluster_weight.sum()
        test_cluster_weight /= test_cluster_weight.sum()

        # Assign weights to the global train and test weight arrays
        train_weights[cluster_points.index] = train_cluster_weight
        test_weights[cluster_points.index] = test_cluster_weight

    # Normalize global weights
    train_weights /= train_weights.sum()
    test_weights /= test_weights.sum()

    # Sample train and test datasets
    source_data = df.sample(frac=0.5, weights=train_weights, random_state=42)
    remaining_data = df.drop(source_data.index)
    target_data = remaining_data.sample(frac=1, weights=test_weights[remaining_data.index], random_state=42)

    # Reset indices
    source_data.reset_index(drop=True, inplace=True)
    target_data.reset_index(drop=True, inplace=True)

    # Prepare the return values
    feature_cols = [col for col in feature_cols if col != label]

    if spatial_column:
        source_data = (source_data[feature_cols].values, 
                       source_data[spatial_column].values, 
                       source_data[label].values)
        
        target_data = (target_data[feature_cols].values, 
                       target_data[spatial_column].values, 
                       target_data[label].values)
    else:
        source_data = (source_data[feature_cols].values, None, source_data[label].values)
        target_data = (target_data[feature_cols].values, None, target_data[label].values)

    return source_data, target_data
