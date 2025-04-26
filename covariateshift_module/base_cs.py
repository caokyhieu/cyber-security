from utils.reweight_func import _get_weights_from_cost_matrix
import jax.numpy as jnp
from jax import random, vmap
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
import pdb
from utils.dataset import jax_collate_fn
import jax

import jax.numpy as jnp

from covariateshift_module.KMM import kernel_mean_matching, KernelMeanMatching
from covariateshift_module.metric_learning import calculate_cost_matrix_NCA
from covariateshift_module.logistic_regression import RegressionClassifier
from utils.preprocess import calculate_optimal_transport_plan_sinkhorn
from utils.pyro_utils import distance_function

def estimate_distance(X_train, X_test, name=''):
    if name == 'KMM':
        return 1/kernel_mean_matching(X_test, X_train, kern='rbf', B=4., eps=None)
        # kmm = KernelMeanMatching()
        # weights = kmm.fit(jnp.array(X_train), jnp.array(X_test))
        # return 1/weights
    elif name == 'OT':
        return calculate_optimal_transport_plan_sinkhorn(X_train,X_test)
    elif name == 'NCA':
        return calculate_cost_matrix_NCA(X_train,X_test)
    elif name == 'euclidean':
        return distance_function(X_train,X_test)
   
    elif name == '':
        return jnp.ones((X_train.shape[0],X_test.shape[0]))

    else:
        raise ValueError('distance method not found')

from utils.reweight_func import EV_scale_covariance_matrix, VM_scale_covariance_matrix,power_scale_covariance_matrix

def scale_cov_matrix(cov_matrix, cost_matrix, name=''):
    if name == 'EV':
        return EV_scale_covariance_matrix(cov_matrix, cost_matrix)
    elif name == 'VM':
        return VM_scale_covariance_matrix(cov_matrix, cost_matrix)
    elif name =='likelihood':
        return cov_matrix
    elif name =='power':
        return power_scale_covariance_matrix(cov_matrix, cost_matrix)
    else:
        raise ValueError('scale method not found')





def pad_batch(batch, batchsize=32):
    if batch[0].shape[0]==batchsize:
        return batch 
    else:
        new_batch = []
        for b in batch:
            if b.ndim==1:
                new_batch.append(jnp.pad(b, (0, batchsize - len(b)) ,mode='reflect'))
            else:
                new_batch.append(jnp.pad(b, ((0, batchsize - len(b)),) + ((0,0),) * (b.ndim -1) ,mode='reflect'))
        return new_batch

## generate train_data
def generate_batch(train_dataloader):
    batch_size = train_dataloader.batch_size    
    batches = [batch for batch in train_dataloader]
    ## patch for last batch
    batches[-1] = pad_batch(batches[-1], batchsize=batch_size)
    
    batches = [(jnp.expand_dims(el,axis=0) for el in batch) for batch in batches]
    batches = list(zip(*batches))
    batches = [jnp.concatenate(b,axis=0) for b in batches]
    return batches   
    


class WrapperCovariateShift:
    BATCHSIZE_TRAINING = False 

    def __init__(self, distance_method='NCA',scale_method='average'):
        self.distance_method = distance_method
        self.scale_method = scale_method
        if distance_method in ['neuralNCA','uLSIF','KLIEP','RandomProjection','WassersteinRatio','logistic','NCE','telescope','KMM', 'GradKLIEP','']:
            '''
            Need code to handle this class
            '''
            self.BATCHSIZE_TRAINING = True
       
    def trainDistanceMethod(self,trainData,testData):
        '''
        This method to train distance method for uLSIF and neuralNCA
        '''
        from covariateshift_module.metric_learning import NeuralNCA
        from covariateshift_module.uLSIF import uLSIF
        from covariateshift_module.wasserstein_ratio import WassersteinRatio,WassersteinRatiov2
        from covariateshift_module.KLIEP import KLIEP, GradKLIEP
        from covariateshift_module.logistic_regression import RegressionClassifier
        from covariateshift_module.neural_contrastive import NCEEstimator
        from covariateshift_module.telescope_cs import TelescopeCS
        from covariateshift_module.KMM import KernelMeanMatchingv2
        ## check trainData and testData is dataloader or not
        if isinstance(trainData, DataLoader) and isinstance(testData, DataLoader):
            cov_train = trainData.dataset.get_covariates()
            cov_test = testData.dataset.get_covariates()
        elif isinstance(testData, DataLoader):
            cov_train = trainData
            cov_test = testData.dataset.get_covariates()
        elif isinstance(trainData, DataLoader):
            cov_train = trainData.dataset.get_covariates()
            cov_test = testData
        else:
            cov_train = trainData
            cov_test = testData

        if self.distance_method == 'uLSIF':
            method = uLSIF(num_basis=500)
            ## if cov_train and cov_test too large
            ## subsample
            if len(cov_train)> 20_000:
                train_index = np.random.randint(low=0, high=len(cov_train),size=(20_000,))
            else:
                train_index = np.arange(len(cov_train))
            if len(cov_test)> 20_000:
                test_index = np.random.randint(low=0, high=len(cov_test),size=(20_000,))
            else:
                test_index= np.arange(len(cov_test))

            method.fit(cov_train[train_index],cov_test[test_index])
            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix
            # method.compute_weights(trainData.dataset.get_covariates())
        elif self.distance_method == 'KLIEP':
            method = KLIEP(num_basis=500)
            method.fit(cov_train,cov_test)

            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix

        elif self.distance_method == 'GradKLIEP':
            method = GradKLIEP(num_basis=500)
            if len(cov_train)> 20_000:
                train_index = np.random.randint(low=0, high=len(cov_train),size=(20_000,))
            else:
                train_index = np.arange(len(cov_train))
            if len(cov_test)> 20_000:
                test_index = np.random.randint(low=0, high=len(cov_test),size=(20_000,))
            else:
                test_index= np.arange(len(cov_test))

            method.fit(cov_train[train_index],cov_test[test_index])

            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix
        elif self.distance_method == 'KMM':
            method = KernelMeanMatchingv2(num_basis=500)
            if len(cov_train)> 20_000:
                train_index = np.random.randint(low=0, high=len(cov_train),size=(20_000,))
            else:
                train_index = np.arange(len(cov_train))
            if len(cov_test)> 20_000:
                test_index = np.random.randint(low=0, high=len(cov_test),size=(20_000,))
            else:
                test_index= np.arange(len(cov_test))

            method.fit(cov_train[train_index],cov_test[test_index])
            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix
        elif self.distance_method == 'RandomProjection':
            from covariateshift_module.uLSIF import RandomProjection
            method = RandomProjection()
            method.fit(cov_train,cov_test)
            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix
        elif self.distance_method == 'telescope':
            method = TelescopeCS(method='KLIEP', n_estimators=5, transformer_name='normal',num_basis=500)
            if len(cov_train)> 20_000:
                train_index = np.random.randint(low=0, high=len(cov_train),size=(20_000,))
            else:
                train_index = np.arange(len(cov_train))
            if len(cov_test)> 20_000:
                test_index = np.random.randint(low=0, high=len(cov_test),size=(20_000,))
            else:
                test_index= np.arange(len(cov_test))

            method.fit(cov_train[train_index],cov_test[test_index])
            matrix = jnp.clip(method.compute_weights(cov_train),min=1e-5)
            matrix = 1/matrix
        
        
        
        elif self.distance_method == 'logistic':
            clf = RegressionClassifier(input_dim=cov_train.shape[-1])
            if len(cov_train)> 20_000:
                train_index = np.random.randint(low=0, high=len(cov_train),size=(20_000,))
            else:
                train_index = np.arange(len(cov_train))
            if len(cov_test)> 20_000:
                test_index = np.random.randint(low=0, high=len(cov_test),size=(20_000,))
            else:
                test_index= np.arange(len(cov_test))

            clf.fit(cov_train[train_index],cov_test[test_index])
            matrix = jnp.clip(clf.compute_weights(cov_train),a_min=1e-5)
            matrix = 1/matrix

        elif self.distance_method == '':
            matrix = jnp.ones((len(cov_train),1))

        elif self.distance_method == 'WassersteinRatio':
            # method = WassersteinRatio(num_projection=1000,layers_config=[256, 128],learning_rate=1e-3,num_epochs=100)
            method = WassersteinRatiov2(num_projection=500,layers_config=[256, 128],learning_rate=1e-1,num_epochs=40)
            method.fit(cov_train,cov_test)
            matrix = jnp.clip(method.compute_weights(cov_train),a_min=1e-5)
            matrix = 1/matrix

        elif self.distance_method == 'NCE':
            method = NCEEstimator(layers_config=[256, 128, 64],learning_rate=1e-3,num_epochs=1000)
            method.fit(cov_train,cov_test)
            matrix = jnp.clip(method.compute_weights(cov_train),a_min=1e-5)
            matrix = 1/matrix
        elif self.distance_method == 'neuralNCA':
            method = NeuralNCA(input_dim=cov_train.shape[-1],embedding_dim=10,
                               key=random.PRNGKey(123),num_layers=2,layers=[128])
            trData = jnp.concatenate([cov_train,cov_test],axis=0)
            labels = jnp.concatenate([jnp.ones((len(cov_train),1)),jnp.zeros((len(cov_test),1))],axis=0)
            method.train(x=trData,y=labels,n_epochs=1000,stepsize=1e-3,batchsize=128)
            matrix = method.pair_distance(cov_train,cov_test)
        else:
            raise ValueError('distance method not found')
        return matrix

    def getDistanceFunction(self):
        if self.distance_method in ['neuralNCA','uLSIF','KLIEP','RandomProjection','WassersteinRatio', 'logistic','NCE','telescope','KMM', 'GradKLIEP', '']:
            return lambda train_data,test_data: self.trainDistanceMethod(train_data,test_data)
        else:
            return lambda train_data,test_data: estimate_distance(train_data,test_data,name=self.distance_method)
    
    def getWeightsFromCostMatrixFunction(self):
        if self.distance_method in ['uLSIF','KLIEP','RandomProjection','WassersteinRatio','logistic','NCE','telescope', 'KMM', '','GradKLIEP']:
            return lambda sim_matrix: _get_weights_from_cost_matrix(sim_matrix,name='')
        else:
            return lambda sim_matrix: _get_weights_from_cost_matrix(sim_matrix,name=self.scale_method)
from jax import random, vmap,jit, lax

class CovariateShiftMethod:
    N_SUBSAMPLES = 4000

    def __init__(self,randomseed, distance_method='NCA',scale_method='average'):

        self.distance_method = distance_method
        self.scale_method = scale_method
        self.randomseed = randomseed
        self.cov_method = WrapperCovariateShift(distance_method=self.distance_method,scale_method=self.scale_method)
        pass

    def prepare_test_data(self,  test_data):
        """
        Args: 
            test_data: jnp.array or dataloader
        
            Return:
                test_data: jnp.array
        """
        ## WIL BE DEBRICATED

        loop_test_data = isinstance(test_data,DataLoader)
        if  loop_test_data:
            ## both train and test are big data
            ### strategy is subsample for test data
            ## and loop through the train data and calculate the similarity between
            ## them 
            sample_idxes = random.choice(self.randomseed, jnp.arange(0,len(test_data.dataset)),shape=(self.N_SUBSAMPLES,))
            sub_test_data = test_data.dataset[sample_idxes]
            ## checkcase for type of sub_test_data
            if isinstance(sub_test_data, (jnp.ndarray, np.ndarray)):
                sub_test_data = jnp.array(sub_test_data)
            elif isinstance(sub_test_data, (list,tuple)):
                ## take the first argument
                sub_test_data = jnp.array(sub_test_data[0])
            elif isinstance(sub_test_data, dict):
                sub_test_data = jnp.array(sub_test_data[list(sub_test_data.keys())[0]])
           
            return  sub_test_data
        else:
            return test_data

   
        
    def compute_similarity_batch_size(self, train_data, test_data) :
        __estimate_distance =  self.cov_method.getDistanceFunction()
        __get_weights_from_cost_matrix = self.cov_method.getWeightsFromCostMatrixFunction()
        
        distance_matrix = __estimate_distance(train_data,test_data)
        result = __get_weights_from_cost_matrix(distance_matrix)
        return result 
    
    def get_len(self, train_data):
        if isinstance(train_data, (jnp.ndarray, np.ndarray)):
            return len(train_data)
        elif isinstance(train_data, (list,tuple)):
            ## take the first argument
            return len(train_data[0])
        elif isinstance(train_data, dict):
            return len(train_data[list(train_data.keys())[0]])
        
    def compute_similarity_core(self, train_data, test_data):

        __estimate_distance =  self.cov_method.getDistanceFunction()
        __get_weights_from_cost_matrix = self.cov_method.getWeightsFromCostMatrixFunction()

        ## cast type for training and testing data
        ## ----------------------------------------------------------------
        if isinstance(train_data, (jnp.ndarray, np.ndarray)):
            train_data = np.array(train_data)
        elif isinstance(train_data, (list,tuple)):
            ## take the first argument
            train_data = np.array(train_data[0])
        elif isinstance(train_data, dict):
            train_data = np.array(train_data[list(train_data.keys())[0]])
        
        if isinstance(test_data, (jnp.ndarray, np.ndarray)):
            test_data = np.array(test_data)
        elif isinstance(test_data, (list,tuple)):
            ## take the first argument
            test_data = np.array(test_data[0])
        elif isinstance(test_data, dict):
            test_data = np.array(test_data[list(test_data.keys())[0]])

        #----------------------------------------------------------------
        # pdb.set_trace()
        if self.distance_method !='':
            distance_matrix = __estimate_distance(train_data,test_data)
            ## calculate the similarity vector
            result = __get_weights_from_cost_matrix(distance_matrix)
        else:
            result = jnp.ones(shape=(len(train_data),))
        
        return result
    
    def compute_similarity_test_batch_train(self, train_data, test_data):
        l_train = self.get_len(train_data)
        result = jnp.ones((l_train,))
        test_data = generate_batch(test_data)
        # jax.debug.print("debug training: {}", len(train_data))
        def iterate_update_test(result):
            def scan_fn(carry, temp_data):
                result,n_total = carry
                idxs = temp_data.pop()
                len_test= self.get_len(temp_data)
                out_type = jax.ShapeDtypeStruct((l_train,), 'float64')
                score = jax.pure_callback(self.compute_similarity_core,out_type ,train_data,temp_data)
                n_total+= len_test
                result += len_test * 1/score
                return (result,n_total), idxs  # Return updated parameters and a dummy second argument

            # Use `lax.scan` to loop over the data (faster than a Python loop)
            (result,n_total),idxs = lax.scan(scan_fn, (result,0.), test_data)
            return result,n_total,idxs

        result,n_total,idxs = iterate_update_test(result)
        result = n_total/result   
        return result,idxs
    
    
    def compute_similarity_train_batch_test(self, train_data, test_data):
        len_train_data = len(train_data.dataset)
        n_batches = len(train_data)
        train_data_batchsize = train_data.batch_size
        result = jnp.ones((n_batches * train_data_batchsize,))

        
        train_data = generate_batch(train_data)
        

        def iterate_update(result):
            def scan_fn(carry, temp_data):
                i, result = carry
                idxs = temp_data.pop()
                out_type = jax.ShapeDtypeStruct((len_train_data,), 'float64')
                score = jax.pure_callback(self.compute_similarity_core,out_type ,temp_data,test_data)
                result = lax.dynamic_update_slice(result,score,(i*train_data_batchsize,))
                i+=1
                return (i,result), idxs  # Return updated parameters and a dummy second argument

            # Use `lax.scan` to loop over the data (faster than a Python loop)
            (i,result),idxs = lax.scan(scan_fn, (0,result), train_data)
            final_result = result[:len_train_data]
            reconstructed = jnp.zeros_like(result)
            idxs = jnp.hstack(idxs)[:len_train_data]
            reconstructed = reconstructed.at[idxs].set(final_result)
            return result,idxs

        result,idxs = iterate_update(result)
        return result, idxs
    
    def compute_similarity_train_batch_test_batch(self, train_data, test_data):
        n_batches = len(train_data)
        train_batchsize = train_data.batch_size
        result = jnp.ones((n_batches * train_batchsize,))
        
        len_train_data = len(train_data.dataset)
        train_data = generate_batch(train_data)

        def iterate_update(result):
            def scan_fn(carry, temp_data):
                i,result = carry
                idxs = temp_data.pop()
                new_result,_ = self.compute_similarity_test_batch_train(temp_data,test_data)
                result = lax.dynamic_update_slice(result,new_result,(i*train_batchsize,))
                i+=1

                return (i,result), idxs  # Return updated parameters and a dummy second argument

            # Use `lax.scan` to loop over the data (faster than a Python loop)
            (i, result),idxs = lax.scan(scan_fn, (0, result), train_data)
            final_result = result[:len_train_data]
            reconstructed = jnp.zeros((len_train_data,))
            idxs = jnp.hstack(idxs)[:len_train_data]
            reconstructed = reconstructed.at[idxs].set(final_result)
            return reconstructed, jnp.vstack(idxs)[:len_train_data]

        result,idxs = iterate_update(result)
    
        return result,idxs


    def compute_similarity(self, train_data, test_data):
        '''
        Args:
            train_data: jnp.array or DataLoader
            test_data: jnp.array or DataLoader
        return:
            similarity: jnp.array
    
        '''
        ## prepare test data, return jnp.array
        train_data.dataset.set_return_index(True)
        test_data.dataset.set_return_index(True)
        # pdb.set_trace()
        
        if self.cov_method.BATCHSIZE_TRAINING:
            result = self.compute_similarity_batch_size(train_data,test_data)

        elif isinstance(train_data, DataLoader):
            if isinstance(test_data, DataLoader):
                result = self.compute_similarity_train_batch_test_batch(train_data,test_data)
            else:
                result = self.compute_similarity_train_batch_test(train_data,test_data)
        elif isinstance(test_data,DataLoader):
            result = self.compute_similarity_test_batch_train(train_data,test_data)
        else:
            result = self.compute_similarity_core(train_data,test_data)
        # pdb.set_trace()
        if isinstance(result, tuple):
            return result[0]
        else:
            return result



    