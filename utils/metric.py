import numpy as np
from utils.pyro_utils import radial_basic_function
from visualization.visualize import plot_error
## TO-DO: move the visusalizations to this module and seperate them from the experiment module
import os
import pandas as pd
from collections import namedtuple
from scipy.stats import f_oneway,ttest_ind
import pdb
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib

from queue import PriorityQueue
MetricResult = namedtuple('MetricResult',['n_data','n_samples_mcmc','MGSE','VarGSE','MSE','VarSE','R2','MAPE'])

class RecordWorstSamples:

    def __init__(self,n_samples:int=1):
        '''
        Args:
            n_samples: number of samples to record
        '''
        ## init a queue to record the worst samples
        self.queue = PriorityQueue(maxsize=n_samples)
        pass 

    def record_worst_samples(self, X, y, y_pred):
        '''
        Args:
            X: input data
            y: true labels
            y_pred: predictions
            n_samples: number of samples to record
        '''
        ## calculate the error
        error = np.mean((y_pred-y)**2,axis=1)
        ## loop through each sample
        for i in range(X.shape[0]):
            ## pop the sample if the queue is full
            if self.queue.full():
                best_samples = self.queue.get()
                ## compare the pop out sampels with current samples
                if best_samples[0] > -error[i]:
                    self.queue.put(best_samples)
                    continue

            ## add the sample to the queue
            self.queue.put((-error[i],(X[i],y[i],y_pred[i])))
            
        pass

    def get_samples(self,):
        '''
        Return the worst samples in order
        '''
        result: list = []
        while not self.queue.empty():
            result.append(self.queue.get())
        return result

            

class MetricMileometer:

    def __init__(self,metric='count',models=[]):
        """
        Args:
            metric: string
            models: list of models names
        """
        self.metric = metric
        self.models = models
        
        ### structure of results:
        ### list of dictionaries
        ### each dictionary is a result of a single experiment
        self.results = []

        # self.samples = []
        # self.true_params = true_params
        pass 
    
    def _check_models(self, models):
        for model in models:
            if model not in self.models:
                self.models.append(model)
                print(f"Model {model} not found in current model list. Added {model} to model list")
        pass

    # def add_samples(self, samples):
    #     self._check_models(samples.keys())
    #     self.samples.append(samples)
    #     print("added posterior samples")
    #     pass

    def add_result(self,result_dict):
        '''
        method to add result into final result list
        '''

        ## first check the final result in list
        ## if the result_dict key not in final result element, update the final result
        assert isinstance(result_dict,dict),'Result should be a dictionary'
        ## check case empty
        if len(self.results)==0:
            self.results.append(result_dict)
            return
        
        ## take last result
        last_el = self.results.pop()
        ## compare with last result
        if all(name in last_el for name in result_dict):
            self.results += [last_el,result_dict]
        else:
            last_el.update(result_dict)
            self.results.append(last_el)

    def measure(self, predictions, labels):
        """
        Measure the difference between predictions and true labels.
        Args:
            predictions: Dictionary of predictions (keys: model name, values:predictions)
            labels: true labels
        """       
        
        self._check_models(predictions.keys())
        
        err = {}
        ## add the error to the results
        for model,prediction in predictions.items():
            ## follow structure of results
            if labels is not None:
                labels = labels.reshape(1,-1)
                assert prediction.shape[1]==labels.shape[1], \
                    f"Prediction and labels should have the same number of data points. Got {prediction.shape[1]} and {labels.shape[1]}"
                
                err[model] = MetricResult(
                                n_data=prediction.shape[1],
                                n_samples_mcmc=prediction.shape[0],
                                MGSE=float(np.mean((prediction-labels)**2)),
                                VarGSE= float(np.var((prediction-labels)**2)),
                                MSE= float(np.mean((np.mean(prediction,axis=0)-labels)**2)),
                                VarSE= float(np.var((np.mean(prediction,axis=0)-labels)**2)),
                                R2= r2_score(labels.squeeze(), np.mean(prediction,axis=0).squeeze()),
                                MAPE= mean_absolute_percentage_error(labels.squeeze(), np.mean(prediction,axis=0).squeeze())
                                
                )
            else:
                err[model] = MetricResult(
                                n_data=prediction,
                                n_samples_mcmc=prediction,
                                MGSE=prediction,
                                VarGSE= prediction,
                                MSE= prediction,
                                VarSE= prediction,
                                R2= prediction,
                                MAPE= prediction
                                
                )
        self.add_result(err)
        return err 
        
    def count(self,criteria='MGSE'):

        assert len(self.results)>0, 'Need to add measure first'
        final_result = {name:[0,0] for name in self.models}
        for result in self.results:
            sorted_model = sorted(result.items(), key=lambda item: getattr(item[1],criteria))
            i = 0
            for key, _ in sorted_model:
                if i== 0:
                    final_result[key][0]+=1
                final_result[key][1]+=1
                i+=1
        print("Wining rate of models:")
        for k,v in final_result.items():
            print(f"{k}: {v[0]}/{v[1]}")
        return final_result
    
    def aggregate(self,agg_func=np.mean,criteria='R2'):
        """
        Aggregate the results of all models.
        Args:
            agg_func: aggregation function
        Returns:
            agg_result: aggregated results
        """
        assert len(self.results)>0, 'Need to add measure first'
        agg_result = {name:[] for name in self.models}
        for result in self.results:
            for model, err in result.items():
                agg_result[model].append(getattr(err,criteria))
        ## aggregate the results
        for model in self.models:
            agg_result[model] = agg_func(agg_result[model],axis=0)
        return agg_result
    
    def plot_accumuate_error(self,fig_path='',criteria='R2'):
        """
        Plot the accumulate error of the models.
        Args:
            fig_path: path to save the figure
            criteria: criteria used to plot the error
        
        """
        assert len(self.results)>0, 'Need to add measure first'
        ## get data to plot in form of list of dictionaries
        data = []
        
        for result in self.results:
            data.append({k:getattr(v,criteria) for k,v in result.items()})
        # pdb.set_trace()
        plot_error(data,fig_path,criteria)
        pass


    def process_df(self,df):
        """
        Process the dataframe to align with the internal data structure.
        Args:
            df: dataframe
        Returns:
            predictions: dictionary of predictions
            labels: true labels
        """
        ## make sure the 'Model' column is in the dataframe
        assert 'Model' in df.columns, "Model column not found in dataframe"
        ## make sure the 'y_pred' column is in the dataframe
        assert 'y_pred' in df.columns, "y_pred column not found in dataframe"
        ## make sure the 'true_y' column is in the dataframe
        assert 'true_y' in df.columns, "true_y column not found in dataframe"

        ## first, group the dataframe by model name
        df_grouped = df.groupby('Model')
        predictions = {}
        ## get list group names
        group_names = list(df_grouped.groups.keys())
        ## take first group
        group_name = group_names[0]
        ## get the number of samples for this group
        num_samples = df_grouped.get_group(group_name)['true_y'].nunique()
        ## get the true labels with the shape (-1,num_samples)
        labels = df_grouped.get_group(group_name)['true_y'].values.reshape(-1,num_samples )[0]
        ## get predictions for this group
        predictions[group_name] = df_grouped.get_group(group_name)['y_pred'].values.reshape(-1,num_samples)
        ## pop this group out of the list
        group_names.pop(0)
        ## loop through each model name:
        for model_name in group_names:
            ## get the number of samples for this model
            num_samples_model = df_grouped.get_group(model_name)['true_y'].nunique()
            ## make sure the number of samples are the same
            assert num_samples == num_samples_model, "Number of samples are not the same for different models"
            ## check the samples are the same
            assert np.all(df_grouped.get_group(model_name)['true_y'].values.reshape(-1,num_samples )[0] == labels), "Samples are not the same for different models"
            ## get the predictions for this model
            predictions[model_name] = df_grouped.get_group(model_name)['y_pred'].values.reshape(-1,num_samples_model)
        
        return predictions, labels
    

    def read_csv(self, csv_path):
        """
        Read the csv file and process the dataframe to align with the internal data structure.
        Args:
            csv_path: path to the csv file
        Returns:
            predictions: dictionary of predictions
            labels: true labels
        """
        assert os.path.exists(csv_path), "csv file not found"
        df = pd.read_csv(csv_path)
        return self.process_df(df)
    
    def process_data_from_rootpath(self,rootpath='',filename='.csv'):
        """
        Process data from the rootpath
        Args:
            rootpath: rootpath
            filename: filename
        Pass
        """
        assert os.path.exists(rootpath), "rootpath not found"
        assert os.path.isdir(rootpath), "rootpath is not a directory"
        ## loop through all subdirectories
        for subdir, dirs, files in os.walk(rootpath):
            for file in files:
                if file.endswith(filename):
                    ## read the csv file
                    predictions, labels = self.read_csv(os.path.join(subdir, file))
                    ## measure the error
                    self.measure(predictions, labels)
        pass

    
    
    def f_test(self,criteria='MGSE',source_model= '',target_model=''):
        """
        F-test to compare the performance of the models.

        Args:
            criteria: criteria used to compare the performance
            source_model: source model
            target_model: target model
        Returns:
            F-statistic and p-value
        """
        assert len(self.results)>0, 'Need to add measure first'
        ## get the criteria for the source model
        source_criteria = [getattr(result[source_model],criteria) for result in self.results if source_model in result]
        ## get the criteria for the target model
        target_criteria = [getattr(result[target_model],criteria) for result in self.results if target_model in result]
        ## calculate the F-statistic and p-value
        F,p = f_oneway(source_criteria,target_criteria)
        return F,p
    
    def f_test_pairwise(self,criteria='R2',source_models=[],target_model=''):
        """
        F-test to compare the performance of the models.

        Args:
            criteria: criteria used to compare the performance
            source_model: list of source models
            target_model: target model
        Returns:
            list of F-statistics and p-values for each source model
        """

        ### loop through each source model
        result_list = []
        for source_model in source_models:
            result_list.append((source_model,) + self.f_test(criteria=criteria,source_model=source_model,target_model=target_model))
        
        return result_list
    
    def _take_root_model_name(self,model_name,prefix=' '):
        """
        Return root model name
        """
        return model_name.split(prefix)[-1]
    
    def _check_prefix(self,prefix):
        """
        Check available prefix
        
        """
        for model in self.models:
            try:
                model.split(prefix)[-1]
                return True 
            except:
                continue
        return False

    
    def train_test_compare(self, train_prefix='train',test_prefix='test'):
        """
        Compare train test error in the results.
        """
        assert train_prefix != '', "train_prefix cannot be empty"
        assert test_prefix != '', "test_prefix cannot be empty"
        assert train_prefix!= test_prefix, "train_prefix and test_prefix should be different"
        assert len(self.results)>0, 'Need to add measure first'

        assert self._check_prefix(train_prefix) and self._check_prefix(test_prefix), f"train_prefix {train_prefix} and/or test_prefix {test_prefix} not found in model names"

        ## get the train and test models name
        train_models = [model for model in self.models if model.startswith(train_prefix)]
        test_models = [model for model in self.models if model.startswith(test_prefix)]
        assert len(train_models) == len(test_models), "Number of train models and test models are different"
        results = []
        ## loop through each train model
        for model in train_models:
            root_model_name = self._take_root_model_name(model,prefix=train_prefix)
            ## take model in test
            model_test = None
            for test_model in test_models:
                if self._take_root_model_name(test_model,prefix=test_prefix) == root_model_name:
                    model_test = test_model
                    break
            assert model_test is not None, f"Test model for {model} not found"
            ## get the F-statistic and p-value
            F,p = self.f_test(criteria='MGSE',source_model=model,target_model=model_test)
            print(f"Train model: {model}, Test model: {model_test}, F-statistic: {F}, p-value: {p}")
            results.append((model,model_test,F,p))
        
        return results
    def save(self, filename):
        """Saves the object to a file using joblib."""
        joblib.dump(self, filename)
        print(f"Object saved successfully to {filename}")
    @classmethod
    def load(cls, filename):
        """Loads the object from a joblib file."""
        obj = joblib.load(filename)
        print(f"Object loaded successfully from {filename}")
        return obj

## CODE TO EVALUATE DENSITY RATIO ESTIMATION

def normalized_mean_squared_error(y_true, y_pred):
    """
    Compute the normalized mean squared error.
    Args:
        y_true: true labels (true density ratio)
        y_pred: predicted labels (estimated denstiy ratio)
    Returns:
        normalized mean squared error
    """
    ## first normalize the true labels
    y_true = y_true/np.sum(y_true)
    ## normalize the predicted labels
    y_pred = y_pred/np.sum(y_pred)
    ## calculate the mean squared error
    return np.mean((y_true-y_pred)**2)
    
        
    
import scipy.stats as stats

def crps_gaussian(y_pred, y_std, true_y):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for Gaussian predictions.

    Parameters:
    - y_pred (array-like): Mean of the predictive distribution.
    - y_std (array-like): Standard deviation of the predictive distribution.
    - true_y (array-like): Observed true values.

    Returns:
    - float: CRPS score (lower is better).
    """
    standardized_error = (true_y - y_pred) / y_std
    cdf_values = stats.norm.cdf(standardized_error)
    pdf_values = stats.norm.pdf(standardized_error)

    crps = y_std * (standardized_error * (2 * cdf_values - 1) + 2 * pdf_values - 1 / np.sqrt(np.pi))
    return np.mean(crps)  # Return mean CRPS over all samples

def compute_predictive_uncertainty_from_samples(df, model_name=None, alpha=0.05):
    """
    Compute predictive uncertainty metrics for a Bayesian model given predictive samples.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['x', 'y_pred', 'true_y', 'Model'], 
                         where 'y_pred' contains 1000 predictions per (x, true_y) pair.
    - model_name (str, optional): If provided, filters results for a specific model.
    - alpha (float, optional): Confidence level for prediction intervals (default: 0.05 for 95% interval).

    Returns:
    - dict: Dictionary containing computed predictive uncertainty metrics.
    """

    # Filter for a specific model if provided
    if model_name:
        df = df[df["Model"] == model_name]

    # Group predictions by (x, true_y) to compute statistics from 1000 samples per prediction
    grouped = df.groupby(["x", "true_y"])

    # Compute mean and standard deviation for each (x, true_y) pair
    y_pred_mean = grouped["y_pred"].mean()
    y_pred_std = grouped["y_pred"].std()

    # Compute lower and upper credible intervals (assuming Gaussianity)
    y_lower = y_pred_mean - stats.norm.ppf(1 - alpha / 2) * y_pred_std
    y_upper = y_pred_mean + stats.norm.ppf(1 - alpha / 2) * y_pred_std

    # Convert to numpy arrays for efficient computation
    y_pred_mean = y_pred_mean.values
    y_pred_std = y_pred_std.values
    true_y = grouped["true_y"].mean().values
    y_lower = y_lower.values
    y_upper = y_upper.values

    # 1. **Sharpness**: Measures dispersion of predictions (variance)
    sharpness = np.mean(y_pred_std ** 2)

    # 2. **Prediction Interval Coverage Probability (PICP)**
    picp = np.mean((true_y >= y_lower) & (true_y <= y_upper))

    # 3. **Mean Prediction Interval Width (MPIW)**
    mpiw = np.mean(y_upper - y_lower)

    # 4. **Continuous Ranked Probability Score (CRPS)**
    crps = crps_gaussian(y_pred_mean, y_pred_std, true_y)


    return {
        "Model": model_name if model_name else "All Models",
        "Sharpness": sharpness,
        "PICP": picp,
        "MPIW": mpiw,
        "CRPS": crps
    }

# Example usage:
# df = pd.read_csv("your_file.csv")  # Load your dataset
# metrics = compute_predictive_uncertainty_from_samples(df, model_name="BRM")
# print(metrics)
