' correlation-based feature selection '
import numpy as np
import pandas as pd
from abc import ABCMeta,abstractmethod

class FeatureSelection(metaclass=ABCMeta):

    def __init__(self):
        self.fitting =False
        pass 

    @abstractmethod
    def fit(self, x, y):
        pass 

    @abstractmethod
    def transform(self, x):
        pass 

    def fit_transform(self, X, y):
        if self.fitting:
            print(f"Model is already fitted. Now it will be refitted with new data.\n")
        self.fit(X,y)
        return self.transform(X)
    
class NoneFeatureSelection(FeatureSelection):

    def __init__(self):
        super(NoneFeatureSelection,self).__init__()

    def fit(self, x, y):
        self.fitting = True
        pass

    def transform(self, x):
        assert self.fitting, "Please fit the model first"
        return x

class PriorityQueue:
    def  __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0
    
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append( (item, priority) )
                break
        else:
            self.queue.append( (item, priority) )
        
    def pop(self):
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)
        
class CFS(FeatureSelection):

    def __init__(self,max_backtrack=5):
        super(CFS,self).__init__()       
        self.queue = PriorityQueue()
        self.max_backtrack = max_backtrack

    def getMerit(self, subset):
        k = len(subset)
    
        # average feature-class correlation
        
        rcf = np.mean( np.abs(self.corr[-1,subset]) )
    
        # average feature-feature correlation
        corr = self.corr[subset,:][:,subset]
        corr[np.tril_indices_from(corr)] = np.nan
        corr = np.abs(corr)
        rff = np.nanmean(corr)
    
        return (k * rcf) / np.sqrt(k + k * (k-1) * rff)

    def fit(self,covariates,label):

        if isinstance(covariates,pd.DataFrame):
            self.feature_names = list(covariates.columns)
        else:
            self.feature_names = list(range(covariates.shape[-1]))
        
        self.corr = np.corrcoef(covariates,label,rowvar=False)
    
        best_value = -1
        best_feature = np.argmax(self.corr[-1,:-1])
        
        print("Feature %s with merit %.4f"%(best_feature, best_value))
        visited = []
        self.queue.push([best_feature], best_value)

        # counter for backtracks
        n_backtrack = 0
        
        # limit of backtracks
        max_backtrack = self.max_backtrack

        # repeat until queue is empty
        # or the maximum number of backtracks is reached
        while not self.queue.isEmpty():
            # get element of queue with highest merit
            subset, priority = self.queue.pop()
            
            # check whether the priority of this subset
            # is higher than the current best subset
            if (priority < best_value):
                n_backtrack += 1
            else:
                best_value = priority
                best_subset = subset
        
            # goal condition
            if (n_backtrack == max_backtrack):
                break
            
            # iterate through all features and look of one can
            # increase the merit
            for feature in set(self.feature_names).difference(subset):
                temp_subset = subset + [feature]
                
                # check if this subset has already been evaluated
                for node in visited:
                    if (set(node) == set(temp_subset)):
                        break
                # if not, ...
                else:
                    # ... mark it as visited
                    visited.append( temp_subset )
                    # ... compute merit
                    merit = self.getMerit(temp_subset)
                    # and push it to the queue
                    self.queue.push(temp_subset, merit)

        self.best_features,self.score = best_subset,best_value
        self.fitting = True
        return self.best_features,self.score
    

    def transform(self, x):

        assert self.fitting, "Please fit the model first"
        if isinstance(x,pd.DataFrame):
            x = x.values
        print(self.best_features)
        return x[:,self.best_features]

from sklearn.feature_selection import SelectKBest,f_regression,RFE,SelectFromModel
from sklearn.linear_model import LinearRegression

class SklearnFeatureSelectionRegression(FeatureSelection):

    def __init__(self, name,n_features=3):
        self.name = name

        if name == 'kbest':
            self.model = SelectKBest(f_regression,k=n_features)
        elif name == 'rfe':
            self.model = RFE(LinearRegression(),n_features_to_select=n_features, step=1)
        elif name == 'select_from_model':
            self.model = SelectFromModel(LinearRegression(),max_features=n_features)
        else:
            raise ValueError("Please specify a valid model name")
        

    def fit(self,X,y):
        self.model.fit(X,y)
        
        return self.model.get_feature_names_out(),0.
    


    def transform(self, X):

        return self.model.transform(X)

def _get_feature_selection(name=''):
    if name == 'cfs':
        return CFS(max_backtrack=10)
    elif name == '':
        return NoneFeatureSelection()
    elif name in ['kbest','rfe','select_from_model']:
        return SklearnFeatureSelectionRegression(name=name)
    else:
        raise ValueError("Please specify a valid feature selection method")


# class KfoldFeatureSelection:

#     def __init__(self,method_name,kfold=5):

#         self.method = _get_feature_selection(method_name)

class DimensionReduction:

    def __init__(self, method_name=''):
        '''
        Init method to reduction dimension
        '''
        self.method_name = method_name


    def pca(self,n_components=2):
        '''
        Unsupervised PCA
        '''

        from sklearn.decomposition import PCA

        _pca = PCA(n_components=n_components)
        pass

import jax.numpy as jnp
from jax import grad, jit
import jax 
from jax import random

class MMDBasedFeatureSelection(FeatureSelection):

    def __init__(self, projected_dim, lr=0.01, epochs=100, gamma=1.0):
        super().__init__()
        self.projected_dim = projected_dim
        self.lr = lr
        self.epochs = epochs
        self.gamma = gamma
        self.W = None

    def rbf_kernel(self, a, b):
        """Gaussian kernel implementation."""
        dist = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-self.gamma * dist)

    def compute_mmd(self, W, X, Y):
        """Compute the Maximum Mean Discrepancy (MMD) in the projected space."""
        XW = X @ W
        YW = Y @ W
        K_XX = self.rbf_kernel(XW, XW)
        K_YY = self.rbf_kernel(YW, YW)
        K_XY = self.rbf_kernel(XW, YW)
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd

    def fit(self, X, y):
        """Fit the projection matrix W to minimize MMD."""
        # Convert data to jax arrays
        X = jnp.array(X)
        Y = jnp.array(y)

        # Initialize projection matrix W
        d = X.shape[1]
        W = random.normal(random.PRNGKey(0),shape=(d, self.projected_dim))

        # Optimization function
        def loss_fn(W):
            return self.compute_mmd(W, X, Y)

        grad_fn = jit(grad(loss_fn))

        # Optimization loop
        for epoch in range(self.epochs):
            gradients = grad_fn(W)
            W -= self.lr * gradients

            if (epoch + 1) % 10 == 0:
                loss = loss_fn(W)
                print(f"Epoch {epoch + 1}/{self.epochs}, MMD Loss: {loss}")

        self.W = W
        self.fitting = True

    def transform(self, X):
        """Transform the dataset using the learned projection."""
        if self.W is None:
            raise ValueError("The model has not been fitted yet. Call fit() before transform().")
        X = jnp.array(X)
        return X @ self.W


       