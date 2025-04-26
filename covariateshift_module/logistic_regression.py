from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.signal import savgol_filter
from sklearn.metrics import f1_score
from covariateshift_module.neural_net import Trainer
import pdb

'''
We can build a logistic regression and classify the data is from source or target
Estimate the density ratio by r(x) = \frac{q(x)}{1-q(x)]} with q is the probability of x is from target (output of logistic regression)

'''
## now I want instead of Logistic regression, I can use a neural network instead
## can use neural network in sklearn
import jax.numpy as jnp
class RegressionClassifier:
    def __init__(self,**kwargs):
        # self.model = LogisticRegression()
        #
        self.n_model = 3
        # layers = [1024,512, 256,128,64]
        # self.models = [MLPClassifier(solver='lbfgs', alpha=1e-5,
        #                 hidden_layer_sizes=layers[:i//2], random_state=1,activation='tanh') if i%2==0
        #                 else LogisticRegression() for i in range(self.n_model)]
        # self.models = [SGDClassifier(loss='log_loss',max_iter=1000, tol=1e-3) for _ in range(self.n_model)]
        input_dim = kwargs.get('input_dim',5)
        self.models = [Trainer(batchsize=256, input_dim=input_dim) for _ in range(self.n_model)]
        self.n_target = -1

        # kernel = 1. * RBF(1.)
        # self.models = [GaussianProcessClassifier(kernel=kernel,random_state=i) for i in range(self.n_model)]

    def fit(self, X_source, X_target):
        ## now we will sub sample X_target into 10 different subsets
        ## and train 10 different models
        ## first shuffle X_target
        indexes = np.arange(len(X_target))
        np.random.shuffle(indexes)
        X_target = X_target[indexes]
        self.n_target = len(X_target)
        ## using kfold split
        kf = KFold(n_splits=self.n_model)
        self.weights = np.ones((self.n_model), dtype=np.float64)/self.n_model

        for i, (train_index, test_index) in enumerate(kf.split(X_source)):
            ## random select 1/2 data points from X_target
            idxs = np.random.choice(X_target.shape[0], X_target.shape[0] // 2, replace=False)
            X_target_sub = X_target[idxs]
            # X_target_sub = X_target
            y_target = np.ones((len(X_target_sub),), dtype=np.float64)
            ## select random source with the same shape as X_target_sub
            # idxs = np.random.choice(X_source.shape[0], min(X_target_sub.shape[0],X_source.shape[0]), replace=False)
            X_source_sub = X_source[train_index]
            y_source = np.zeros((len(X_source_sub),), dtype=np.float64)
            X = np.vstack((X_source_sub, X_target_sub))
            y = np.hstack((y_source, y_target))
            # if i>0:
            #     ## add one more column for X  as the prediction from previous models
            #     for j in range(i):
            #         X = np.hstack((X, self.models[j].predict_proba(X)[:,1].reshape(-1,1)))
            ## fit the model
            self.models[i].fit(X, y,num_epochs=30)
            ## compute error for this model compared with previous fitted models
            # if i > 0:
            #     ## prepare data for test
            #     target_idxs = np.random.choice(X_target.shape[0], X_target.shape[0] // 2, replace=False)
            #     X_target_sub = X_target[target_idxs]
            #     X = np.vstack((X_source[test_index], X_target_sub))
            #     y = np.hstack((np.zeros((len(X_source[test_index]),)), np.ones((len(X_target_sub),))))
            #     all_previous_score = []
            #     # ################################################################
            #     for j in range(i):

            #         all_previous_score.append(f1_score(y, self.models[j].predict(X),zero_division=1))
            #         # X = np.hstack((X, self.models[j].predict_proba(X)[:,1].reshape(-1,1)))
            #     # print(f"all score: {all_previous_score}")

            #     all_previous_score = np.mean(all_previous_score)
            #     ## compute error for this model
            #     current_score = f1_score(y, self.models[i].predict(X),zero_division=1)
            #     # ##
            #     self.weights[i] = current_score / all_previous_score
        ## eliminate models and weights which have weight not floar (nan or inf)
        # self.models = [model for i,model in enumerate(self.models) if np.isfinite(self.weights[i])]
        # self.weights = [1e-6 if not np.isfinite(weight) else weight for weight in self.weights]
        ## normalzie weights

        self.weights = self.weights / np.sum(self.weights)

    def compute_weights(self, X):
        result = np.zeros((len(X),), dtype=np.float64)
        for i, model in enumerate(self.models):
            # if i>0:
            #     X = np.hstack((X, self.models[i-1].predict_proba(X)[:,1].reshape(-1,1)))
            pos_prob = model.predict_proba(X)[:, 1]
            pos_prob = np.clip(pos_prob, 1e-6, 1-1e-6)
            result += self.weights[i] * pos_prob / (1. - pos_prob)
            result *= self.n_target / len(X)
        return np.array(result, dtype=np.float64)
