import numpy as np
from tqdm import tqdm 
from metric_learn import NCA
import ot
from sklearn.neighbors import KernelDensity
from itertools import product
import jax.numpy as jnp
from jax import grad, jit, vmap,value_and_grad,random ,lax
import jax
lambd = 0.1
BANDWIDTH = 0.2



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
    # batches = [batch for batch in train_dataloader if batch[0].shape[0]==batch_size]
    batches = [batch for batch in train_dataloader]
    ## patch for last batch
    batches[-1] = pad_batch(batches[-1], batchsize=batch_size)
    
    batches = [(jnp.expand_dims(el,axis=0) for el in batch) for batch in batches]
    batches = list(zip(*batches))
    batches = [jnp.concatenate(b,axis=0) for b in batches]
    return batches   

def calculate_cost_matrix_NCA(X_train,X_test):
    ## modify the code
    y_train = np.array(['train' for i in range(len(X_train))])
    y_test = np.array(['test' for i in range(len(X_test))])
    # if len(X_train)<10000:
    nca = NCA(random_state=42)
    
    nca.fit(np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0))
    pairs = product(X_train[:,np.newaxis,:],X_test[:,np.newaxis,:])
    pairs = list(map(lambda x: np.vstack(x), pairs))
    pairs = np.array(pairs)
    dist = nca.pair_distance(pairs)
    dist = dist.reshape(len(X_train),-1)
    # else:
    #     nca = NeuralNCA(input_dim=X_train.shape[-1],embedding_dim=X_train.shape[-1],key = random.PRNGKey(42))
    #     nca.train(np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0),batchsize=512,n_epochs=5)
    #     dist = nca.pair_distance(X_train,X_test)
    return dist

# def get_distance(X_train,X_test,n_iter=100):
#     nca = NCA(random_state=42)
#     y_train = np.array(['train' for i in range(len(X_train))])
#     y_test = np.array(['test' for i in range(len(X_test))])
#     nca.fit(np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0))
#     dist = 0. 
#     ## need to fix the loop
#     for i in tqdm(range(n_iter)):
#         test_samples = random.choices(range(len(X_test)),k=len(X_train))
#         X_pairs = np.concatenate((X_train[:,np.newaxis,:],X_test[test_samples,np.newaxis,:]),axis=1)
#         dist += nca.score_pairs(X_pairs)
    
#     return dist/n_iter

def compute_wasserstein_distance(X1,X2):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH).fit(X1)
    prob1 = np.exp(kde.score_samples(X1))
    kde = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH).fit(X2)
    prob2 = np.exp(kde.score_samples(X2))
    # loss matrix
    M = ot.dist(X1, X2)
    Gs = ot.sinkhorn(prob1, prob2, M, lambd, verbose=True)

    return Gs.sum(1) 

from functools import partial
from utils.dataset import PhotoDataset,jax_collate_fn
from torch.utils.data import Dataset,DataLoader
from flax import linen as nn
from flax.training.train_state import TrainState
from optax import adam

class SimilarNetwork(nn.Module):
    layers_config: list

    def setup(self):
        self.layers = []
        self.residuals = []
        for nodes in self.layers_config:
            self.layers+= (nn.Dense(nodes),nn.relu,nn.Dense(nodes),nn.relu)
            self.residuals+= (nn.Dense(nodes),)
         
        self.layers+= (nn.Dense(nodes),)

    def __call__(self, x):
        for i,res in enumerate(self.residuals):
            residual = res(x)
            x = self.layers[i*4](x) 
            x = self.layers[i*4+1](x)
            x = self.layers[i*4+2](x) 
            x = self.layers[i*4+3](x) + residual
       
        x = self.layers[-1](x)
        return x
    

    
class ClassifierLayer(nn.Module):
    features_dim: int
    def setup(self):
        self.classifier = nn.Dense(self.features_dim)

    def __call__(self, x):
        x = self.classifier(x)
        return x

class NeuralNCA:

    def __init__(self,input_dim,embedding_dim,key,num_layers=2,layers=[128, 64, 32],**kwargs):

        self.key = key
        # self._params = [self.random_layer_params(input_dim,embedding_dim,key)] 
        # for _ in range(num_layers -1 ):
        #     key,_ = random.split(key)
        #     self._params.append(self.random_layer_params(embedding_dim,embedding_dim,key))
        self._params = None
        
        self.model = SimilarNetwork(layers_config=layers)
        # self.classifier = ClassifierLayer(features_dim=embedding_dim)
        self._mean = None
        self._stdev = None

    def random_layer_params(self, m, n, key, scale=1e-2):
        return scale * random.normal(key, (n, m))
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def stdev(self):
        return self._stdev
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value
    

    def _pairwise_distance(self,x):
        
        dot = jnp.matmul(x,x.T)
        norm_sq = jnp.diag(dot)
        dist = norm_sq[:,None] + norm_sq[None,:] - 2*dot
        dist = jnp.clip(dist,a_min=0.)
        return dist
    
    def _forward(self, x, params):
        # for i, param in enumerate(params):
        #     if i!=(len(params)-1):
        #         x = nn.relu(jnp.matmul(x, param.T))
        #     else:
        #         x = jnp.matmul(x, param.T)
        # return x

        return self.model.apply(params, x)


    def loss(self,params,x,y_mask):
        """
        calculate the pair loss
        """
        # compute pairwise squared Euclidean distances
        # in transformed space
        # embedding = jnp.matmul(x, params.T)
        embedding = self._forward(x, params)
        distances = self._pairwise_distance(embedding)

        # fill diagonal values such that exponentiating them
        # makes them equal to 0
        diag_indices = jnp.diag_indices_from(distances)
        distances = distances.at[diag_indices].set(jnp.inf)

        # compute pairwise probability matrix p_ij
        # defined by a softmax over negative squared
        # distances in the transformed space.
        # since we are dealing with negative values
        # with the largest value being 0, we need
        # not worry about numerical instabilities
        # in the softmax function
        p_ij = nn.softmax(-distances,axis=-1)

        # for each p_i, zero out any p_ij that
        # is not of the same class label as i
        p_ij_mask = p_ij * y_mask

        # sum over js to compute p_i
        p_i = p_ij_mask.sum(axis=1)

        # compute expected number of points
        # correctly classified by summing
        # over all p_i's.
        # to maximize the above expectation
        # we can negate it and feed it to
        # a minimizer
        # for numerical stability, we only
        # log_sum over non-zero values
        p_i = jnp.where(x > 0, x, 1.0)
        classification_loss = -jnp.log(p_i).sum()

        # to prevent the embeddings of different
        # classes from collapsing to the same
        # point, we add a hinge loss penalty
        distances = distances.at[diag_indices].set(0.)
        margin_diff = (1 - distances) * (~y_mask)
        hinge_loss = jnp.power(jnp.clip(margin_diff, a_min=0),2).sum(axis=-1).mean()

        # sum both loss terms and return
        loss = classification_loss + hinge_loss
        return loss
    

    def train(self,x,y,batchsize=128,stepsize=0.01,n_epochs=200):
        """
        train the model
        """
        # compute mean and stdev of the data
        self._mean = x.mean(axis=0)
        self._stdev = x.std(axis=0)
        # self._mean = 0.
        # self._stdev = 1.
        # normalize the data
        x = (x - self.mean) / self.stdev

        # create a data loader
        # for batched training
        n = x.shape[0]
        idx = jnp.arange(n)

        ## create training state
        params = self.model.init(self.key, x)
        state = TrainState.create(apply_fn=self.model.apply, params=params, tx=adam(stepsize))

        # @jit
        # def update(params, x, y_mask):
        #     loss,loss_grad = value_and_grad(self.loss)(params, x, y_mask)
        #     return loss,[params[i] - stepsize * loss_grad[i] for i in range(len(loss_grad))]
        
        @jit
        def update(state, x, y_mask):
            loss,loss_grad = value_and_grad(self.loss)(state.params, x, y_mask)
            return loss,state.apply_gradients(grads=loss_grad)
        
        dset = PhotoDataset(x,y)
        dloader = DataLoader(dset,batch_size=batchsize, shuffle=True,collate_fn=jax_collate_fn)
        
        @jit
        def epoch_train(state, data):
            def scan_fn(carry, batch):
                batch_x,batch_y = batch
                _state = carry

                batch_y_mask = (batch_y[:,None] == batch_y[None,:])
                batch_loss,_state = update(_state, batch_x, batch_y_mask)

                return _state, batch_loss  # Return updated parameters and a dummy second argument
            # Use `lax.scan` to loop over the data (faster than a Python loop)
            final_params,total_loss = lax.scan(scan_fn, state, data)
            return jnp.sum(total_loss)/len(dset),final_params

        # # @jit
        # # def epoch_train(state):
        # #     d = iter(dloader)
        # #     def body_fn(i, val):
        # #         loss_sum,_state = val
        # #         batch_x,batch_y = d.__next__()
        # #         batch_y_mask = (batch_y[:,None] == batch_y[None,:])
        # #         batch_loss,_state = update(_state, batch_x, batch_y_mask)
        # #         loss_sum += batch_loss                
        # #         return loss_sum,_state

        #     return lax.fori_loop(0, round(len(idx)//batchsize + 0.5), body_fn, (0.0,state))
        epoch_iters = tqdm(range(n_epochs))
        # key ,_= random.split(self.key)
        params = state
        for epoch in epoch_iters:
            train_data = generate_batch(dloader)
            epoch_loss,params = epoch_train(params, train_data)            
            epoch_iters.set_description(f"Epoch {epoch+1}/{n_epochs}, Epoch Loss: {epoch_loss:.4f}")
        
        ## reassign params with new values
        self._params = params.params
    def __call__(self,x):
        """
        transform data to learned space
        """
        assert self._mean is not None and self._stdev is not None, "Not trained model yet"
        x = (x - self.mean) / self.stdev

        return self._forward(x,self.params)
    
    def pair_distance(self,x1,x2,batchsize=128):
        """
        compute pairwise distance in the learned space
        """
        x1_emb = self(x1)
        x2_emb = self(x2)
        # n_iters = jnp.ceil(len(x1_emb)/batchsize).astype(int)
        dset = PhotoDataset(x1_emb,jnp.ones((len(x1_emb)))) ## dummy y
        dloader = DataLoader(dset,batch_size=batchsize, shuffle=False,collate_fn=jax_collate_fn)
        train_data = generate_batch(dloader)

        @jit 
        def iter_distance(data):
            def scan_fn(carry, batch):
                batch_x,_ = batch
                d = (( batch_x[:, None, :] - x2_emb[None, :, :]) ** 2).sum(-1)
                # d = d.sum(-1,keepdims=True)
                return None, d
            # Use `lax.scan` to loop over the data (faster than a Python loop)
            _,distance = lax.scan(scan_fn, None, data)
            return distance
        
        distance = iter_distance(train_data)
        distance = distance.reshape(-1, len(x2_emb))
        ## return the mathc shape
        distance = distance[:len(x1_emb),...]
        return distance
            
        # @jit
        # def iter_distance():
        #     dat = iter(dloader)
        #     def body_fn(i, val):
        #         distance = val
        #         batch_x = dat.__next__()
        #         d = (( batch_x[:, None, :] - x2_emb[None, :, :]) ** 2).sum(-1)
        #         d = d.sum(-1,keepdims=True)
        #         distance = jax.lax.dynamic_update_slice_in_dim(distance, d, i*batchsize,axis=0)
        #         return distance
        #     return lax.fori_loop(0, n_iters, body_fn, jnp.zeros((x1_emb.shape[0],1)))

        # distance = iter_distance()
        # return distance



