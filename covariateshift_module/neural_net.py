import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataset import PhotoDataset, jax_collate_fn
from jax.tree_util import tree_map
import optax
import pdb
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = jnp.array([label for _, label in dataset])
        self.key = jax.random.PRNGKey(123)

    def __getitem__(self, index):
        anchor, anchor_label = self.dataset[index]

        positive_indices = jnp.nonzero(self.labels == anchor_label)[0]
        self.key, subkey = jax.random.split(self.key)
        positive_index = positive_indices[jax.random.randint(key=subkey, minval=0, maxval=len(positive_indices), shape=())]

        negative_indices = jnp.nonzero(self.labels != anchor_label)[0]
        self.key, subkey = jax.random.split(self.key)
        negative_index = negative_indices[jax.random.randint(key=subkey, minval=0, maxval=len(negative_indices), shape=())]

        positive, _ = self.dataset[positive_index]
        negative, _ = self.dataset[negative_index]

        return anchor, positive, negative, anchor_label

    def __len__(self):
        return len(self.dataset)

class MLP(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        intermediate_outputs = []
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                intermediate_outputs.append(x)
        return intermediate_outputs, nn.sigmoid(x)

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = jnp.sum(jnp.square(anchor - positive), axis=-1)
    neg_dist = jnp.sum(jnp.square(anchor - negative), axis=-1)
    return jnp.maximum(pos_dist - neg_dist + margin, 0.0)

def logistic_regression_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

class Trainer(object):
    def __init__(self, key=123, layers=[256, 128, 64, 32, 1], input_dim=12, learning_rate=1e-2, batchsize=32):
        self.key = jax.random.PRNGKey(key)
        self.model = MLP(features=layers)
        params = self.model.init(self.key, jnp.ones((1, input_dim)))
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(params)
        self.params = params
        self.batchsize = batchsize

    def loss_fn(self, params, batch):
        anchor, positive, negative, anchor_label = batch
        anchor_emb, anchor_out = self.model.apply(params, anchor)
        positive_emb, positive_out = self.model.apply(params, positive)
        negative_emb, negative_out = self.model.apply(params, negative)

        trip_loss = tree_map(lambda x, y, z: triplet_loss(x, y, z), anchor_emb, positive_emb, negative_emb)
        # pdb.set_trace()
        trip_loss = jnp.sum(jnp.array(trip_loss))
        positive_label = anchor_label
        negative_label = 1. - anchor_label
        X = jnp.vstack([anchor_out, positive_out, negative_out])
        y = jnp.vstack([anchor_label.reshape(-1,1), positive_label.reshape(-1,1), negative_label.reshape(-1,1)])
        log_loss = logistic_regression_loss(y, X)
        loss = log_loss + trip_loss

        return loss

    def update(self, params, opt_state, batch):
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def train_step(self, dataloader):
        loss = 0.
        for data in dataloader:
            new_loss, self.params, self.opt_state = self.update(self.params, self.opt_state, data)
            loss += new_loss

        return loss / len(dataloader)

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            loss = self.train_step(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {loss:.3f}")

    def predict(self, X):
        X = jnp.array(X)
        return jnp.astype(self.model.apply(self.params, X)[-1], jnp.int64)

    def predict_proba(self, X):
        X = jnp.array(X)
        probs = jnp.zeros((len(X), 2))
        probs = probs.at[:, 1].set(self.model.apply(self.params, X)[-1].flatten())
        probs = probs.at[:, 0].set(1. - probs[:, 1])
        return probs

    def fit(self, X, y, num_epochs=200):
        dataset = TripletDataset(PhotoDataset(X, y))
        dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True, collate_fn=jax_collate_fn)
        # pdb.set_trace()
        self.train(dataloader, num_epochs)
