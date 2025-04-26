import jax.numpy as jnp
import numpy as np
import pdb
class BallTreeNode:
    def __init__(self, points, indices, depth=0, leaf_size=40, metric='euclidean'):
        self.points = points
        self.indices = indices
        self.left = None
        self.right = None
        self.metric = metric
        self.depth = depth

        if len(points) <= leaf_size:
            self.center = jnp.mean(points, axis=0,keepdims=True)
            self.radius = jnp.max(self._pairwise_dist(points, self.center))
        else:
            self._split(leaf_size)
        
    def _pairwise_dist(self, X, Y):
        if isinstance(self.metric, str):
            if self.metric == 'euclidean':
                return jnp.sum((X - Y) ** 2, axis=1)
            # Implement other metrics if needed
            elif self.metric == 'manhattan':
                return jnp.sum(jnp.abs(X - Y), axis=1)
            else:
                raise NotImplementedError
        else:
            return self.metric(X, Y)
        
            

    def _split(self, leaf_size):
        # Splitting the points based on the median of the selected axis
        axis = self.depth % len(self.points[0])  # cycle through dimensions
        sorted_indices = jnp.argsort(self.points[:, axis])
        sorted_points = self.points[sorted_indices]
        sorted_indices = self.indices[sorted_indices]
        median_index = len(sorted_points) // 2

        self.left = BallTreeNode(sorted_points[:median_index], sorted_indices[:median_index], self.depth + 1, leaf_size, self.metric)
        self.right = BallTreeNode(sorted_points[median_index:], sorted_indices[median_index:], self.depth + 1, leaf_size, self.metric)
        self.center = jnp.mean(self.points, axis=0,keepdims=True)
        self.radius = jnp.max(self._pairwise_dist(self.points, self.center))

    def query_radius(self, point, r):
        indices_in_radius = []

        if self.left is None and self.right is None:
            # Leaf node, check each point
            dist = self._pairwise_dist(self.points, point)
            indices = jnp.argwhere(dist<=r)
            indices_in_radius.append(self.indices[indices].T)
            # for i, p in enumerate(self.points):
            #     if jnp.sqrt(self._pairwise_dist(jnp.array([p]), point)) <= r:
            #         indices_in_radius.append(self.indices[i])
        else:
            # Check if hyperspheres intersect
            if self._pairwise_dist(self.center, point) - self.radius <= r:
                if self.left:
                    indices_in_radius.extend(self.left.query_radius(point, r))
                if self.right:
                    indices_in_radius.extend(self.right.query_radius(point, r))

        return indices_in_radius

class BallTree:
    def __init__(self, points, leaf_size=40, metric='euclidean'):
        indices = jnp.arange(len(points))
        self.root = BallTreeNode(points, indices, leaf_size=leaf_size, metric=metric)

    def query_radius(self, point, r):
        return self.root.query_radius(point, r)

# # Example usage
# points = np.random.rand(100000, 2)  # 100 points in 2D
# ball_tree = BallTree(points, leaf_size=5, metric='euclidean')

# query_point = np.random.rand(1, 2)
# radius = 0.2
# indices_in_radius = ball_tree.query_radius(query_point, radius)
# pdb.set_trace()
# print("Query Point:", query_point)
# print("Indices of points within radius:", indices_in_radius)
