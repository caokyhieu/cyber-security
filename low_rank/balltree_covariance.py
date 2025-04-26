### this implementation for Ball Tree Covariance matrix in jax

## first we need to protoptype the method need for a covariance matrix
import pdb
class CovarianceMatrix:

    def __init__(self, points):
        '''
        points: a list of points
        
        '''
        self.points = points
        pass 

    def covariance(self):
        '''
        This method will return the covariance matrix of the points
        '''
        
        pass 

from sklearn.neighbors import BallTree
# from low_rank.ball_tree import BallTree
import numpy as np
from sklearn.metrics.pairwise import distance_metrics
import jax.numpy as jnp
import jax.scipy as jsp
from itertools import product
from jax.experimental.sparse import BCOO
from low_rank.ga import GeneticAlgorithm
from low_rank.ga_numpy import NumpyGeneticAlgorithm
import re
from tqdm import tqdm
class BallTreeCovarianceMatrix(CovarianceMatrix):

    def __init__(self, points, random_state=32,metric='haversine',
                 upper_bound=3000,lower_bound=2000,dtype=jnp.float16,
                 radius=0.2, num_iter=100):
        super(BallTreeCovarianceMatrix, self).__init__(points)
        ## build the ball tree
        self.points = self.points.astype(dtype=dtype)
        ## create a mapping from values to index
        # self.mapping_index = {
        #     tuple(value.round(decimals=5).tolist()): index for index, value in enumerate(self.points)
        # }
        ## check the uniqueness of values
        # assert len(self.points) == len(self.mapping_index), "Please check the uniqueness of the points"
        # print(f"length dataset: {len(self.points)} == {len(self.mapping_index)} length mapping")

        np.random.RandomState(random_state)
        self.tree = BallTree(self.points, leaf_size=2,metric=metric) 
        self.upper_bound = upper_bound   
        self.lower_bound = lower_bound 
        # self.values, self.index_for_sparse  = None, None
        self.metric_func = True
        if isinstance(metric, str):
            self.metric = distance_metrics()[metric]
            self.metric_func = False
        else:
            self.metric = metric
        self.dtype= dtype
        ## avaiable indexes
        self.available_indexes =  set(range(len(self.points)))
        self.num_iter = num_iter
        def fitness_func(solution):
            # Calculating the fitness value of each solution in the current population.
            # The fitness function calulates the sum of products between each input and its corresponding weight.
            items  = self.tree.query_radius(solution.reshape(1,-1), r=radius)
            fitness = 0.
            ## relax the constraints
            # if self.upper_bound>len(items[0])>self.lower_bound:
            if len(items)>0:
                # items = jnp.hstack(items).tolist()
                selected_indexes = set(items[0]).intersection(self.available_indexes)
                fitness = len(selected_indexes)
            return -fitness
        
        self.ga_instance = GeneticAlgorithm(num_generations=100,
                                            population_size=100,
                                            solution_dim=self.points.shape[1],
                                            mutation_rate=0.05,
                                            num_parents=50,
                                            fitness_func=fitness_func,
                                            tournament_size=10)
                                
        self.radius = radius

        
    def _search_algorithms(self, radius):
        '''
        This method will return the indices of the points that are within the radius
        '''
        ## first we need to find the intial centroid
        # centroid_index = np.random.choice(self.points.shape[0])
        centroid = self.points.mean(axis=0)
        total_indexes = set([i for i in range(len(self.points))])
        # reindex = []
        values = []
        index_for_sparse = []
        i = 0 
        pbar = tqdm(total=100)
        while len(total_indexes)>0:
            items = self.tree.query_radius(centroid.reshape(1,-1), r=radius)
            ##handle items for self implement
            # if len(items)>0:
            #     # pdb.set_trace()
            #     items = jnp.hstack(items).tolist()
            if len(items)==0:
                items = [[]]
            #     centroid_index = np.random.choice(self.points.shape[0])
            #     centroid = self.points[centroid_index:centroid_index,:]
            #     continue
            # items = jnp.hstack(items)
            ### relax this condition
            # if self.upper_bound>len(items[0])>self.lower_bound:
                ## accept this centroid and data
                # reindex += items[0].tolist()
            # pdb.set_trace()
            selected_indexes = set(items[0]).intersection(self.available_indexes)
            ## update the available indexes
            self.available_indexes = self.available_indexes.difference(selected_indexes)
            ## covarinace matrix
            if len(selected_indexes)>0:
                if self.metric_func:
                    cov = self.metric(self.points[jnp.array(list(selected_indexes))],self.points[jnp.array(list(selected_indexes))])
                else:
                    cov = self.metric(self.points[jnp.array(list(selected_indexes))])
            else:
                cov = jnp.array([])
            values += cov.astype(dtype=self.dtype).flatten().tolist()
            ## update index
            index_for_sparse += list(product(selected_indexes,selected_indexes))
            ### get new centroid
            ### here should update to use GE algorithm
            centroid,_,_ = self.ga_instance.run(verbose=False)
            ## update memory usage
            # self.print_memory_usage(values)
            i+=1
            pbar.update(1)
            if i>self.num_iter:
                break
            
            # if self.ga_instance.best_solution_generation != -1:
            #     print("Best fitness value reached after {ga_instance.best_solution_generation} generations.")
            #     centroid = self.ga_instance.best_solution_generation.reshape(1,-1)
            # else:
            #     print("GE algorithm didn't find a solution for the problem after {ga_instance.num_generations} generations. Using random choice for centroid")
            #     random_radians = np.random.uniform(0, np.pi)
            #     centroid_index += np.array([[0,radius]])
            #     ## rotation
            #     centroid_index = centroid_index @ np.array([[np.cos(random_radians), -np.sin(random_radians)], [np.sin(random_radians), np.cos(random_radians)]])
        pbar.close()
        return values, index_for_sparse
    
    def print_memory_usage(self,values):
        pattern = r"\d+"
        match = re.search(pattern,str(self.dtype))
        str_type = int(match.group())
        # shape = len(values[0])
        print(f"Length: {len(values)},Size: {len(values)*str_type  / 8 /1e9:.2f} GB")
    
    def covariance(self):
        # if (self.values is None) and (self.index_for_sparse is None):

        values, index_for_sparse = self._search_algorithms(radius=self.radius)
        self.bcoo = BCOO((values,index_for_sparse  ), shape=(len(self.points), len(self.points))).todense()
        ## fillout the diagonal
        self.bcoo = self.bcoo + jnp.eye(len(self.bcoo))
        self.bcoo = jnp.where(self.bcoo==0,jnp.inf,self.bcoo)
        ## substract back
        self.bcoo = self.bcoo - jnp.eye(len(self.bcoo))

        # return self.bcoo
    

    # def filter_matrix(self,value_to_index):
    #     '''
    #     This method helps to slice the coariane matrix
    #     base on the indexes
    #     '''
    #     assert (self.values is not None) and (self.index_for_sparse is not None), "Please run covariance method first"
    #     value_to_index = jnp.array(value_to_index,dtype=self.dtype)
    #     ## filter the values
    #     # filtered_values = []
    #     # filtered_indexes = []
    #     indexes = [self.mapping_index[tuple(value.round(decimals=5).tolist())] for value in value_to_index]
    #     assert len(indexes) == len(value_to_index), 'not equal length'
        
    #     # for value, index in zip(self.values,self.index_for_sparse):
    #     #     if (index[0] in indexes) and (index[1] in indexes):
    #     #         filtered_values.append(value)
    #     #         filtered_indexes.append(index)
        
    #     return self.bcoo[indexes,:][:,indexes].todense()
    
    def filter_matrix_index(self,indexes):
        '''
        This method helps to slice the covariance matrix
        base on the indexes
        '''
        # assert (self.values is not None) and (self.index_for_sparse is not None), "Please run covariance method first"
        # value_to_index = jnp.array(value_to_index,dtype=self.dtype)
        # ## filter the values
        # # filtered_values = []
        # # filtered_indexes = []
        # indexes = [self.mapping_index[tuple(value.round(decimals=5).tolist())] for value in value_to_index]
        # assert len(indexes) == len(value_to_index), 'not equal length'
        
        # for value, index in zip(self.values,self.index_for_sparse):
        #     if (index[0] in indexes) and (index[1] in indexes):
        #         filtered_values.append(value)
        #         filtered_indexes.append(index)
        
        return self.bcoo[indexes,:][:,indexes]
        
        