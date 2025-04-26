
# from ott.core import gromov_wasserstein as gw
from ott.geometry import pointcloud,grid,costs
# from ott.core import sinkhorn,sinkhorn_lr
# from ott.core import linear_problems
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
# from ott.linear_problem import
import jax.numpy as jnp
import pdb
import jax

def calculate_optimal_transport_plan_sinkhorn(x,y):
    """
    calculate optimal transport plan
    Arguments:
        x {numpy array} -- source data
        y {numpy array} -- target data
    Returns:
        numpy array -- optimal transport cost matrix
    """
    ## care about covariate shift first
    if x.ndim==1 and y.ndim==1:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    geomxy =  pointcloud.PointCloud(x=x, y=y,epsilon=.1,cost_fn=costs.Cosine())
    # Define a linear problem with that cost structure.
    
    ot_prob = linear_problem.LinearProblem(geomxy)
    
    solver = sinkhorn.Sinkhorn( 
                                lse_mode=False,
                                threshold=1e-6,
                                 max_iterations=1000,
                                    norm_error=1e-3,
                                    initializer='sorting',
                               )
    # Solve OT problem
    ot = solver(ot_prob)
    d  = jnp.multiply(ot.matrix,ot.geom.cost_matrix)
    
    return d


    