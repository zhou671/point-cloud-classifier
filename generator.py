import numpy as np
from scipy.spatial.transform import Rotation as R

NUM_POINT = 1024
TRANSLATE_LOW = -20.0
TRANSLATE_HIGH = 20.0

def generate_circle_point_cloud():
    """Generates a point cloud of a unit circule in 3D with size of 1024 in np array
    
    Same as above
    
    Args:
        None
    
    Returns:
        Same as introduction 
    """
    res = np.random.normal(size = (NUM_POINT, 3))
    norms = np.linalg.norm(res, axis = 1)
    norms = norms.reshape((-1, 1))
    return res / norms

def generate_cube_point_cloud():
    """Generates a point cloud of a unit cube in 3D with size of 1024 in np array
    
    Same as above
    
    Args:
        None
    
    Returns:
        Same as introduction    
    """
    seeds = np.random.randint(6, size=NUM_POINT)
    base = np.random.uniform(low=-1.0, high=1.0, size=(NUM_POINT, 3))
    const1 = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    const2 = np.array([[0,1,1], [0,1,1], [1,0,1], [1,0,1], [1,1,0], [1,1,0]])
    comp1 = const1[seeds]
    comp2 = const2[seeds]
    comp2 = np.multiply(comp2, base)
    return comp1 + comp2
    

def random_transform(point_cloud):
    """return point_could after scaling, rotating and translating
    
    scaling is a N'(1.0, 0.1)' for x, y, z
    rotating and translating are uniformly distributed
    
    Args:
        point_could: object point_cloud
    
    Returns:
        Same as introduction    
    """
    
    scaling = np.random.normal(loc = 1.0, scale = 0.1, size = 3)
    point_cloud = scaling * point_cloud 
    
    rotating = R.random(num = 1).as_matrix()[0]
    point_cloud = np.matmul(point_cloud, rotating)
    
    translating = np.random.uniform(low=TRANSLATE_LOW, high=TRANSLATE_HIGH, size = 3)
    point_cloud = point_cloud + translating
    
    return point_cloud

def generate_data(size):
    seeds = np.random.randint(2, size=size)
    data = np.array([random_transform(generate_circle_point_cloud()) if seed == 0 else random_transform(generate_cube_point_cloud()) for seed in seeds])
    return data, seeds

