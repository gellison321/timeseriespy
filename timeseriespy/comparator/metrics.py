import numpy as np
from scipy.signal import correlate
from timeseriespy.utils.utils import utils
from timeseriespy.comparator.dynamic_time_warping import dynamic_time_warping

def early_abandon_euclidean_distance(I, J, r = None) -> float:
    '''
    Computes the euclidean distance between two arrays with an early abandon condition.

    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
        r: float
            Early abandon condition.
    
    Returns:
        euclidean_distance: float
    '''
    if r == None:
        return np.linalg.norm(I-J)
    else:
        sum = 0
        for i in range(len(I)):
            sum += (I[i]-J[i])**2
            if sum > r:
                return np.inf
        return sum**0.5
    
def euclidean_distance(arr1, arr2):
    '''
    Computes the euclidean distance between two arrays.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)
    
    Returns:
        euclidean_distance: float
    '''
    return np.linalg.norm(arr1-arr2)
    
def cross_correlation(arr1, arr2, method):
    '''
    Computes the cross-correlation between two arrays.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)
        method: str
            'avg' : average
            'max' : maximum
            'min' : minimum
    Returns:
        cross_correlation: float
    '''
    cases = {'avg': np.mean, 'max' : np.max, 'min' : np.min}
    return cases[method](correlate(arr1, arr2))

def cosine_similarity(arr1, arr2):
    '''
    Computes the cosine similarity between two arrays.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)
    
    Returns:
        cosine_similarity: float
    '''
    return np.dot(arr1, arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2))

def shape_based_distance(arr1, arr2):
    '''
    Computes the shape-based distance between two arrays.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)

    Returns:
        shape_based_distance: float
    '''
    pass

metrics  = {'euclidean' : euclidean_distance,
            'correlation' : cross_correlation,
            'dtw' : dynamic_time_warping,
            'ea_euclidean' : early_abandon_euclidean_distance,
            'cosine' : cosine_similarity,
            }