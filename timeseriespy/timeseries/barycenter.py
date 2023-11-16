from timeseriespy.utils.utils import utils
import numpy as np

#-------------------------------------------#
# Helper methods for barycenter computation #
#-------------------------------------------#

def interpolated_average_barycenter(C, method = 'avg'):
    '''
    Interpolates each time series in a collection of time series to the same length and computes the average time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        method: str
            'avg' : average
            'max' : maximum
            'min' : minimum

    Returns:
        interpolated_average_barycenter: np.array, shape = (length, )
    '''
    do = {'avg' : np.mean, 'max' : np.max, 'min' : np.min}
    length = do[method](list(map(len, C)), dtype = int)
    C = np.array([utils['interpolate'](c, length) for c in C])
    return np.mean(C, axis = 0)

def average_barycenter(C):
    '''
    Computes the average time series in a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        
    Returns:
        average_barycenter: np.array, shape = (length, )
    '''
    return np.mean(C, axis = 0)

def shape_based_barycenter(C):
    '''
    Computes the shape-based barycenter according to the method described in Paparazzo et al.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
    
    Returns:
        shape_based_barycenter: np.array, shape = (length, )
    '''
    pass

def dtw_barycenter():
    pass

def soft_dtw_barycenter():
    pass 

barycenters = {'interpolated_average' : interpolated_average_barycenter,
               'average' : average_barycenter,
              }