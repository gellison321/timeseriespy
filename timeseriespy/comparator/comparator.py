import numpy as np
from timeseriespy.utils.utils import *
from timeseriespy.comparator.metrics import *
from timeseriespy.comparator.bounding import *
import itertools

def query(q, C, w = 0, set_length = False):
    '''

    Queries a time series database for the closest match to a query time series.
    
    Parameters:
        q: array-like, shape = (q_length, )
        C: array-like, shape = (n_instances, length)
        w: int or float
        set_length: int or float
    
    Returns:
        best_index: int
    '''
    
    # if set_length != False:
    #     if type(set_length) in [int, float]:
    #         C = np.array([interpolate(c, set_length) for c in C])
            
    #     else:
    #         raise TypeError('set_length must be an int or float')
        
    # elif any(np.diff(list(map(len, C))) != 0):
    #     raise ValueError('All series must be of the same length.')

    w = set_window(q, C[0], w = w)
    best_so_far = np.inf
    best_index = None
    wedge = DTW_wedge(C, w = w)

    for i in range(len(C)):

        # Compute the lower bound to avoid unnecessary dtw computations
        if LB_Keogh_DTW(q, wedge, r = best_so_far) > best_so_far:
            continue
        dist = metrics['dtw'](q, C[i], w = w, r = best_so_far)
        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index

def pairwise_argmin(C):
    '''
    '''

    for (arr, comparator) in itertools.product([C, C.T], [np.argmin, np.argmax]):
        yield comparator(arr, axis = 1)
        