import numpy as np
from timeseriespy.utils.utils import *

def dtw_matrix(I, J, w = 0, r = np.inf):

    '''
    Constructs the matrix used to compute the dynamic time warping distance between two arrays.
    Allows for early adandon condition of the algorithm.
    
    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
        w: int or float
        r: float
    
    Returns:
        cum_sum: np.array, shape = (I_length, J_length)
    '''
    # Constructing the matrix
    n, m = len(I), len(J)
    cum_sum = np.ones((n+1, m+1))*np.inf
    cum_sum[0, 0] = 0

    # Filling the matrix with the cumulative sum of the squared distances
    for i in range(1, n+1):
        for j in range(max([1, i-w]), min([m, i+w])+1):
            cost = (I[i-1] - J[j-1])**2
            cum_sum[i, j] = cost + min([cum_sum[i-1, j], cum_sum[i, j-1], cum_sum[i-1, j-1]])

            # Early abandoning condition
            if cum_sum[i, j] > r:
                return cum_sum
    return cum_sum

def dtw_path(matrix):
    '''
    Computes the path of the dynamic time warping algorithm, given a dtw matrix.
    
    Parameters:
        matrix: np.array, shape = (I_length, J_length)
    
    Returns:
        path: list of tuples
    '''
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    path = []
    while i != 0 and j != 0:
        path.append((i - 1, j - 1))
        min_index = argmin([matrix[i-1, j-1], matrix[i-1, j], matrix[i, j-1]])
        if min_index == 0:
            i -= 1
            j -= 1
        elif min_index == 1:
            i -= 1
        else:
            j -= 1
    return path

def dynamic_time_warping(I, J, w = None, r = np.inf):
    '''
    Computes the dynamic time warping distance between two arrays. The window parameter
    is used to limit the search space of the algorithm.

    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
        w: int or float
        r: float
    
    Returns:
        dtw_distance: float
    '''
    if type(I) is not np.ndarray:
        I = np.array(I, dtype = np.float64)
    if type(J) is not np.ndarray:
        J = np.array(J, dtype = np.float64)
    w = set_window(I, J, w = w)
    return dtw_matrix(I, J, w = w, r = r)[-1, -1]

def set_window(I, J, w = None):
    '''
    Sets the window parameter of the dynamic time warping algorithm. The window parameter
    is used to limit the search space of the algorithm.
    
    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
        w: int or float
    
    Returns:
        window: int
    '''
    n, m = I.size, J.size 
    if w is None:
        return int(max([n, m])*0.1)
    elif type(w) is int:
        return w
    elif type(w) is float:
        return int(max([n, m])*w)
    else:
        raise TypeError('window must be an int or float')
