import numpy as np
from timeseriespy.utils.utils import utils

def match_lengths(I, J):
    '''
    Interpolates the longer of two time series down to the length of the smaller one.
    
    Parameters:
        I: array-like, shape = (I_length, )
        J: array-like, shape = (J_length, )
    
    Returns:    
        I: array-like, shape = (min(I_length, J_length), )
        J: array-like, shape = (min(I_length, J_length), )
    '''

    i_length, j_length = len(I), len(J)

    # Interpolating the longer of the two time series down to the length of the smaller one
    if i_length < j_length:
        J = utils['interpolate'](J, i_length)
    elif i_length > j_length:
        I = utils['interpolate'](I, j_length)
    return I, J


def euclidean_wedge(C): 
    '''
    Computes the wedge of a series. The wedge is the difference between the first and last
    points of the series.

    Parameters:
        C: array-like, shape = (n_instances, length)

    Returns:
        lower: array-like, shape = (length, )
    '''
    try:
        return np.min(C, axis = 0), np.max(C, axis = 0)
    except:
        raise ValueError('Data must be two dimensionsal.')


def DTW_wedge(C, w = 0):
    '''
    Computes the wedge of a series. The wedge consists of two arrays, one of the maximum value of the
    candidate series at within a window of each index and one of the minimum value of the candidate 
    series within a window of each index.


    Parameters:
        C: array-like, shape = (n_instances, length)
        w: int

    Returns:
        lower: array-like, shape = (length, )
    '''
    # if np.any(np.diff(list(map(len, C)))!=0):
    #     raise ValueError('All series must be of the same length.')
    
    L, U = np.min(C, axis = 0), np.max(C, axis = 0)
    lower, upper = [],[]

    for i in range(len(C[0])):
        l = max(0, i-w)
        u = min(len(U), i+w+1)
        lower.append(min(L[l:u]))
        upper.append(max(U[l:u]))
    return np.array(lower), np.array(upper)



def LB_Keogh_squared_distances(q, C, r = np.inf):
    '''
    Computes the LB Keogh lower bound of a sequence being compared to a set of sequences.

    Parameters:
        q: array-like, shape = (q_length, )
        C: array-like, shape = (n_instances, length)
        r: float

    Returns:
        sum: float
    '''
    r = r if r == np.inf else r**2
    q, C = match_lengths(q, C)

    sum = 0
    for i in range(len(q)):
        U = np.min(C[:,i])
        L = np.max(C[:,i])
        if q[i] < L:
            sum += (q[i] - L)**2
        elif q[i] >= U:
            sum += (q[i] - U)**2
        if sum > r:
            return sum**0.5
    return sum**0.5
 

def LB_Keogh_DTW(q, wedge, r = np.inf):
    '''
    Computes the LB Keogh lower dynamic time warping bound of a sequence being compared to a set of sequences.

    Parameters:
        q: array-like, shape = (q_length, )
        wedge: tuple of arrays, shape = (2, q_length)
        r: float
    
    Returns:
        sum: float
    '''
    r = r**2
    L,U = wedge[0], wedge[1]
    sum = 0
    for i in range(len(q)):
        if q[i] < L[i]:
            sum += (q[i] - L[i])**2
        elif q[i] > U[i]:
            sum += (q[i] - U[i])**2
        if sum > r:
            return sum**0.5
    return sum**0.5

bounding = {'match_lengths': match_lengths,
            'euclidean_wedge': euclidean_wedge,
            'DTW_wedge': DTW_wedge,
            'LB_Keogh_squared_distances': LB_Keogh_squared_distances,
            'LB_Keogh_DTW': LB_Keogh_DTW}