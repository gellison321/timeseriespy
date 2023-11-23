import numpy as np
from timeseriespy.utils.utils import utils
from timeseriespy.comparator.metrics import metrics
from timeseriespy.comparator.bounding import bounding
import itertools
import multiprocessing
import os
from itertools import repeat


###################################################################################
# query() function returns the index of the best matching series in the database. #                                               # 
###################################################################################

def query_worker(args):
    '''
    Processes a single query against a time series data point.

    Parameters:
    args (tuple): Contains query series, target series, wedge, best distance so far, and window size.

    Returns:
    float: The distance measure between the query and the series.
    '''
    q, C, w = args
    return metrics['dtw'](q, C, w=w)


def parallelized_query(*args):
    '''
    Executes a parallelized query over multiple time series data points.

    Parameters:
    args (tuple): Contains the query series, collection of series, wedge, window size, and best distance so far.

    Returns:
    int: The index of the best matching series.
    '''

    q, C, w, pool_size = args

    pool = multiprocessing.Pool(processes=pool_size)
    args = [(q, C[i], w) for i in range(len(C))]
    results = pool.map(query_worker, args)
    pool.close()
    pool.join()

    best_so_far = np.inf
    best_index = None

    for i, dist in enumerate(results):
        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index

def sequential_query(*args):
    '''
    Executes a sequential query over multiple time series data points. Allows for early abandon condition.
    
    Parameters:
        args (tuple): Contains the query series, collection of series, wedge, window size,
        and best distance so far.
        
    Returns:
        int: The index of the best matching series.
    '''

    q, C, wedge, w = args

    best_so_far = np.inf
    best_index = None

    for i in range(len(C)):

        if bounding['LB_Keogh_DTW'](q, wedge, r = best_so_far) > best_so_far:
            continue

        dist = metrics['dtw'](q, C[i], w = w, r = best_so_far)

        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index



def query(q, C, w = 0, parallel_cores = 1):
    '''
    Queries a time series database for the closest match to a query time series.
    
    Parameters:
        q: array-like, shape = (q_length, )
        C: array-like, shape = (n_instances, length)
        w: int or float
        set_length: int or float
        parallel: bool
    
    Returns:
        best_index: int
    '''
    w = utils['set_window'](q, C[0], w = w)
    wedge = bounding['DTW_wedge'](C, w = w)

    if parallel_cores == 'all':
        parallel_cores = max(1, os.cpu_count() - 1)
        return parallelized_query(q, C, w, parallel_cores)

    elif parallel_cores > 1:
        if parallel_cores > os.cpu_count():
            print('Warning: Number of parallel cores exceeds number of available cores. \
                  Using all available cores.')
            parallel_cores = os.cpu_count() - 1
        return parallelized_query(q, C, w, parallel_cores)
    
    else:
        return sequential_query(q, C, wedge, w)
    
    
###############################################################################
# pairwise_argmin() function returns the index of the series with the minimum #
# pairwise distance.                                                          #
###############################################################################

def pairwise_worker(chunk, C, w=None):
    min_distance = float('inf')
    min_distance_index = None
    for i in chunk:
        total_distance = 0
        for j in range(len(C)):
            if i != j:
                total_distance += metrics['dtw'](C[i], C[j], w=w)
        if total_distance < min_distance:
            min_distance = total_distance
            min_distance_index = i
    return min_distance, min_distance_index

def pairwise_argmin_parallel(C, parallel_cores, w=None):
    '''
    Computes the pairwise minimum argument in a parallelized manner across multiple time series.

    Parameters:
        C (array-like): Collection of time series.
        parallel_cores (int): The number of parallel cores to use.
        w (int or float, optional): Window parameter for the DTW algorithm.

    Returns:
        int: The index of the time series with the minimum pairwise distance.
    '''

    chunks = [range(i, len(C), parallel_cores) for i in range(parallel_cores)]

    with multiprocessing.Pool(parallel_cores) as pool:
        results = pool.starmap(pairwise_worker, zip(chunks, repeat(C), repeat(w)))

    min_distance, min_distance_index = min(results, key=lambda x: x[0])
    return min_distance_index

def sequential_pairwise_argmin(C):
    '''
    Calculates the pairwise minimum argument for a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        parallel: bool
        
    Returns:
        min_distance_index: int
    '''
    min_distance = float('inf')
    min_distance_index = None
    for i in range(len(C)):
        total_distance = 0
        for j in range(len(C)):
            if i != j:
                total_distance += metrics['dtw'](C[i], C[j], w=0, r=min_distance)
        if total_distance < min_distance:
            min_distance = total_distance
            min_distance_index = i
    return min_distance_index



def pairwise_argmin(C, parallel_cores = 1, w = None):
    '''
    Calculates the pairwise minimum argument for a collection of time series.
    
    Parameters:
        C: array-like, shape = (n_instances, length)
        parallel: bool
        
    Returns:
        min_distance_index: int
    '''
    if parallel_cores == 'all':
        parallel_cores = max(1, os.cpu_count() - 1)
        return pairwise_argmin_parallel(C, parallel_cores)
    elif parallel_cores > 1:
        if parallel_cores > os.cpu_count():
            print('Warning: Number of parallel cores exceeds number of available cores. \
                  Using all available cores.')
            parallel_cores = os.cpu_count() - 1
        return pairwise_argmin_parallel(C, parallel_cores)
    else:
        return sequential_pairwise_argmin(C)
