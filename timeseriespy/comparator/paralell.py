import numpy as np
from timeseriespy.utils.utils import *
from timeseriespy.comparator.metrics import *
from timeseriespy.comparator.bounding import *
import multiprocessing

import numpy as np
from timeseriespy.utils.utils import *
from timeseriespy.comparator.metrics import *
from timeseriespy.comparator.bounding import *
import itertools
import multiprocessing

def worker(args):
    q, series, wedge, best_so_far, w = args
    if LB_Keogh_DTW(q, wedge, r=best_so_far) > best_so_far:
        return np.inf
    dist = metrics['dtw'](q, series, w=w, r=best_so_far)
    return dist

def query_parallel(q, C, w=0, set_length=False):
    if set_length != False:
        if isinstance(set_length, (int, float)):
            C = np.array([interpolate(c, set_length) for c in C])
        else:
            raise TypeError('set_length must be an int or float')
    elif any(np.diff(list(map(len, C))) != 0):
        raise ValueError('All series must be of the same length.')

    w = set_window(q, C[0], w=w)
    best_so_far = np.inf
    best_index = None
    wedge = DTW_wedge(C, w=w)

    pool = multiprocessing.Pool()  # Using default number of processes
    args = [(q, C[i], wedge, best_so_far, w) for i in range(len(C))]
    results = pool.map(worker, args)
    pool.close()
    pool.join()

    for i, dist in enumerate(results):
        if dist < best_so_far:
            best_so_far = dist
            best_index = i

    return best_index
