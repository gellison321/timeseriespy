import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def interpolate(array, length):
    '''
    Interpolates an array to a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        length: int
    Returns:
        interpolated_array: np.array, shape = (n_instances, length)
    '''
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))

def reinterpolate(array, window_length):
    '''
    Repeats an array until it reaches a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        window_length: int
    Returns:
        reinterpolated_array: np.array, shape = (n_instances, window_length)
    '''
    length = len(array)
    return np.concatenate([np.tile(array, window_length//length),array[:window_length%length]])

def pad(array, length):
    '''
    Pads an array with zeros until it reaches a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        length: int
    Returns:
        padded_array: np.array, shape = (n_instances, length)
    '''
    return np.pad(array, (0,length - len(array)), 'constant')

def center_moving_average(array, period):
    '''
    Computes the centered moving average of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        period: int
    Returns:
        centered_moving_average: np.array, shape = (n_instances, length - period + 1)
    '''
    ret = np.cumsum(array)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

def argmin(args):
    '''
    Returns the index of the minimum element in the list.

    Parameters:
        args: list
    
    Returns:
        min_index: int
    '''
    min_index = 0
    for i in range(1, len(args)):
        if args[i] < args[min_index]:
            min_index = i
    return min_index

def z_normalize(I: np.array) -> np.array:
    '''
    '''
    return (I-np.mean(I))/np.std(I)

def z(i: float, mu: float, sigma: float) -> float:
    '''
    Z-normalizes a given value according to a given mean and standard deviation.

    Parameters:
        i: float
        mu: float
        sigma: float
    
    Returns:
        z: float
    '''
    return (i-mu)/sigma
            
def find_first_peak(array, thres = 0.9):
    '''
    Finds the first peak in an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        thres: float
    
    Returns:
        peak_index: int
    '''
    thres = np.quantile(array, thres)
    first = [0]
    first.append(array[1] - array[0])
    for i in range(2, len(array)-1):
        first_diff = array[i] - array[i-1]
        first.append(first_diff)
        second_diff = first_diff - first[i-1]
        if first_diff < 0 and first[i-1] > 0 and second_diff < 0 and array[i] > thres:
            return i
        
def find_peaks(array, min_dist = 60, thres = 0.9):
    '''
    Finds peaks in an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        min_dist: int
        thres: float
        
    Returns:
        peak_indexes: np.array, shape = (n_peaks, )
    '''
    return find_peaks(array, height=np.quantile(array, thres), distance=min_dist)[0]

utils = {'interpolate' : interpolate,
        'reinterpolate' : reinterpolate,
        'pad' : pad,
        'moving_average' : center_moving_average,
        'find_peaks' : find_peaks,
        'first_peak' : find_first_peak,
        'z_normalize' : z_normalize,
        'z' : z
        }