import numpy as np
from timeseriespy.utils.utils import utils
from timeseriespy.comparator.metrics import metrics
from timeseriespy.timeseries.barycenter import barycenters
from timeseriespy.timeseries.features import statistical_features, time_series_features, frequency_domain_features

class TimeSeries:
    
    ''' The TimeSeries class does preprocessing, feature extraction, 
        and shapelet extraction for 1-dimensional time series data.'''

    def __init__ (self, series):
        self.series = np.array(series)
        if self.series.ndim > 1:
            raise ValueError('The series must be one-dimensional.')
        self.shape = self.series.shape
        self.dtype = self.series.dtype
        self.size = self.series.size
        self.original = self.series
        self.features = {}
        self.candidates = []
        self.shapelets = {}

    #------------------------------------------------------------------------#
    # The following methods manipulate the series directly for preprocessing #
    #------------------------------------------------------------------------#

    def quantile_normalization(self, quantile = 0.5):
        '''
        Normalizes the series by subtracting the quantile from the series.
        
        Parameters
        ----------
        quantile : float
            The quantile to be subtracted from the series.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the normalized series.
            
        '''
        self.series = self.series - np.quantile(self.series, quantile)
        return self
    
    def z_normalization(self):
        '''
        Normalizes the series by subtracting the mean and dividing by the standard deviation.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the normalized series.
            
        '''
        self.series = utils['z_normalize'](self.series)
        return self
    
    def min_max_normalization(self):
        '''
        Normalizes the series by subtracting the minimum and dividing by the maximum.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the normalized series.
            
        '''
        self.series = (self.series - np.min(self.series)) / (np.max(self.series) - np.min(self.series))
        return self

    def smooth(self, period):
        '''
        Smooths the series by applying a moving average.

        Parameters
        ----------
        period : int
            The period of the moving average.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the smoothed series.
        '''
        self.series = utils['moving_average'](self.series, period)
        return self

    def phase_sync(self, thres = .9):
        '''
        Removes the beginning of the series until the first peak is found.

        Parameters
        ----------
        thres : float
            The threshold to be used to find the first peak.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the phase synced series.
        '''
        self.series = self.series[utils['first_peak'](self.series, thres = thres):]
        return self

    def rescale(self, factor):
        '''
        Rescales the series by interpolating it to a new length.
        
        Parameters
        ----------
        factor : float
            The factor by which to rescale the series.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the rescaled series.
        '''
        self.series = utils['interpolate'](self.series, int(len(self.series)*factor))
        return self
    
    #------------------------------------------------------------------------#
    # The following methods extract features from the series and add them to #
    # the self.features attribute                                            #
    #------------------------------------------------------------------------#

    def extract_features(self, **functions):
        '''
        Extracts time series features from the series.
        
        Parameters
        ----------
        func : 
            The features to be extracted in the form of 
                keyword = function.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted time series features.
        '''
        for function in functions:
            self.features[function] = functions[function](self.series)
        return self

    def extract_statistical_features(self, *args):
        '''
        Extracts statistical features from the series.
        
        Parameters
        ----------
        args : 
            The statistical features to be extracted.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted statistical features.
        '''
        if any([arg not in statistical_features for arg in args]):
            raise ValueError(f'Valid inputs include: {[f for f in statistical_features]}')
        
        self.extract_features(**{key: statistical_features[key] for key in args})
        return self
    
    def extract_time_series_features(self, *args):
        '''
        Extracts time series features from the series.
        
        Parameters
        ----------
        args : 
            The time series features to be extracted.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted time series features.
        '''
        if any([arg not in time_series_features for arg in args]):
            raise ValueError(f'Valid inputs include: {[f for f in time_series_features]}')
        
        self.extract_features(**{key: time_series_features[key] for key in args})
        return self
    
    def extract_frequency_domain_features(self, *args):
        '''
        Extracts frequency domain features from the series.
        
        Parameters
        ----------
        args : 
            The frequency domain features to be extracted.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted frequency domain features.
        '''
        if any([arg not in frequency_domain_features for arg in args]):
            raise ValueError(f'Valid inputs include: {[f for f in frequency_domain_features]}')
        
        self.extract_features(**{key: frequency_domain_features[key] for key in args})
        return self

    #----------------------------------------------------------#
    # The following methods extract candidates from the series #
    #----------------------------------------------------------#

    def peak_extraction(self, min_dist = 60, thres = 0.6, max_dist = 150):
        '''
        Extracts the subsequences between the peaks from the series.

        Parameters
        ----------
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        
        Returns
        -------
        self : Shapelet

        '''
        peaks = utils['find_peaks'](self.series, min_dist = min_dist, thres = thres)
        self.candidates = []
        start = 0
        for i in peaks:
            candidate = self.series[start:i]
            start = i 
            if min_dist <= len(candidate) <= max_dist:
                self.candidates.append(candidate)
        return self

    def random_extraction(self, qty, min_dist = 60, max_dist = 150):
        '''
        Extracts random subsequences from the series of a random length within a range.
        
        Parameters
        ----------
        qty : int
            The number of subsequences to be extracted.
        min_dist : int
            The minimum length of the subsequences.
        max_dist : int
            The maximum length of the subsequences.
            
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        self.candidates = []
        for _ in range(qty):
            index = np.random.randint(max_dist, len(self.series)-max_dist)
            length = np.random.randint(min_dist, max_dist) if min_dist != max_dist else max_dist
            self.candidates.append(self.series[index-length//2 : index+length//2])
        return self

    def normal_extraction(self, qty = None, min_dist = 10, max_dist = 100, thres = 0.9):
        '''
        Extracts random subsequnces from the series of a length within one standard deviation of the mean length of the peaks.
        
        Parameters
        ----------
        qty : int
            The number of subsequences to be extracted.
        min_dist : int
            The minimum distance between peaks.
        thres : float
            The threshold to be used to find the peaks.
        
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        indices = utils['find_peaks'](self.series, min_dist = min_dist, thres = thres)
        lengths = np.diff(indices)
        mean = np.mean(lengths, dtype = int)
        std = np.std(lengths, dtype = int)
        qty = len(self.series) // mean if qty == None else qty
        self.candidates = []
        for _ in range(qty):
            index = np.random.randint(max_dist, len(self.series)-max_dist)
            length = np.random.randint(min_dist, max_dist) if min_dist != max_dist else max_dist
            self.candidates.append(self.series[index-length//2 : index+length//2])
        return self

    def windowed_extraction(self, window_length = 80, step = 1):
        '''
        Extracts subsequences from the series of a fixed length with a fixed step size.
        
        Parameters
        ----------
        window_length : int
            The length of the subsequences.
        step : int
            The step size between subsequences.
            
        Returns
        -------
        self : Shapelet
            The Shapelet object with the extracted subsequences.
        '''
        self.candidates = []
        for i in np.arange(0, len(self.series) - window_length, step):
            self.candidates.append(self.series[i:i+window_length])
        return self
    
    #-------------------------------------------------------------#
    # The following methods extract shapelets from the candidates #
    #-------------------------------------------------------------#

    def random_shapelet(self):
        pass

    def exhaustive_shapelet(self):
        pass

    def barycenter_shapelet(self):
        pass