from timeseriespy.comparator.metrics import metrics
from timeseriespy.timeseries.barycenter import barycenters
import numpy as np
import multiprocessing

class KMeans():
    
    def __init__ (self, n_init = 5, k_clusters = 3, max_iter = 100, centroids = [], metric = 'dtw', averaging = 'interpolated'):
        self.k_clusters = k_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.centroids = centroids
        self.metric = metric
        self.method = averaging
        
    def _assign_clusters(self, X, centroids):
        '''
        Assigns each instance of X to the nearest centroid.

        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        return [np.argmin(np.array([metrics[self.metric](x, centroid)**2 for centroid in centroids])) for x in X]
    
    def _initialize_centroids(self, k_centroids):
        '''
        Initializes k centroids by randomly selecting k instances of X.
        
        Parameters:
            k_centroids: int, number of centroids to initialize
        Returns:
            centroids: array-like, shape = (k_centroids, length)
        '''
        centroids = [self.X[np.random.randint(0, self.X.shape[0])] for _ in range(k_centroids)]
        return np.array(centroids, dtype = self.dtype)

    def _update_centroids(self, X, centroids, clusters):
        '''
        Updates the centroids by computing the barycenter of each cluster.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            new_centroids: array-like, shape = (k_centroids, length)
        '''
        new_centroids = []
        for k in range(len(centroids)):  
            cluster = X[np.where(clusters==k)[0]]
            if cluster.shape[0] == 0:
                new_centroids.append(centroids[k])
            elif cluster.shape[0] == 1:
                new_centroids.append(cluster[0])
            else:
                new_centroids.append(barycenters[self.method](cluster))
        return np.array(new_centroids, dtype = self.dtype)

    def _check_solution(self, new_centroids, old_centroids):
        '''
        Checks if the solution has converged by checking whether the new centroids 
        are equal to the old centroids.
        
        Parameters:
            new_centroids: array-like, shape = (k_centroids, length)
        Returns:
            bool
        '''
        return all([np.array_equal(old_centroids[i], new_centroids[i]) for i in range(len(old_centroids))])

    def local_kmeans(self, i = None):
        '''
        Solves the local cluster problem according to Lloyd's algorithm.
        '''
        clusters = []
        centroids = []
        if len(centroids) < self.k_clusters:
            centroids = self._initialize_centroids(self.k_clusters)
        for _ in range(self.max_iter):
            clusters = self._assign_clusters(self.X, centroids)
            new_centroids = self._update_centroids(self.X, centroids, clusters)
            if self._check_solution(new_centroids, centroids):
                break
            else:
                centroids = new_centroids
        
        inertia = sum([metrics[self.metric](self.X[i], centroids[clusters[i]])**2 for i in range(len(self.X))])

        return clusters, centroids, inertia

    def sample_kmeans(self):
        '''
        Solves the global cluster problem by sampling the local cluster problem n_init times.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        '''
        best_cost = None
        best_clusters = None
        best_centroids = None

        for _ in range(self.n_init):
            clusters, centroids, inertia = self.local_kmeans()
            if best_cost is None or inertia < best_cost:
                best_cost = inertia
                best_clusters = clusters
                best_centroids = centroids

        self.inertia = best_cost
        self.clusters = best_clusters
        self.centroids = best_centroids

    def sample_kmeans_parallel(self):

        num_cpus = multiprocessing.cpu_count()
        pool_size = max(1, num_cpus - 1) 

        with multiprocessing.Pool(pool_size) as pool:
            results = pool.map(self.local_kmeans, range(self.n_init))

        best_cost = None
        best_clusters = None
        best_centroids = None

        for clusters, centroids, inertia in results:
            if best_cost is None or inertia < best_cost:
                best_cost = inertia
                best_clusters = clusters
                best_centroids = centroids

        self.inertia = best_cost
        self.clusters = best_clusters
        self.centroids = best_centroids

    def fit(self, X, parallel = False):
        '''
        Checks the type of data for varied length arrays, and calls the sample_kmeans method.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        self.dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        self.X = np.array(X, dtype = self.dtype)

        if parallel:
            self.sample_kmeans_parallel()
        else:
            self.sample_kmeans()

    def predict(self, X):
        '''
        Assigns each instance of X to the nearest centroid.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        X = np.array(X, dtype = dtype)
        return self._assign_clusters(X)

    def soft_cluster(self):
        '''
        Computes the distance of each instance of X to each centroid.

        Parameters:
            None
        Returns:
            soft_clusters: array-like, shape = (n_instances, k_centroids)
        '''
        soft_clusters = []
        for centroid in self.centroids:
            distances = []
            for i in range(len(self.X)):
                distances.append(metrics[self.metric](self.X[i], centroid))
            soft_clusters.append(distances)
        a = np.array(soft_clusters)
        a = a.reshape(a.shape[1], a.shape[0]);
        return a