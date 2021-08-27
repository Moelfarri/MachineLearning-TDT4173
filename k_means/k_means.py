import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


import sys
def initialize_centroids_kmeans_pp(data, k):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points
        k - number of clusters
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])

    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):

        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = euclidean_distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    return centroids



class KMeans:
    
    def __init__(self, K=2,iterations=300, init="K-Means++", threshold=0.0001):
        self.K = K
        self.iterations = iterations
        self.centroids = []
        self.distances = []
        self.init      = init
        self.threshold = threshold
        
    
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = np.array(X).copy() 
        
        #Init Centroids with one of the methods.
        if self.init == "Random":
            idx = np.random.choice(len(X), self.K, replace=False)
            self.centroids = X[idx, :]
            
        elif self.init == "K-Means++":
            self.centroids = initialize_centroids_kmeans_pp(X, self.K)

        
        
        #Find the distance between centroids and all the data points       
        self.distances = np.concatenate([[euclidean_distance(X, centroid)] for centroid in self.centroids]).T
        
        #Assign datapoints to clusters based on minimum distance from Centroid
        point_cluster_assignment = np.array([np.argmin(i) for i in self.distances]) 

        #Repeating the above steps for a defined number of iterations
        
        prev_centroids_norm    = 0
        current_centroids_norm = 0 

        for i in range(self.iterations): 
            self.centroids = []
            prev_centroids_norm = current_centroids_norm
            
            for idx in range(self.K):
                #Updating Centroids by taking mean of Cluster it belongs to
                temp_centroid = X[point_cluster_assignment==idx].mean(axis=0) 
                self.centroids.append(temp_centroid)
            
            #Updated Centroids and distances
            self.centroids = np.vstack(self.centroids) 
            self.distances = np.concatenate([[euclidean_distance(X, centroid)] for centroid in self.centroids]).T
            point_cluster_assignment = np.array([np.argmin(i) for i in self.distances])
           
            
            #break algorithm if converged before for-loop iterations is over
            current_centroids_norm = np.sum(euclidean_distance(self.centroids,0))
            if self.threshold > np.abs(prev_centroids_norm-current_centroids_norm):
                break
            
             
                
                
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        point_cluster_assignment = np.array([np.argmin(i) for i in self.distances])
        return point_cluster_assignment
        
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
