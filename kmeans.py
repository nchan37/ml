
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [5 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
                """
        temp = x[:,np.newaxis]
        temp2 = temp - y
        temp2 = np.square(temp2)
        dist = np.sum(temp2,axis = 2)
        dist = np.sqrt(dist)
        return dist
    
    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        d = points.shape
        centers = points[np.random.randint(0,d[0],K)]
        return centers

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        temp = self.pairwise_dist(centers,points)
        cluster_idx = np.argmin(temp,0)
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        centers = np.zeros(old_centers.shape)
        for x in range(len(old_centers)):
            check = points[cluster_idx == x,:].shape
            if  check[0] != 0 :
                centers[x,:] = points[cluster_idx == x,:].mean(axis=0)
            else:
                centers[x,:] = old_centers[x,:]
        return centers

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        loss = 0
        for x in range(len(centers)):
            temp = points[cluster_idx == x, :] - centers[x,:]
            loss = loss + np.sum(np.square(np.linalg.norm(temp, axis = 0)))
        return loss

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):  # [10 pts]
        """Plots loss values for different number of clusters in K-Means

        Args:
            image: input image of shape(H, W, 3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        """
        lossArr = np.zeros([max_K])
        for x in range(1,max_K+1):
            cluster_idx, centers, loss = self.__call__(data,x)
            lossArr[x-1] = loss
        k = np.arange(1,max_K+1)
        plt.plot(k,lossArr)
        plt.show()
        return lossArr

def intra_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster

    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    temp = KMeans().pairwise_dist(data[labels == cluster_idx],data[labels == cluster_idx])
    points,_ = temp.shape
    intra_dist_cluster = np.sum(temp, axis = 1)/(points-1)
    return intra_dist_cluster

def inter_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    n,d = data[labels == cluster_idx].shape
    inter_dist_cluster = np.zeros([n,1])
    for x in np.unique(labels):
        if x != cluster_idx:
            temp = KMeans().pairwise_dist(data[labels == cluster_idx],data[labels == x])
            temp2 = np.mean(temp, axis = 1)
            if all(inter_dist_cluster == 0):
                inter_dist_cluster = temp2
            else:
                inter_dist_cluster = np.minimum(temp2, inter_dist_cluster)
    return inter_dist_cluster
    

def silhouette_coefficient(data, labels):  # [2 pts]
    """
    Finds the silhouette coefficient of the current cluster assignment

    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        silhouette_coefficient: Silhouette coefficient of the current cluster assignment
    """
    si = 0
    for x in np.unique(labels):
        intrad = inter_cluster_dist(x, data, labels)
        interd = intra_cluster_dist(x, data, labels)
        si = si + np.sum((intrad-interd)/np.maximum(interd,intrad))
    silhouette_coefficient = si/len(data)
    return silhouette_coefficient