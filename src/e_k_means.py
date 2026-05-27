import numpy as np
from scipy.spatial import distance

"""
"Enhanced K-means" clustering algorithm (Fahim's Method).

Based off of the algorithm presented in:

- An efficient enhanced k-means clustering algorithm
(DOI: https://doi.org/10.1631/jzus.2006.A1626)

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-27
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. a list of numpy arrays. Should be normalized!
#
# k: Number of partitions *k*.
#
# rng: Random number generator object.
#
# OUTPUT:
# 
# cluster_assignments: Membership array modelling the partitioning of 
# the input data into clusters.
# 
# centroids: Centerpoint (mean vector) of each partition (cluster).
#
# nr_iterations: Number of iterations until convergence.

def e_k_means(k, in_data, rng):
    dimensions = in_data[0].size
    centroids = rng.random((k, dimensions)) - 0.5
    converged = False
    point_distances = [ np.inf for _ in range(in_data.shape[0]) ]
    cluster_assignments = np.zeros(in_data.shape[0], dtype='int64')
    n_iterations = 0
    converge_threshold = 1e-3

    while not converged:
        converged_centroids = 0
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
            # Enhanced K-means: skip iterations
            if (
                n_iterations > 1
                and point_distances[i] 
                >= distance.sqeuclidean(in_data[i], centroids[cluster_assignments[i]])
            ):
                point_distances[i] = distance.sqeuclidean(in_data[i], centroids[cluster_assignments[i]])
                continue
            
            point_distances[i] = distance.sqeuclidean(in_data[i], centroids[0])
            cluster_assignments[i] = 0

            for j in range(1, k):
                dist = distance.sqeuclidean(in_data[i], centroids[j]) 
                if (dist < point_distances[i]):
                    point_distances[i] = dist
                    cluster_assignments[i] = j

        # Step 2: recompute the centroids
        vector_sums = np.zeros((k, dimensions))
        assignment_counts = np.zeros(k)
        for (assignment, data_point) in zip(cluster_assignments, in_data):
            vector_sums[assignment] += data_point
            assignment_counts[assignment] += 1

        new_centroids = ( 
                np.where(vector_sums == np.zeros(dimensions), 
                         rng.random(dimensions) - 0.5, 
                         vector_sums) 
                /
                np.where(assignment_counts[:, None] == 0,
                         [1],
                         assignment_counts[:, None])
        )

        deltas = np.array(list(map(distance.euclidean, centroids, new_centroids)))
    
        if np.sum(deltas < converge_threshold) == k:
            converged = True

        centroids = new_centroids
        n_iterations += 1
    
    return (cluster_assignments, centroids, n_iterations)
