import numpy as np
from scipy.spatial import distance

"""
Standard K-means clustering algorithm.

Based off of the pseudocode in:

- K-means clustering algorithms: A comprehensive review, variants analysis, and 
advances in the era of big data 
(DOI: https://doi.org/https://doi.org/10.1016/j.ins.2022.11.139)

- K-means clustering
(https://en.wikipedia.org/w/index.php?title=K-means_clustering&oldid=1348629839)

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-26
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. a list of numpy arrays. Should be normalized!
#
# k: Number of partitions *k*.
#
# rng: Random number generator object
#
# OUTPUT:
# 
# clusters: Partition of the datapoints in `in_data`. i.e. a 2-d list of numpy arrays.
# centroids: Centerpoint of each partition in `clusters`.

def k_means(k, in_data, rng):
    dimensions = in_data[0].size
    centroids = rng.random((k, dimensions)) - 0.5
    converged = False
    n_iterations = 0
    cluster_assignments = np.zeros(in_data.shape[0], dtype='int64')
    converge_threshold = 1e-3

    while not converged:
        n_iterations += 1
        converged_centroids = 0
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
                
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[0])
            cluster_assignments[i] = 0
            
            for j in range(1, k):
                new_distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[j])
                if (new_distance_to_centroid < distance_to_centroid):
                    distance_to_centroid = new_distance_to_centroid
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

    return (cluster_assignments, centroids, n_iterations)
