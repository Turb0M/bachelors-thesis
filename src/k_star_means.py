import numpy as np
from scipy.spatial import distance

"""
K*-means clustering algorithm.

Based off of the algorithm described in:

k*-Means: A new generalized k-means clustering algorithm
(DOI: 10.1016/S0167-8655(03)00146-6)

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-28
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
# cluster_assignments: Membership array modeling the partitioning of
# input data into clusters.
#
# centroids: Centerpoint (mean vector) of each partition (cluster).
#
# n_iterations: Number of iterations until convergence.

eta = 2e-2
eta_s = 1e-3
conv_thresh = 1e-3

def weight(assignment_counts, index, total):
    return (
        1 if assignment_counts[index] == 0
        else assignment_counts[index] / total 
    )

def cov_upd_rule(cov, eta_s, point, centroid):
    z = point - centroid
    frac1 = cov / (1 - eta_s)
    frac2_nmrtr = eta_s * np.linalg.outer(z, z) @ cov
    frac2_denom = 1 - eta_s + eta_s * z @ cov @ z
    frac2 = frac2_nmrtr / frac2_denom
    return frac1 @ (np.identity(centroid.size) - frac2)

def k_star_init(k, in_data, rng):
    dimensions = in_data[0].size
    centroids = rng.random((k, dimensions)) - 0.5
    
    cluster_assignments = np.zeros(in_data.shape[0], dtype='int64')
    assignment_counts = np.zeros(k, dtype='int64')
    assignments_total = 0

    n_iterations = 0
    converged = False
    
    while not converged:
        converged = True
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
                
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = (
                distance.sqeuclidean(in_data[i], centroids[0])
                * weight(assignment_counts, 0, assignments_total)
            )
            cluster_assignment = 0
            
            for j in range(1, k):
                new_distance_to_centroid = (
                    distance.sqeuclidean(in_data[i], centroids[j])
                    * weight(assignment_counts, j, assignments_total)
                )
                if (new_distance_to_centroid < distance_to_centroid):
                    distance_to_centroid = new_distance_to_centroid
                    cluster_assignment = j

            if cluster_assignments[i] != cluster_assignment:
                converged = False
            cluster_assignments[i] = cluster_assignment
            
            assignment_counts[cluster_assignments[i]] += 1
            assignments_total += 1
 
        # Step 2: recompute the centroids
        vector_sums = np.zeros((k, dimensions))
        cluster_sizes = np.zeros(k, dtype='int64')
        for (assignment, data_point) in zip(cluster_assignments, in_data):
            vector_sums[assignment] += data_point
            cluster_sizes[assignment] += 1

        centroids = (
                np.where(vector_sums == np.zeros(dimensions), 
                         rng.random(dimensions) - 0.5, 
                         vector_sums) 
                / 
                np.where(cluster_sizes[:, None] == 0,
                         [1],
                         cluster_sizes[:, None])
        )

        n_iterations += 1

    return (cluster_assignments, centroids, n_iterations)

def k_star_means(k, in_data, rng):
    assignments, centroids, n_iterations = k_star_init(k, in_data, rng)
    converged = False

    weights = np.full(k, 1/k)
    w_learn_vec = np.zeros(k)
    clusters = [ in_data[assignments == i] for i in range(k) ]
   
    # Handle empty clusters
    cov_mtrx_arr = [ np.cov(cluster, rowvar=False) 
                     if cluster.shape[0] > 1 
                     else np.identity(in_data.shape[1])
                     for cluster in clusters ]

    # Regularize covariance
    cov_mtrx_arr = [ np.linalg.inv(matrix + 1e-6 * np.identity(matrix.shape[0]))
                     for matrix in cov_mtrx_arr ]

    while not converged:
        converged = True
        dots_changed = 0
        for i in range(in_data.shape[0]):
            
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = (
                (in_data[i] - centroids[0]) 
                @ cov_mtrx_arr[0] 
                @ (in_data[i] - centroids[0])
                - np.linalg.slogdet(cov_mtrx_arr[0])[1] # See API for slogdet
                - 2 * np.log(weights[0])
            )
            cluster_assignment = 0

            for j in range(1, k):
                new_distance_to_centroid = (
                    (in_data[i] - centroids[j]) 
                    @ cov_mtrx_arr[j] 
                    @ (in_data[i] - centroids[j])
                    - np.linalg.slogdet(cov_mtrx_arr[j])[1] # See API for slogdet
                    - 2 * np.log(weights[j])
                )
                if distance_to_centroid > new_distance_to_centroid:
                    cluster_assignment = j
                    distance_to_centroid = new_distance_to_centroid
                
            if assignments[i] != cluster_assignment:
                converged = False
                dots_changed += 1
            assignments[i] = cluster_assignment            
        
            centroid_update = (
                eta 
                * cov_mtrx_arr[cluster_assignment] 
                @ (in_data[i] - centroids[cluster_assignment])
            )

            cov_mtrx_arr[cluster_assignment] = (
                cov_upd_rule(cov_mtrx_arr[cluster_assignment], 
                             eta_s, in_data[i], centroids[cluster_assignment])
            )

            centroids[cluster_assignment] += centroid_update
            
            w_learn_vec[cluster_assignment] += eta * (1 - weights[cluster_assignment])

        weights = np.exp(w_learn_vec) / np.sum(np.exp(w_learn_vec))
        n_iterations += 1
    return (assignments, centroids, n_iterations)
