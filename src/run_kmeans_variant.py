import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import silhouette_score

from k_means_standard import k_means
from k_meanspp import k_meanspp
from e_k_means import e_k_means

"""
run_kmeans_variant.py

Run a selected k-means algorithm variant and gather cluster validity metrics.

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-17
"""

def run_kmeans_variant(variant, k, rng, input_data, plot=False, plot_denom=100):
    if variant == "standard":
        partitions, centroids, n_iterations = k_means(k, input_data, rng)
    elif variant == "kpp":
        partitions, centroids, n_iterations = k_meanspp(k, input_data, rng)
    elif variant == "enhanced":
        partitions, centroids, n_iterations = e_k_means(k, input_data, rng)
    else:
        print("No variant to test provided")
        return 1
    
    if plot:
        plot_results(partitions, centroids, 
                     global_centroid, variant, k, plot_denom)

    return partitions, centroids, n_iterations

## Calinski-Harabasz Index
def ch_score(k, n, partitions, centroids, global_centroid):
    bcss = 0.0
    wcss = 0.0
    for i in range(len(partitions)):
        bcss += (
            partitions[i].shape[0] 
            * distance.sqeuclidean(centroids[i], global_centroid)
        )
    for i in range(len(partitions)):
        for j in range(partitions[i].shape[0]):
            wcss += (
                distance.sqeuclidean(partitions[i][j], centroids[i])
            )
    chi = (bcss / (k - 1)) / (wcss / (n - k))
    return chi

# Silhouette Score
# Just wrapping scikitlearn's implementation, so implementation details are
# uncertain.
def sil_score(partitions):
    # Switching to the format SciPy uses
    # Below function is AI-generated
    def flatten_partitions(partitions):
        partitions_vstack = np.vstack(partitions)
        labels = np.concatenate([
            np.full(len(cluster), i)
            for i, cluster in enumerate(partitions)
        ])
        return partitions_vstack, labels

    partitions_vstack, labels = flatten_partitions(partitions)
    return silhouette_score(partitions_vstack, labels)

# Silhouette Score (full dataset)
# This implementation is unfortunately too slow.
# I'll have to use what Copilot suggested.
#def sil_score(partitions, centroids):
#    score_sum = 0.0
#    points = 0
#    
#    for i in range(len(partitions)):
#        for j in range(partitions[i].shape[0]):
#            other_points = np.delete(partitions[i], j, axis=0)
#
#            a = np.mean(distance.cdist(other_points, 
#                                       np.array([partitions[i][j]]), 
#                                       metric='sqeuclidean'))
#
#            temp_distance = np.finfo(np.float64).max
#            next_closest_index = -1
#            
#            for centroid in np.delete(centroids, i, axis=0):
#                if (distance.sqeuclidean(centroid, partitions[i][j]) 
#                    < temp_distance):
#                    temp_distane = distance.sqeuclidean(centroid, 
#                                                        partitions[i][j])
#                    next_closest_index += 1
#
#            b = np.mean(distance.cdist(partitions[next_closest_index],
#                                       np.array([partitions[i][j]]),
#                                       metric='sqeuclidean'))
#
#            points += 1
#            score_sum += (b - a) / np.max([b, a])
#
#    return score_sum / points

def score(k, input_data, partitions, centroids, n_iterations):
    # Scoring
    global_centroid = np.sum(input_data, axis=0) / input_data.shape[0]
    chi = ch_score(k, input_data.shape[0], 
                   partitions, centroids, global_centroid)
    sil = sil_score(partitions)
    
    return {
        "iterations": n_iterations,
        "ch_score": chi,
        "sil_score": sil
    }

def plot_results(parts, centroids, global_centroid, variant, k, plot_denom):
    # Plotting
    colors = ['red', 'orange', 'yellow', 
              'green', 'blue', 'purple', 
              'pink', 'brown', 'cyan']

    # Plot a fraction of the first two PCA components of each part
    rng = np.random.default_rng()
    parts_sample = [
        rng.choice(
            part[:,0:2], 
            size=part.shape[0]//plot_denom, 
            replace=False
        ) 
        for part in parts
    ]
    
    for i in range(len(parts_sample)):
        plt.scatter(
            parts_sample[i][:,0], 
            parts_sample[i][:,1], 
            color=colors[i],
            marker=".",
            s=2.0
        )
    
    for i in range(centroids.shape[0]):
        plt.scatter(centroids[i][0], centroids[i][1], color='black', marker='X')
    
    plt.scatter(global_centroid[0], global_centroid[1], color='black', marker='*')
    
    if k:
        plt.savefig(f"kmeans-{variant}-k-{k}-clustering.png", dpi=300)
    else:
        plt.savefig(f"kmeans-{variant}-clustering.png", dpi=300)

