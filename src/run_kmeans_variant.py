import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance

from k_means_standard import k_means
from k_meanspp import k_meanspp
from e_k_means import e_k_means

"""
run_kmeans_variant.py

Pre-processes input data and runs the selected k-means algorithm variant.

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-12
"""

def run_kmeans_variant(variant, k, rng, input_data, plot=False):
    if variant == "standard":
        partitions, centroids, n_iterations = k_means(k, input_data, rng)
    elif variant == "kpp":
        partitions, centroids, n_iterations = k_meanspp(k, input_data, rng)
    elif variant == "enhanced":
        partitions, centroids, n_iterations = e_k_means(k, input_data, rng)
    else:
        print("No variant to test provided")
        return 1

    # Scoring
    global_centroid = np.sum(input_data, axis=0) / input_data.shape[0]
    
    ## Calinski-Harabasz Index
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
    chi = (bcss / (k - 1)) / (wcss / (input_data.shape[0] - k))
    
    ## TODO: Silhouette Score

    if plot:
        plot_results()

    return {
        "iterations": n_iterations,
        "ch_score": chi,
        # "silhouette": sil_score
    }


def plot_results(partitions, centroids, global_centroid):
    # Plotting
    colors = ['red', 'orange', 'yellow', 
              'green', 'blue', 'purple', 
              'pink', 'brown', 'cyan']

    # Plot 1% of the first two PCA components of each partition
    partitions_sample = [
        rng.choice(
            partition[:,0:2], 
            size=partition.shape[0]//100, 
            replace=False
        ) 
        for partition in partitions
    ]
    
    for i in range(len(partitions_sample)):
        plt.scatter(
            partitions_sample[i][:,0], 
            partitions_sample[i][:,1], 
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

   
