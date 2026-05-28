import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from k_means_standard import k_means
from k_meanspp import k_meanspp
from e_k_means import e_k_means
from k_star_means import k_star_means

"""
run_kmeans_variant.py

Run a selected k-means algorithm variant and gather cluster validity metrics.

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-28
"""

def run_kmeans_variant(variant, k, rng, input_data, plot=False, plot_denom=100):
    if variant == "standard":
        assignments, centroids, n_iterations = k_means(k, input_data, rng)
    elif variant == "kpp":
        assignments, centroids, n_iterations = k_meanspp(k, input_data, rng)
    elif variant == "enhanced":
        assignments, centroids, n_iterations = e_k_means(k, input_data, rng)
    elif variant == "star":
        assignments, centroids, n_iterations = k_star_means(k, input_data, rng)
    else:
        print("No variant to test provided")
        return 1
    
    if plot:
        plot_results(input_data, assignments, centroids, 
                     k, variant, plot_denom, rng)

    return assignments, centroids, n_iterations

def score(input_data, assignments, n_iterations):
    # Scoring
    chi = calinski_harabasz_score(input_data, assignments)
    sil = silhouette_score(input_data, assignments)
    
    return {
        "iterations": n_iterations,
        "ch_score": chi,
        "sil_score": sil
    }

def plot_results(input_data, assignments, centroids, k, variant, plot_denom, rng):
    colors = np.array(['red', 'orange', 'yellow', 
                       'green', 'blue', 'purple', 
                       'pink', 'brown', 'cyan'])

    sample_vector = rng.choice(range(input_data.shape[0]), 
                               size=input_data.shape[0]//plot_denom,
                               replace=False)

    plt.scatter(input_data[sample_vector,0], input_data[sample_vector,1], 
                color=colors[assignments[sample_vector]], marker=".", s=2.0)
    
    plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='X')
    
    if k:
        plt.savefig(f"kmeans-{variant}-k-{k}-clustering.png", dpi=300)
    else:
        plt.savefig(f"kmeans-{variant}-clustering.png", dpi=300)

