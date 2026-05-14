import numpy as np
import argparse
import time
import preprocessing
from run_kmeans_variant import run_kmeans_variant

"""
main.py

Main module for testing of k-means variants for Eynar Eklöf's and
Hannes Hultin's bachelor's thesis project.

eaeklof@kth.se
2026-05-14
"""

# NOTE: Mostly Copilot generated
def benchmark(variant, k, n_runs, data_file='kidney_sc_dataset.csv'):
    input_data = preprocessing.prepare(data_file)
    rng = np.random.default_rng(seed=1234)
    results = []

    if k:
        print(f"\nRunning experiment on variant \"{variant}\"" 
              f" with k={k} and n={n_runs}\n")
    else:
        print(f"\nRunning experiment on variant \"{variant}\""
              f" with n={n_runs}\n")

    for i in range(n_runs):
        print(f"Run {i+1}:", end=' ')
        
        start = time.perf_counter()
        res = run_kmeans_variant(variant, k, rng, input_data)
        elapsed = time.perf_counter() - start
        
        res["time"] = elapsed
        results.append(res)

        print("Converged!")
        
    return results

# NOTE: Copilot generated
def summarize(results):
    return {
        "iterations_mean": np.mean([r["iterations"] for r in results]),
        "time_mean": np.mean([r["time"] for r in results]),
        "ch_mean": np.mean([r["ch_score"] for r in results]),
        "silhouette_mean": np.mean([r["sil_score"] for r in results]),

        # Optional: variability
        # "time_std": np.std([r["time"] for r in results]),
    }

# NOTE: Mostly Copilot generated
def main(k, n_runs, data_file, variant):
    variants = ["standard", "kpp", "enhanced"]
    
    if variant in variants:
        results = benchmark(variant, k, n_runs, data_file)
        summary = summarize(results)
        print(f"\nVariant: {variant}")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        for variant in variants:
            results = benchmark(variant, k, n_runs, data_file)
            summary = summarize(results)

            print(f"\nVariant: {variant}")
            for key, value in summary.items():
                print(f"{key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run benchmark on all k-means variants")
    parser.add_argument("filename", nargs="?", default="kidney_sc_dataset.csv",
                        help="Input data CSV file")
    parser.add_argument("-k", help="Number of clusters (if required for variant)")
    parser.add_argument("-v", "--variant", default=None,
                        help="Variant to run benchmark on. Leave blank to test all.")
    parser.add_argument("-n", "--nr-iterations", default=20,
                        help="Number of experiment iterations to use for mean value")
    args = parser.parse_args()
    main(int(args.k), int(args.nr_iterations), args.filename, args.variant)


