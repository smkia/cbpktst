"""Example of Cluster-Based Permutation Kernel Two-Sample Test (CBPKTST).
"""

import numpy as np
from cbpktst import *
import pickle
from simulate_data import simulate_2d, plot_map2d, simulate_2d_distance

if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    verbose = True
    parallel = True
    save = True
    precompute_permutations = True

    print("Cluster-Based Permutation Kernel Two-Sample Test (CBPKTST).")
    p_value_threshold = 0.05
    iterations = 100
    homogeneous_statistic = 'normalized MMD2u' # '1-p_value' # 'unit_statistic' # 

    print("Creating simulated data.")
    d = 20
    k = 10
    m = 200
    n = 200
    threshold = np.sqrt(2)
    XX, YY, proximity_matrix, coordinates = simulate_2d_distance(d=d, k=k, m=m, n=n, threshold=threshold, seed=seed)

    print("Retrieving or creating Kernel matrices and samples from null distributions.")
    filename = "data/kernels_mmd2u_null_distributions_k%d_it%d.pickle" % (k, iterations)
    try:
        print("Loading %s" % filename)
        data = pickle.load(open(filename))
        Ks = data['Ks']
        sigma2s = data['sigma2s']
        unit_statistic = data['unit_statistic']
        unit_statistic_permutation = data['unit_statistic_permutation']
    except IOError:
        print("File not found!")
        print("Pre-computing the kernel matrix for each unit.")
        Ks, sigma2s = precompute_gaussian_kernels(XX, YY, verbose=verbose)
        print("Computing MMD2us and null-distributions.")
        if precompute_permutations:
            print("Precomputing all permutations to enforce the exact SAME permutations across units when doing parallel computations.")
            np.random.seed(seed) # setting the same seed for each subject
            permutation = np.vstack([np.random.permutation(m+n) for i in range(iterations)])
        else:
            permutation = None

        unit_statistic, unit_statistic_permutation = compute_mmd2u_and_null_distributions(Ks, m, n, iterations=iterations, seed=seed)
        if save:
            print("Saving kernels, sigmas, MMD2us and null-distributions in %s" % filename)
            pickle.dump({'Ks': Ks,
                         'sigma2s': sigma2s,
                         'unit_statistic': unit_statistic,
                         'unit_statistic_permutation': unit_statistic_permutation,
                         },
                        open(filename, 'w'),
                        protocol = pickle.HIGHEST_PROTOCOL
                        )


    print("")
    print("Cluster-based permutation test.")

    cluster, cluster_statistic, p_value_cluster, p_value_threshold, max_cluster_statistic, unit_statistic_homogeneous = cluster_based_permutation_test(unit_statistic, unit_statistic_permutation, proximity_matrix, p_value_threshold=p_value_threshold, homogeneous_statistic=homogeneous_statistic, verbose=verbose)

    print("")
    print("Visualising maps.")
    print("Zeroing all unit statistic (homogeneous too) related non-significant clusters.")
    unit_statistic_homogeneous_significant = np.zeros(unit_statistic_homogeneous.size)
    for cs in cluster[p_value_cluster <= p_value_threshold]:
        unit_statistic_homogeneous_significant[cs] = unit_statistic_homogeneous[cs]

    vmin = unit_statistic_homogeneous_significant.min()
    vmax = unit_statistic_homogeneous_significant.max()
    plot_map2d(k, coordinates, unit_statistic_homogeneous_significant, vmin=vmin, vmax=vmax)
    plot_map2d(k, coordinates, unit_statistic_homogeneous, vmin=vmin, vmax=vmax)
    print("Detected %d units." % len(unit_statistic_homogeneous_significant.nonzero()[0]))
