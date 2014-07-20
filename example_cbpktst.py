"""Example of Cluster-Based Permutation Kernel Two-Sample Test (CBPKTST).
"""

from cbpktst import *
import pickle
from simulate_data import simulate_2d, plot_map2d

if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    verbose = True
    parallel = True
    save = True

    print("Cluster-Based Permutation Kernel Two-Sample Test (CBPKTST).")
    p_value_threshold = 0.05
    iterations = 1000
    homogeneous_statistic = 'normalized MMD2u' # 'unit_statistic' # '1-p_value' # 

    print("Creating simulated data.")
    d = 20
    k = 10
    m = 200
    n = 200
    threshold = 1.5
    XX, YY, proximity_matrix, coordinates = simulate_2d(d=d, k=k, m=m, n=n, threshold=threshold)

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

    cluster, cluster_statistic, p_value_cluster, p_value_threshold, max_cluster_statistic = cluster_based_permutation_test(unit_statistic, unit_statistic_permutation, proximity_matrix, p_value_threshold=p_value_threshold, homogeneous_statistic=homogeneous_statistic, verbose=verbose)

    # print("")
    # print("Visualising maps.")
    # vmin = unit_statistic_homogeneous_significant.min()
    # vmax = unit_statistic_homogeneous_significant.max()
    # plot_map2d(k, coordinates, unit_statistic_homogeneous_significant, vmin=vmin, vmax=vmax)
    # plot_map2d(k, coordinates, unit_statistic_homogeneous, vmin=vmin, vmax=vmax)
