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

    print("Creating simulated data.")
    k = 10
    XX, YY, proximity_matrix, coordinates = simulate_2d(k=k, threshold=1.5)

    print("Retrieving or creating Kernel matrices and samples from null distributions.")
    filename = "data/kernels_mmd2u_null_distributions.pickle"
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

    homogeneous_statistic = 'normalized MMD2u' # '1-p_value' # 'unit_statistic_permutation'
    print("Homogeneous statistic: %s" % homogeneous_statistic)

    print("Computing MMD2u thresholds for each unit with p-value=%f" % p_value_threshold)
    mmd2us_threshold = np.sort(unit_statistic_permutation, axis=1)[:, np.int((1.0-p_value_threshold)*iterations)]

    print("Computing actual p-values of unpermuted data")
    p_value = (unit_statistic_permutation.T >= unit_statistic).sum(0).astype(np.float) / iterations
    unit_significant = p_value <= p_value_threshold
    print("Computing the p-value of each permutation of each unit.")
    p_value_permutation = (iterations - np.argsort(np.argsort(unit_statistic_permutation, axis=1), axis=1)).astype(np.float) / iterations # argsort(argsor(x)) given the rankings of x in the same order. Example: a=[40,30,50,10,20] , then argsort(argsort(a)) gives array([3, 2, 4, 0, 1])
    unit_significant_permutation = p_value_permutation <= p_value_threshold

    if homogeneous_statistic == '1-p_value':
        unit_statistic_permutation_homogeneous = 1.0 - p_value_permutation
        unit_statistic_homogeneous = 1.0 - p_value
    elif homogeneous_statistic == 'normalized MMD2u':
        mmd2us_mean = unit_statistic_permutation.mean(1)
        mmd2us_std = unit_statistic_permutation.std(1)
        # delta = mmd2us_threshold - mmd2us_mean
        delta = mmd2us_std
        unit_statistic_permutation_homogeneous = (unit_statistic_permutation - mmd2us_mean[:,None]) / delta[:,None]
        unit_statistic_homogeneous = (unit_statistic - mmd2us_mean) / delta
    elif homogeneous_statistic == 'unit_statistic_permutation':
        unit_statistic_permutation_homogeneous = unit_statistic_permutation
        unit_statistic_homogeneous = unit_statistic
    else:
        raise Exception

    print("For each permutation compute the max cluster statistic.")
    max_cluster_statistic = np.zeros(iterations)
    for i in range(iterations):
        max_cluster_statistic[i] = 0.0
        if unit_significant_permutation[:,i].sum() > 0:
            idx = np.where(unit_significant_permutation[:,i])[0] # BEWARE! If you don't use where() and stick with boolean indices, then the next slicing fails, unexpectedly...(!). Is this a SciPy bug?
            pm_permutation = proximity_matrix[idx][:,idx]
            cluster_permutation, cluster_statistic_permutation = compute_clusters_statistic(unit_statistic_permutation_homogeneous[idx,i], pm_permutation, verbose=True)
            max_cluster_statistic[i] = cluster_statistic_permutation.max()

    print("Computing the null-distribution of the max cluster statistic.")
    max_cluster_statistic_threshold = np.sort(max_cluster_statistic)[int((1.0-p_value_threshold) * iterations)]
    print("Max cluster statistic threshold (p-value=%s) = %s" % (p_value_threshold, max_cluster_statistic_threshold))

    print("")
    print("Computing significant clusters on unpermuted data.")
    idx = np.where(unit_significant)[0] # BEWARE! If you don't use where() and stick with boolean indices, then the next slicing fails, unexpectedly...(!). Is this a SciPy bug?
    cluster_significant = []
    if len(idx) > 0:
        pm = proximity_matrix[idx][:,idx]
        cluster, cluster_statistic = compute_clusters_statistic(unit_statistic_homogeneous[idx], pm, verbose=True)
        print("Cluster statistic: %s" % cluster_statistic)
        p_value_cluster = (max_cluster_statistic[:,None] > cluster_statistic).sum(0).astype(np.float) / iterations
        print "p_value_cluster:", p_value_cluster
        for i, pvc in enumerate(p_value_cluster):
            if pvc <= p_value_threshold:
                cluster_significant.append(np.where(unit_significant)[0][cluster[i]])

    else:
        print("No significant clusters in unpermuted data!")

    print("Zeroing all unit statistic (homogeneous too) related non-significant clusters.")
    unit_statistic_significant = np.zeros(unit_statistic.size)
    unit_statistic_homogeneous_significant = np.zeros(unit_statistic.size)
    for cs in cluster_significant:
        unit_statistic_significant[cs] = unit_statistic[cs]
        unit_statistic_homogeneous_significant[cs] = unit_statistic_homogeneous[cs]

    vmin = unit_statistic_homogeneous_significant.min()
    vmax = unit_statistic_homogeneous_significant.max()
    plot_map2d(k, coordinates, unit_statistic_homogeneous_significant, vmin=vmin, vmax=vmax)
    plot_map2d(k, coordinates, unit_statistic_homogeneous, vmin=vmin, vmax=vmax)
