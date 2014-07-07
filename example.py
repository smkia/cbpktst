from cbpktst import *
import pickle

if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    verbose = True
    parallel = True
    save = True

    print("Creating simulated data.")
    d = 20
    n_units = 100
    m = n = 200 # per sample
    sigma2 = 0.2
    XX = np.random.multivariate_normal(np.zeros(d), sigma2 * np.eye(d), size=(m, n_units))
    YY = np.random.multivariate_normal(np.ones(d), sigma2 * np.eye(d), size=(n, n_units))
    coordinates = np.random.rand(n_units, 3)
    threshold = 0.3
    # proximity_matrix = compute_boolean_proximity_matrix(coordinates, threshold=threshold)
    proximity_matrix = compute_sparse_boolean_proximity_matrix(coordinates, threshold=threshold)
    print("%f of the pairs of units are proximal." % ((len(proximity_matrix.nonzero()[0]) - n_units) / 2.0  / (n_units * (n_units - 1.0) / 2.0)))

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
    
