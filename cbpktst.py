import numpy as np
from sklearn.metrics import pairwise_distances
from kernel_two_sample_test import MMD2u, compute_null_distribution
from joblib import Parallel, delayed
from scipy.sparse import issparse
from networkx import from_scipy_sparse_matrix, from_numpy_matrix, connected_components


def precompute_gaussian_kernels(XX, YY, verbose=False):
    """For each unit, precompute Gaussian kernel between the trials of
    the two samples XX and YY. Estimate each sigma2 parameter as median
    distance between the trials of each sample.
    """
    print("Pre-computing the kernel matrix for each unit.")
    n_units = XX.shape[1] # or YY.shape[1]
    Ks = [] # here we store all the kernel matrices
    sigma2s = np.zeros(n_units) # here we store all the sigma2s, one per unit
    m = XX.shape[0]
    n = YY.shape[0]
    for i in range(n_units):
        print("Unit %s" % i)
        X = XX[:,i,:].copy()
        Y = YY[:,i,:].copy()
        if verbose: print("%s  %s" % (X.shape, Y.shape))
        if verbose: print("Computing Gaussian kernel.")
        dm = pairwise_distances(np.vstack([X, Y]), metric='sqeuclidean')
        # Heuristic: sigma2 is the median value among all pairwise
        # distances between X and Y. Note: should we use just
        # dm[:m,m:] or all dm?
        sigma2 = np.median(dm[:m,m:]) 
        sigma2s[i] = sigma2
        if verbose: print("sigma2 = %s" % sigma2)
        K = np.exp(-dm / sigma2)
        Ks.append(K)

    return Ks, sigma2s


def compute_mmd2u_and_null_distributions(Ks, m, n, iterations=1000, seed=0, parallel=True, n_jobs=-1, verbose=False):
    """Compute MMD2u statistic and its null-distribution for each unit
    from kernel matrices Ks. Each null-distributions is approximated
    with the given number of iterations. Parallel (multiprocess, with
    n_jobs processes) computation is available. Note: n_jobs=-1 means
    'use all available cores.'
    """
    n_units = len(Ks)
    unit_statistic = np.zeros(n_units)
    unit_statistic_permutation = np.zeros((n_units, iterations))

    print("Computing MMD2u for each unit.")
    for i, K in enumerate(Ks):
        mmd2u = MMD2u(K, m, n)
        unit_statistic[i] = mmd2u

    print("Computing MMD2u's null-distribution, for each unit.")
    if not parallel:
        for i, K in enumerate(Ks):
            mmd2u_null = compute_null_distribution(K, m, n, iterations=iterations, verbose=verbose, seed=seed, marker_interval=100) # NOTE: IT IS FUNDAMENTAL THAT THE SAME IS USED SEED FOR EACH UNIT!
            unit_statistic_permutation[i, :] = mmd2u_null
    else:
        print("Parallel computation!")
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(compute_null_distribution)(K, m, n, iterations=iterations, verbose=False, seed=seed) for K in Ks) # NOTE: IT IS FUNDAMENTAL THAT THE SAME SEED IS USED FOR EACH UNIT!
        unit_statistic_permutation = np.vstack(results)

    return unit_statistic, unit_statistic_permutation

    
def compute_clusters_statistic(test_statistic, proximity_matrix, verbose=False):
    """Given a test statistic for each unit and a boolean proximity
    matrix among units, compute the cluster statistic using the
    connected components graph algorithm. It works for sparse
    proximity matrices as well.

    Returns the clusters and their associated cluster statistic.
    """
    # Build a graph from the proximity matrix:
    if issparse(proximity_matrix):
        graph = from_scipy_sparse_matrix(proximity_matrix)
    else:
        graph = from_numpy_matrix(proximity_matrix)

    # Compute connected components:
    clusters = connected_components(graph)
    if verbose: print("Nr. of clusters: %s. Clusters sizes: %s" % (len(clusters), np.array([len(cl) for cl in clusters])))
    # Compute the cluster statistic:
    cluster_statistic = np.zeros(len(clusters))
    for i, cluster in enumerate(clusters):
        cluster_statistic[i] = test_statistic[cluster].sum()

    # final cleanup to prepare easy-to-use results:
    idx = np.argsort(cluster_statistic)[::-1]
    clusters = np.array([np.array(cl, dtype=np.int) for cl in clusters], dtype=np.object)[idx]
    if clusters[0].dtype == np.object: # THIS FIXES A NUMPY BUG (OR FEATURE?)
        # The bug: it seems not possible to create ndarray of type
        # np.object from arrays all of the *same* lenght and desired
        # dtype, i.e. dtype!=np.object. In this case the desired dtype
        # is automatically changed into np.object. Example:
        # array([array([1], dtype=int)], dtype=object)
        clusters = clusters.astype(np.int)

    cluster_statistic = cluster_statistic[idx]
    return clusters, cluster_statistic
