import numpy as np
from scipy.sparse import coo_matrix
# from scipy.spatial.kdtree import KDTree
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from kernel_two_sample_test import MMD2u, compute_null_distribution
from joblib import Parallel, delayed
from scipy.sparse import issparse
from networkx import from_scipy_sparse_matrix, from_numpy_matrix, connected_components


def compute_boolean_proximity_matrix(coordinates, threshold):
    """Compute the boolean proximity matrix of units (with given
    coordinates) which have Euclidean distance less than threshold.
    """
    dm = pairwise_distances(coordinates)
    proximity_matrix = dm < threshold
    return proximity_matrix.astype(np.int)


def compute_sparse_boolean_proximity_matrix(coordinates, threshold):
    """Compute the boolean proximity matrix of units (with given
    coordinates) which have Euclidean distance less than
    threshold. This implementation is efficient for a large number of
    units.

    CSC format is necessary for future slicing of this matrix. This
    implementation uses a COO sparse matrix internally because KDTree
    can be queried on multiple units at once (which is very efficient)
    and the COO matrix provides the means to build a sparse matrix
    from three vectors: row, columns and value (data).

    Note: this function is similar to
    sklearn.neighbors.kneighbors_graph() but differs in the definition
    of the neighborhood which is distance-based instead of being a
    kNN.
    """
    tree = KDTree(coordinates)
    neighbors = tree.query_radius(coordinates, r=threshold) # use query_ball_tree() with SciPy's KDTree
    row = np.concatenate([i * np.ones(len(item)) for i, item in enumerate(neighbors)])
    column = np.concatenate(neighbors.tolist())
    data = np.ones(len(row), dtype=np.bool)
    proximity_matrix = coo_matrix((data, (row, column)), shape=(coordinates.shape[0], coordinates.shape[0]), dtype=np.bool)
    return proximity_matrix.tocsc()


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
    clusters = np.array([np.array(cl) for cl in clusters], dtype=np.object)[idx]
    cluster_statistic = cluster_statistic[idx]
    return clusters, cluster_statistic
