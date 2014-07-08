"""Mass-univariate cluster-based permutation test (CBPT). This should
be equivalent to what FieldTrip provides. This implementation follows
the description in Groppe et al. (Psychophysiology, 2011) p.1718.
"""

import numpy as np
from simulate_data import simulate_2d
from proximity import compute_sparse_boolean_proximity_matrix_space_time
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from cbpktst import compute_clusters_statistic
from sys import stdout


def compute_ttest_clusters(XX, y, p_value_threshold, proximity_matrix_space_time):
    """See Groppe et al. (Psychophysiology 2011), p.1718.
    """
    # print "1) Compute t scores for every timepoint and sensor of interest."
    t, p = ttest_ind(XX[y==-1], XX[y==1], axis=0)

    # print "2) threshold p_values."
    # mask = p <= p_value_threshold
    t_flat = t.T.flatten() # ABS??
    idx = np.where(p.T.flatten() <= p_value_threshold)[0]

    # print "3) remove all mask entries with less than e.g. two adjacent below-threshold value. This step is OPTIONAL."
    # print "WE SKIP THIS STEP."

    # print "4) cluster below threshold values in time and space."
    clusters, cluster_statistic = compute_clusters_statistic(t_flat[idx], proximity_matrix_space_time[idx[:, None], idx])
    # give original indices to elements in clusters:
    clusters = [[idx[ci] for ci in cl] for cl in clusters]
    return clusters, cluster_statistic


if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    verbose = True
    parallel = True
    save = True

    print("Mass-univariate cluster-based permutation test (CBPT)")
    p_value_threshold = 0.05
    iterations = 1000
    n_jobs = -1

    print("Creating simulated data.")
    d = 20
    k = 10
    m = 200
    n = 200
    threshold_space = 1.5
    threshold_timesteps = 1
    XX, YY, proximity_matrix, coordinates = simulate_2d(d=d, k=k, m=m, n=n, threshold=threshold_space)
    # We prefer to represent the data by stacking together XX and YY
    # by creating a label (yy). This representation helps better
    # coding later on.
    yy = np.concatenate([-np.ones(XX.shape[0]), np.ones(YY.shape[0])])
    XX = np.vstack([XX,YY])

    print "Computing the proximity matrix in space and time."
    proximity_matrix_space_time = compute_sparse_boolean_proximity_matrix_space_time(coordinates, n_timesteps=XX.shape[2], threshold_space=threshold_space, threshold_timesteps=threshold_timesteps)

    print "Computing clusters on unpermuted data."
    clusters, cluster_statistic = compute_ttest_clusters(XX, yy, p_value_threshold, proximity_matrix_space_time)
    print "Clusters found (p_value_threshold=%s) : %s" % (p_value_threshold, len(clusters))
    print "Max cluster_statistic = %s" % cluster_statistic.max()
    print "Min cluster_statistic = %s" % cluster_statistic.min()

    print "Computing %s permutations." % iterations

    if not parallel:
        max_cluster_statistic = np.zeros(iterations)
        for i in range(iterations):
            print i,
            stdout.flush()
            y = np.random.permutation(yy)
            clusters_i, cluster_statistic_i = compute_ttest_clusters(XX, y, p_value_threshold, proximity_matrix_space_time)
            max_cluster_statistic[i] = np.abs(cluster_statistic_i).max()
    else:
        print("Parallel computation!")
        def myfunc(XX, yy, p_value_threshold, proximity_matrix_space_time):
            y = np.random.permutation(yy)
            clusters_i, cluster_statistic_i = compute_ttest_clusters(XX, y, p_value_threshold, proximity_matrix_space_time)
            max_cluster_statistic = np.abs(cluster_statistic_i).max()
            return max_cluster_statistic

        max_cluster_statistic = Parallel(n_jobs=n_jobs, verbose=10)(delayed(myfunc)(XX, yy, p_value_threshold, proximity_matrix_space_time) for i in range(iterations))

    cluster_statistic_threshold = np.sort(max_cluster_statistic)[iterations * (1.0 - p_value_threshold)]
    print "cluster_statistic threshold (for p_value=%s) = %s" % (p_value_threshold, cluster_statistic_threshold)
    clusters_over_threshold = np.where(np.abs(cluster_statistic) > cluster_statistic_threshold)[0]
    print "Number of clusters more extreme than threshold: %s" % len(clusters_over_threshold)
    if len(clusters_over_threshold) > 0: print "Clusters:"
    for i, cl in enumerate(clusters_over_threshold):
        print "%s) id: %s, size: %s, clust.stat.: %s" % (i, cl, len(clusters[cl]), cluster_statistic[cl])
    
    t, p = ttest_ind(XX[yy==-1], XX[yy==1], axis=0)

                  
