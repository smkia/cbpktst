import numpy as np
import matplotlib.pyplot as plt
from proximity import compute_boolean_proximity_matrix, compute_sparse_boolean_proximity_matrix

def simulate_2d(d=20, k=10, m=200, n=200, threshold=1.0):
    """This data simulator generates two samples, of m and n trials
    each. Each trial consists of measurements on a 2D grid of (k x k)
    sensors (units). Each measurement is a d-dimensional vector. By
    design, the sensors (units) close to location (0,0) show a strong
    difference between the two samples. Those far from (0,0) show
    almost no difference between the two samples.
    """
    print("Creating simulated data.")
    n_units = k * k
    coordinates_2D = np.meshgrid(np.arange(k), np.arange(k))
    coordinates = np.array(zip(coordinates_2D[0].flat,coordinates_2D[1].flat))
    sigma2_spatial = (np.sqrt(((coordinates + 1.0)**2).sum(1)) * 50.0)
    XX = np.array([np.random.multivariate_normal(np.zeros(d), s2 * np.eye(d), size=m) for s2 in sigma2_spatial]).transpose(1,0,2)
    YY = np.array([np.random.multivariate_normal(np.ones(d), s2 * np.eye(d), size=n) for s2 in sigma2_spatial]).transpose(1,0,2)
    # proximity_matrix = compute_boolean_proximity_matrix(coordinates, threshold=threshold) # this is a dense matrix, which is OK for low k
    proximity_matrix = compute_sparse_boolean_proximity_matrix(coordinates, threshold=threshold) # this is a sparse matrix which is good for large k
    print("%f of the pairs of units are proximal." % ((len(proximity_matrix.nonzero()[0]) - n_units) / 2.0  / (n_units * (n_units - 1.0) / 2.0)))

    return XX, YY, proximity_matrix, coordinates


def plot_map2d(k, coordinates, unit_statistic_homogeneous_significant, vmin=None, vmax=None):
    """Plots 2d sensor (unit) map related to simulate_2d() data.
    """
    map2d = np.zeros((k, k))
    for i, (x, y) in enumerate(coordinates):
        map2d[x, y] = unit_statistic_homogeneous_significant[i]
        
    plt.interactive(True)
    plt.figure()
    plt.imshow(map2d, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    


if __name__ == '__main__':

    XX, YY, proximity_matrix = simulate_2d()




