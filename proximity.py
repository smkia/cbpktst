import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
# from scipy.spatial.kdtree import KDTree
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from sys import stdout


def compute_boolean_proximity_matrix(coordinates, threshold):
    """Compute the boolean proximity matrix of units (with given
    coordinates) which have Euclidean distance less than threshold.

    coordinates is a matrix where each row is the vector of
    coordinates of a unit/sensor.
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


def compute_sparse_boolean_proximity_matrix_space_time(coordinates, n_timesteps, threshold_space=0.1, threshold_timesteps=1, space_sparse=False, verbose=False):
    """Create the proximity matrix of a set of units with given
    coordinates in space and time, where two units are proximal if
    their Euclidean distance is less then the given threshold_space
    and their linear distance is less then the given threshold_time.

    In practice we create a proximity matrix where each node is one
    sensor at one specific timestep. Two sensors (irrespective of
    time) are proximal if their Euclidean distance is less than
    threshold_space. Moreover the same sensor is proximal to itself at
    a different timesteps is the time distance is equal or less then
    threshold_timesteps. Since the proximity matrix becomes huge,
    i.e. ((n_units * n_timesteps) , (n_units * n_timesteps)), but
    sparse, we build it as a scipy.sparse.lil_matrix. The matrix is
    constructed by replicating the spatial proximity matrix on the
    diagonal a number of times equal to the number of timesteps. The
    time proximity is encoded with (or more) offset diagonal(s) set to
    True.

    The ordered list of nodes on the rows (and equally on the columns)
    is:
    un0[t=0],...,unN[t=0],un0[t=1],...,unN[t=1],...,un0[t=M],...,unN[t=M]
    where N is the number of units (n_units) and M is the number
    of timesteps (n_timesteps).

    Note: lil_matrix is the necessary format to allow fancy indexing
    when patching the spatial proximity matrix into the proximity
    matrix and to set the diagonals. But in order to extract
    submatrices in ktst_map.compute_clusters_statistic(), it is
    necessary to convert the lil_matrix into a csc_matrix.
    """
    if verbose: print("compute_sparse_boolean_proximity_matrix_space_time()")
    if verbose: print("Computing the space proximity matrix.")
    if space_sparse:
        proximity_matrix_space = compute_sparse_boolean_proximity_matrix(coordinates, threshold_space)
    else:
        proximity_matrix_space = compute_boolean_proximity_matrix(coordinates, threshold_space)

    if verbose: print("Creating the very large sparse space time proximity matrix.")
    n_units = len(coordinates)
    proximity_matrix = lil_matrix((n_units * n_timesteps, n_units * n_timesteps), dtype=np.bool)

    if verbose: print("Filling the sparse matrix with the space proximity matrix as a block on the diagonal")
    for i in range(n_timesteps):
        if verbose:
            print(i),
            stdout.flush()
            
        proximity_matrix[i*n_units:(i+1) * n_units, i*n_units:(i+1) * n_units] = proximity_matrix_space

    if verbose: print("Filling the offset diagonals with 'True' to encode proximity in time.")
    for k in range(1, threshold_timesteps + 1):
        proximity_matrix.setdiag(np.ones(n_units * (n_timesteps - k), dtype=np.bool), n_units * k)
        proximity_matrix.setdiag(np.ones(n_units * (n_timesteps - k), dtype=np.bool), - n_units * k)

    return proximity_matrix.tocsc()
