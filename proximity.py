import numpy as np
from scipy.sparse import coo_matrix
# from scipy.spatial.kdtree import KDTree
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances


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
