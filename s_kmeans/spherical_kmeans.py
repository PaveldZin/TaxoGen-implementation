#Copyright (c) 2015-2017, François Role, Stanislas Morbieu, Mohamed Nadif
#All rights reserved.

#This source code is licensed under the BSD-style license found in the
#LICENSE file in the same directory of this file. 

# Modified by: Pavel Zinovev
# Removed sparse matrix support as we work with full embeddings and adjusted the code to work with numpy arrays


import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize

def random_init_clustering(n_clusters, n_rows, random_state=None):
    """Create a random row cluster assignment matrix.

    Each row contains 1 in the column corresponding to the cluster where the
    processed data matrix row belongs, 0 elsewhere.

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    n_rows: int
        Number of rows of the data matrix (i.e. also the number of rows of the
        matrix returned by this function)
    random_state : int or :class:`numpy.RandomState`, optional
        The generator used to initialize the cluster labels. Defaults to the
        global numpy random number generator.

    Returns
    -------
    matrix
        Matrix of shape (``n_rows``, ``n_clusters``)
    """

    random_state = check_random_state(random_state)
    Z_a = random_state.randint(n_clusters, size=n_rows)
    Z = np.zeros((n_rows, n_clusters))
    Z[np.arange(n_rows), Z_a] = 1
    return Z

class SphericalKmeans:
    """Spherical k-means clustering.

    Parameters
    ----------
    n_clusters : int, optional, default: 2
        Number of clusters to form
    init : numpy array,
        shape (n_features, n_clusters), optional, default: None
        Initial column labels
    max_iter : int, optional, default: 300
        Maximum number of iterations
    n_init : int, optional, default: 10
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-4
        Relative tolerance with regards to criterion to declare convergence
    verbose : boolean, optional, default: False
    
    Attributes
    ----------
    labels_ : array-like, shape (n_rows,)
        cluster label of each row
    criterion : float
        criterion obtained from the best run
    criterions : list of floats
        sequence of criterion values during the best run
    """
    def __init__(self, n_clusters=2, init=None, max_iter=300, n_init=10,
                 tol=1e-4, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.labels_ = None
        self.criterions = []
        self.criterion = -np.inf
        self.Z = None
        self.Z_fuzzy = None
        self.verbose = verbose
        
    def fit(self, X):
        """Perform clustering.

        Parameters
        ----------
        X : numpy array, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        criterion = self.criterion

        X = normalize(X)

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            if self.verbose:
                print(" == New init == ")
            self.random_state = seed
            self._fit_single(X)
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                labels_ = self.labels_
                z = self.Z
                z_fuzzy = self.Z_fuzzy

        self.random_state = random_state

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = labels_
        self.Z = z
        self.Z_fuzzy = z_fuzzy

    def _fit_single(self, X):
        """Perform one run of clustering.

        Parameters
        ----------
        X : numpy array, shape=(n_samples, n_features)
            Matrix to be analyzed
        """        
        K = self.n_clusters

        if self.init is None:
            Z = random_init_clustering(K, X.shape[0], self.random_state)
        else:
            Z = self.init

        change = True

        c_init = -np.inf
        c_list = []
        n_iter = 0

        while change and n_iter < self.max_iter:
            if self.verbose:
                print("iteration:", n_iter)
            change = False

            # compute centroids (in fact only summation along cols)
            centers = Z.T @ X  

            # normalize centroids
            centers = normalize(centers)

            # hard assignment
            #Z=centers*X.T
            Z1 = X @ centers.T
  
            Z = np.zeros_like(Z1)
            Z[np.arange(len(Z1)), Z1.argmax(1)] = 1

            # compute and check if cosine criterion still evolves
            k_times_k = Z.T @ X @ centers.T
            c = np.trace(k_times_k)

            if np.abs(c - c_init) > self.tol:
                c_init = c
                change = True
                c_list.append(c)
                if self.verbose:
                    print(c)
            n_iter += 1

        self.criterion = c
        self.criterions = c_list
        self.labels_ = Z.argmax(axis=1).tolist()
        self.Z = Z
        self.Z_fuzzy = Z1
