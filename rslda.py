import numpy as np
from sklearn.covariance import LedoitWolf
from qpsolvers import solve_qp
from qpsolvers.exceptions import SolverError, ParamError, ProblemError
from toeplitzlda.classification.covariance import ToepTapLW


def multitarget_shrinkage(data):
    """ Shrink mean of first data matrix towards the others

    Parameters
    ----------
    data: dict
        data[0]: matrix with all subclass data points
        data[rest]: matrices with other subclass data points

    Returns
    -------
    theta_shr: float array
        The shrinkage mean of the current subclass
    """

    # initialize some variables
    n1 = len(data.keys())
    X0 = data[0]
    [N0, D] = X0.shape

    # shrink the mean

    theta = np.mean(X0, axis=0)

    # stats for similar data sets
    if n1 > 1:
        target_means = np.zeros((n1-1, D))
        for i in range(n1-1):
            target_means[i, :] = np.mean(data[list(data.keys())[i+1]], axis=0)

    # Optimization
    A = np.zeros((n1-1, n1-1))
    for k in range(n1-1):
        for l in range(n1-1):
            A[k, l] = (target_means[k, :] - theta) @ (target_means[l, :]
                                                      - theta)

    b = np.zeros(n1-1)
    for k in range(n1-1):
        b[k] = np.sum(np.var(target_means[k, :]))

    G = np.ones(n1-1)
    try:
        lamb = solve_qp(A, b, G=G, h=np.ones(1), lb=np.zeros(n1-1),
                        solver='daqp', iter_limit=2e9)
        lamb[lamb < 0] = 0
    except ParamError:
        print('Parmeter wrong')
        return None
    except ProblemError:
        print('Problem incorrectly defined')
        return None
    except SolverError:
        print('Failed during execution')
        return None
    except TypeError:
        print('None found, but no solver error')
        return None

    # Shrinkage mean
    theta_shr = (1 - sum(lamb)) * theta
    for i in range(n1-1):
        theta_shr += lamb[i] * target_means[i,:]
    return theta_shr, lamb


class SeparateLDA:
    """
    Fit an LDA model for each subclass.

    Attributes
    ----------
    subLDA: dict
        The LDA weights for each subclass
    n_channels : int
        The number of channels recorded, needed for Toeplitz covariance shrinkage
    unique_sublabels : ndarray
        The subclass labels present
    """
    def __init__(self, n_channels=None):
        """
        Initialize the classifier

        Parameters
        ----------
        n_channels: int, optional
            Number of recorded channels, only necessary for Block-Toeplitz
        """
        self.n_channels = n_channels
        self.subLDA = dict()

    def fit(self, X, y, sublab, toeplitz=False, share_cov=True, channel_prime=False):
        """
        Fit an LDA model for each subclass

        Parameters
        ----------
        X: ndarray
            The training dataZ
        y: ndarray
            Main labels of each data point
        sublab: ndarray
            Subclass labels of each data point
            Must all be int from 1 to 9 (including)
        toeplitz: Bool, optional
            If True, covariance is estimated with block-Toeplitz, otherwise with Ledoit-Wolf
            Only set to true for time series data
        share_cov: Bool, optional
            If True, a global covariance is used to train the separate models
            Otherwise, a separate covariance is estimated per subclass
        channel_prime: Bool, optional
            Whether the data in X is channel prime

        Returns
        -------
        self
        """

        self.unique_sublabels = np.unique(sublab)

        if share_cov:
            if toeplitz:
                C_est = ToepTapLW(n_channels=self.n_channels,
                                  assume_centered=False,
                                  data_is_channel_prime=channel_prime).fit(X).covariance_
            else:
                C_est = LedoitWolf(assume_centered=False).fit(X).covariance_
            C_invcov = np.linalg.pinv(C_est)

        # Finalize cls
        for sub in self.unique_sublabels:
            X_sub = X[sublab==sub]
            y_sub = y[sublab==sub]
            if not share_cov:
                if toeplitz:
                    C_est = ToepTapLW(n_channels=self.n_channels,
                                      assume_centered=False,
                                      data_is_channel_prime=channel_prime).fit(X_sub).covariance_
                else:
                    C_est = LedoitWolf(assume_centered=False).fit(X_sub).covariance_
                C_invcov = np.linalg.pinv(C_est)
            # Get target and non-target means
            M1_est = np.mean(X_sub[y_sub==0], axis=0).T
            M2_est = np.mean(X_sub[y_sub==1], axis=0).T
            # Train LDA -> add to class's attribute of subclass LDAs
            self.subLDA[sub] = self.train_lda(C_invcov, M1_est, M2_est)
        return self

    def train_lda(self, C_inv, M1_est, M2_est):
        """
        Train an LDA model on the class means and the inverse of the covariance

        Parameters
        ----------
        C_inv : ndarray
            The inverse of the estimated covariance of both classes
        M1_est : ndarray
            The estimated mean of the first class
        M2_est : ndarray
            The estimated mean of the second class

        Returns
        -------
        clf : dict
            The weights and bias of the LDA model
        """
        clf = dict()
        clf['w'] = C_inv @ (M2_est - M1_est)
        clf['b'] = -0.5 * clf['w'].T @ (M1_est + M2_est)
        return clf

    def decision_function(self, X, sublab):
        """
        Apply the decision function to each data point in X

        Parameters
        ----------
        X: [MxN]
            Data matrix with M data points and N features
        sublab: [M]
            The sublabel for each data point in X

        Returns
        -------
        f: float array
            The distance to the decision boundary for each point in X
        """
        f = np.zeros(len(X))
        for sub in self.unique_sublabels:
            X_sub = X[sublab==sub]
            f[sublab==sub] = self.subLDA[sub]['w'] @ X_sub.T + self.subLDA[sub]['b']
        return f

    def predict(self, X, sublab):
        """
        Predict the class of each data point in X

        Parameters
        ----------
        X: [MxN]
            Data matrix with M data points and N features
        sublab: [M]
            The sublabel for each data point in X

        Returns
        -------
        res: int array
            The class labels
        """
        f = self.decision_function(X, sublab)
        res = [0 if i < 0 else 1 for i in f]
        return res

    def predict_proba(self, X, sublab):
        """
        Probability of predicting each class for each data point in X
        Credit to Jan Sosulski

        Parameters
        ----------
        X: [MxN]
            Data matrix with M data points and N features
        sublab: [M]
            The sublabel for each data point in X

        Returns
        -------
        float 2D array
            The probability of classifying each class for each data point
        """
        prob = self.decision_function(X, sublab)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        return np.column_stack([1 - prob, prob])


class RSLDA(SeparateLDA):
    """
    Relevance Subclass LDA (Höhne et al., 2016)

    Attributes
    ----------
    subLDA : dict
        The LDA weights for each subclass
    n_channels : int
        The number of channels recorded, needed for Toeplitz covariance shrinkage
    unique_sublabels : ndarray
        The subclass labels present
    M_ests : dict
        The estimated shrunk means of each subclass
    lambs : dict
        The weights used in shrinking the class means of each subclass,
        keyed with numbers below 10 for non-target means, and over 10 for target means
    """

    def fit(self, X, y, sublab, whitening=True, toeplitz=True,
            channel_prime=False, global_mean=[False, False]):
        """
        Train RSLDA on training data

        Parameters
        ----------
        X: [MxN]
            Training data with N features and M data points
        y: [1XM]
            Main labels of each data point
        sublab: [1xM]
            Subclass labels of each data point
            Must all be int from 1 to 9 (including)
        whitening: Bool, optional
            If True, X is whitened before estimating the subclass means
        toeplitz: Bool
            If True, covariance is estimated with block-Toeplitz, otherwise with Ledoit-Wolf
        channel_prime: Bool, optional
            Whether the data in X is channel prime
        global_mean: [Bool, Bool], optional
            Whether to use the global class mean instead of subclass means for Non-Target and Target

        Returns
        -------
        self
        """

        # Get unique sublabels
        self.unique_sublabels = np.unique(sublab)
        # Perform whitening & Compute Cov matrix
        if whitening:
            M1 = np.mean(X[y==0], axis=0)
            M2 = np.mean(X[y==1], axis=0)
            # subtract class means from X
            X_meanfree_dum = np.array([X[i, :] - M1 if y[i] == 0 else X[i, :]
                                       - M2 for i in range(len(X))])
            # estimate global cov
            # (with shrinkage Ledoit-Wolf or block-Toeplitz)
            if toeplitz:
                global_cov = ToepTapLW(n_channels=self.n_channels,
                                       assume_centered=True,
                                       data_is_channel_prime=channel_prime).fit(X_meanfree_dum).covariance_
            else:
                global_cov = LedoitWolf(assume_centered=True).fit(X_meanfree_dum).covariance_
            # eigenvalue decomp
            [eigval, eigvec] = np.linalg.eig(global_cov)
            A_feat2white = np.diag(eigval**-0.5) @ eigvec
            A_white2feat = np.linalg.inv(A_feat2white)
        else:
            A_feat2white = np.identity(X.shape[1])
            A_white2feat = np.identity(X.shape[1])
        Xwhite = X @ A_feat2white

        # Create array with corresponding subclass mean for each data point
        M = np.zeros(Xwhite.shape)
        self.M_ests = dict()
        self.lambs = dict()
        for sub in self.unique_sublabels:
            for i in range(2):
                # Optional: check for presence of subclass
                # Estimate class means
                X_class = Xwhite[y==i]
                sublabs_class = sublab[y==i]
                if not global_mean[i]:
                    shrink_data = {0: X_class[sublabs_class==sub]}
                    for other_sub in self.unique_sublabels:
                        if other_sub != sub:
                            shrink_data[other_sub] = X[sublab==other_sub]
                    # Perform multi-target shrinkage on subclass data shrinked
                    # towards all others
                    M_est, lamb = multitarget_shrinkage(shrink_data)
                    self.lambs[10 * i + sub] = lamb
                else:
                    M_est = np.mean(X_class, axis=0)
                # Transform means back into feature space (unwhiten)
                M_est_original_space = np.transpose(A_white2feat) @ M_est
                # Save subclass mean
                M[sublab==sub] = M_est_original_space
                # Save mean estimators
                if i == 0:
                    self.M_ests[sub] = M_est_original_space
                else:
                    self.M_ests[sub] = np.vstack((self.M_ests[sub],
                                                  M_est_original_space))

        # Estimate Cov with optimized means
        X_meanfree = X - M
        if toeplitz:
            C_est = ToepTapLW(n_channels=self.n_channels, assume_centered=True,
                              data_is_channel_prime=channel_prime).fit(X_meanfree).covariance_
        else:
            C_est = LedoitWolf(assume_centered=True).fit(X_meanfree).covariance_
        C_invcov = np.linalg.pinv(C_est)

        # Finalize cls
        for sub in self.unique_sublabels:
            # Get target and non-target means
            M1_est = self.M_ests[sub][0,:].T
            M2_est = self.M_ests[sub][1,:].T
            # Train LDA -> add to class's attribute of subclass LDAs
            self.subLDA[sub] = self.train_lda(C_invcov, M1_est, M2_est)

        return self
