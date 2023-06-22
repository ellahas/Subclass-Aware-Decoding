import numpy as np
from scipy.linalg import fractional_matrix_power
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann


class ParallelTransport:
    """
    Parallel transport to the Identity for each class mean.
    Based on Kolkhorst et al. (2020).

    Attributes
    ----------
    means: ndarray
        The means of each subclass
    ts: TangentSpace
    """
    def fit(self, X, sublab, fit_ts=False):
        """
        Calculate the class mean for each subclass

        Parameters
        ----------
        X: ndarray
            ERP covariance matrices of training data
        sublab: ndarray
            Subclass labels of each data point
            Must all be int from 1 to 9 (including)
        fit_ts: Bool
            Whether to fit the tangent point of the tangent space projection,
            or to take the identity as the tangent point

        Returns
        -------
        self
        """
        self.means = dict()
        for sub in np.unique(sublab):
            X_sub = X[sublab==sub]
            self.means[sub] = mean_riemann(X_sub)
        self.ts = TangentSpace()
        if fit_ts:
            self.ts.fit(X)
        return self

    def transform(self, X, sublab):
        """
        Transport X in Riemannian space in such a way that the class means
        per subclass are transported to the Identity.
        Then project onto tangent space.

        Parameters
        ----------
        X: ndarray
            ERP covariance matrices of data
        sublab: ndarray
            Subclass labels of each data point
            Must all be int from 1 to 9 (including)

        Returns
        -------
        X_ts: ndarray
            The transported and projected data
        """
        # transport each mean to I
        X_tp = np.zeros(X.shape)
        for i in range(X.shape[0]):
            sub_mean = self.means[sublab[i]]
            sub_mean_inv_root = fractional_matrix_power(sub_mean, -0.5)
            X_tp[i, :, :] = sub_mean_inv_root @ X[i, :, :] @ sub_mean_inv_root
        # tangent space projection
        X_ts = self.ts.transform(X_tp)
        return X_ts
