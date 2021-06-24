import numpy as np
from kernels import *


class SmoothBetaProcess:
    def __init__(self, delta=0.1):

        # Delta is a smoothing hyperparameter, larger --> smoother
        self.delta = delta
        # kernel can be changed; see kernels.py
        self.kernel = lambda x1, x2: uniform_kernel(x1, x2, self.delta)

        self.prior_alpha = 1.
        self.prior_beta = 1.

        self.x_good = None   # Training points where y = 1
        self.x_bad = None    # Training points where y = 0

    def fit(self, x, y):
        """
        Args:
            x: input features, shape (n_samples, n_dim)
            y: 0/1 binary outcomes, shape (n_samples,)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        self.x_good = x[y == 1]
        self.x_bad = x[y == 0]

    def get_posterior_params(self, x):
        """
        Args:
            x: input features, shape (n_samples, n_dim)

        Returns:
            A pair (alpha, beta) corresponding to the posterior parameters
            of a Beta distribution at every point in x.
        """
        posterior_alpha = np.zeros(x.shape[0]) + self.prior_alpha
        posterior_beta = np.zeros(x.shape[0]) + self.prior_beta

        for i, pt in enumerate(x):
            posterior_alpha[i] = np.sum([self.kernel(pt, x_g) for x_g in self.x_good])
            posterior_beta[i] = np.sum([self.kernel(pt, x_b) for x_b in self.x_bad])

        return posterior_alpha, posterior_beta

    def get_mean(self, x):
        """
        Gets the posterior means for the points in x.
        """
        posterior_alpha, posterior_beta = self.get_posterior_params(x)

        return posterior_alpha / (posterior_alpha + posterior_beta)

    def get_mode(self, x):
        """
        Gets the posterior modes for the points in x.
        """
        posterior_alpha, posterior_beta = self.get_posterior_params(x)

        return (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)