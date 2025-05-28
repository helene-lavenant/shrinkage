import numpy as np


from .varma import Varma
from .rie_lp import LedoitPecheShrinkage


class TwoExpDecay(LedoitPecheShrinkage):
    """
    We further build on the Ledoit-Peche shrunk eigenvalues xi^LP_i, i = 1, ..., N,
    and perform nonlinear shrinkage, computing shrunk eigenvalues xi_i,
    in the case of auto-correlations given as the sum of two exponentially decaying matrices.
    """
    def __init__(
        self,
        tau1,
        tau2,
        alpha1,
        alpha2,
        **kwargs,
    ):
        super().__init__(
            name = 'TED',
            **kwargs,
        )

        self.tau1 = tau1
        self.tau2 = tau2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.build_autocorr_matrix()

        self.alpha, self.beta, self.u_range = self.u_alpha_beta(self.q)

        self.calculate_xi()
        self.calculate_epanechnikov_estimates_xi()

        self.plot_colors = dict(
            **self.plot_colors,
            **{
                'xi_bars': 'xkcd:ochre',
                'xi_line': 'xkcd:crimson',
                'xi_hilbert': 'xkcd:rust',
            },
        )


    def build_autocorr_matrix(self):
        """
        Build A = alpha1 * exp(-|i-j|/tau1) + alpha2 * exp(-|i-j|/tau2),
        where each term is a Toeplitz matrix.
        """
        from scipy.linalg import toeplitz

        indices = np.arange(self.T)
        A1 = np.exp(-np.abs(indices) / self.tau1)
        A2 = np.exp(-np.abs(indices) / self.tau2)
        self.A = self.alpha1 * toeplitz(A1) + self.alpha2 * toeplitz(A2)

        # Precompute eigenvalues of A for resolvent/M-transform
        self.A_eigvals = np.linalg.eigvalsh(self.A)

    def G_A(self, z):
        """
        Stieltjes transform (resolvent) of matrix A.
        """
        return np.mean(1 / (z - self.A_eigvals))

    def M_A(self, z):
        """
        M-transform: M_A(z) = z * G_A(z) - 1
        """
        return z * self.G_A(z) - 1

    def chi_A(self, u):
        """
        Numerically solve for chi such that M_A(1 / chi) = u.
        """
        from scipy.optimize import root_scalar

        def func(x):
            return self.M_A(1 / x).real - u.real  # assuming u is real

        sol = root_scalar(func, bracket=[1e-6, 100], method='brentq')
        return sol.root if sol.converged else np.nan

    def u_alpha_beta(self, q):
        """
        u = alpha + i * beta,
        for each observed eigenvalue.
        """
        alpha = q * (np.pi * self.E_eigval * self.E_eigval_kernel_Hilbert - 1.)
        beta = np.pi * q * self.E_eigval * self.E_eigval_kernel_density

        u_range = np.array([complex(al, be) for al, be in zip(alpha, beta)])

        return alpha, beta, u_range

    def calculate_xi(self):
        """
        Compute shrunk eigenvalues xi_i = lambda_i * chi(u_i) / beta_i,
        where u_i = alpha_i + i * beta_i is defined from the observed spectrum.
        """
        self.chi = np.array([self.chi_A(u) for u in self.u_range])
        self.xi = self.E_eigval * self.chi.real / self.beta
    
    def calculate_epanechnikov_estimates_xi(self):
        """
        Perform Epanechnikov kernel estimation of the density and Hilbert transform
        of the shrunk eigenvalues xi.
        """
        self.xi_kernel_density, self.xi_kernel_Hilbert = self.epanechnikov_estimates(
            x = self.xi,
            bandwidth = self.bandwidth,
        )
    
    