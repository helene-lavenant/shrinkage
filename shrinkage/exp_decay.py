import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, eigh, inv
from scipy.stats import gaussian_kde

class TwoExpDecayShrinkage:
    """
    Nonlinear shrinkage for autocorrelation matrix A given by the sum
    of two exponentially decaying Toeplitz matrices.
    """
    def __init__(self, Y, tau1, tau2, alpha1=0.5, alpha2=0.5):
        self.Y = Y
        self.N, self.T = Y.shape
        self.tau1 = tau1
        self.tau2 = tau2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.name = "Two Exp Decay Shrinkage"

        self.build_autocorr_matrix()

        # Compute empirical eigenvalues of the sample covariance matrix
        self.lambdas = np.linalg.eigvalsh((1 / self.T) * Y @ Y.T)
        self.E_eigval = np.sort(self.lambdas)

        # Prepare a grid of u = alpha + i * beta
        self.u_range = self.generate_u_range()

        self.plot_colors = {
            'xi_bars': 'xkcd:mauve',
            'xi_line': 'xkcd:maroon',
            'xi_hilbert': 'xkcd:rose',
        }
        self.bns = 100 

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

    def generate_u_range(self):
        """
        Construct a complex-valued grid of u = alpha + i * beta,
        for each observed eigenvalue.
        """
        self.beta = np.sqrt(self.E_eigval)
        self.alpha = np.zeros_like(self.beta)
        return self.alpha + 1j * self.beta

    def calculate_xi(self):
        """
        Compute shrunk eigenvalues xi_i = lambda_i * chi(u_i) / beta_i,
        where u_i = alpha_i + i * beta_i is defined from the observed spectrum.
        """
        self.chi = np.array([self.chi_A(u) for u in self.u_range])
        self.xi = self.E_eigval * self.chi.real / self.beta
    
    @staticmethod
    def epanechnikov_estimates(x, bandwidth):
        # Estimate density with Epanechnikov kernel using Gaussian KDE as proxy
        kde = gaussian_kde(x, bw_method=bandwidth / np.std(x, ddof=1))
        density = kde.evaluate(x)

        # Hilbert transform via numerical Hilbert transform
        from scipy.signal import hilbert
        hilbert_transform = np.imag(hilbert(density))

        return density, hilbert_transform
    
    def calculate_epanechnikov_estimates_xi(self):
        self.xi_kernel_density, self.xi_kernel_Hilbert = self.epanechnikov_estimates(
            x=self.xi,
            bandwidth=self.bandwidth if hasattr(self, "bandwidth") else 0.1,
        )
    def hist(
    self,
    show_xi=False,
    show_xi_density=False,
    show_xi_Hilbert=False,
    ylim=None,
    xlim=None,
    legend=True,
    savefig=None,
    **kwargs,):
        """
        Plot histogram and density of the shrunk eigenvalues xi_i.
        """
        if show_xi:
            plt.hist(
                self.xi,
                bins=self.bns,
                alpha=0.5,
                color=self.plot_colors.get('xi_bars', 'black'),
                density=True,
                label=f'{self.name} shrunk eigval',
            )

        if show_xi_density and hasattr(self, "xi_kernel_density"):
            plt.plot(
                self.xi,
                self.xi_kernel_density,
                color=self.plot_colors.get('xi_line', 'black'),
                label=f'{self.name} shrunk eigval density',
            )

        if show_xi_Hilbert and hasattr(self, "xi_kernel_Hilbert"):
            plt.plot(
                self.xi,
                self.xi_kernel_Hilbert,
                color=self.plot_colors.get('xi_hilbert', 'black'),
                label=f'{self.name} shrunk eigval Hilbert',
            )

        plt.xlabel('eigenvalue')
        plt.ylabel('probability density')

        if legend:
            plt.legend()
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)
        if savefig:
            plt.savefig(savefig)

    def plot(
        self,
        show_xi=False,
        xlim=None,
        ylim=None,
        legend=True,
        savefig=None,
        **kwargs,
        ):
        """
        Plot xi_i as a function of the original sorted eigenvalues.
        """
        if show_xi:
            plt.plot(
                self.E_eigval,
                self.xi,
                color=self.plot_colors.get('xi_line', 'black'),
                label=f'{self.name} shrunk eigval',
            )

        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\xi$')

        if legend:
            plt.legend()
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)
        if savefig:
            plt.savefig(savefig)
