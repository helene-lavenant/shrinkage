import numpy as np
from scipy.linalg import toeplitz, eigh, inv

class Auto2Exp:
    """
    Constructs a matrix A = exp(-|i-j|/tau1) + exp(-|i-j|/tau2)
    and exposes its eigenvalues and a method for computing the empirical Chi-transform.
    """
    def __init__(self, T, tau, rotate_A=False):
        self.T = T
        self.tau = np.atleast_1d(tau)
        assert len(self.tau) == 2, "tau must be a list or array of two values"
        self.rotate_A = rotate_A
        self._generate_A()

    def _generate_A(self):
        indices = np.arange(self.T)
        A1 = toeplitz(np.exp(-indices / self.tau[0]))
        A2 = toeplitz(np.exp(-indices / self.tau[1]))
        A = A1 + A2
        if self.rotate_A:
            from scipy.stats import ortho_group
            O = ortho_group.rvs(dim=self.T)
            A = O @ A @ O.T
        self.A = A
        self.A_eigval, _ = eigh(self.A)

    def chi_transform(self, u):
        return (1 / self.T) * np.trace(inv(np.eye(self.T) + u * self.A))

# Example usage
if __name__ == '__main__':
    T = 100
    taus = [3.0, 10.0]
    model = Auto2Exp(T=T, tau=taus)
    u = 0.5
    chi_val = model.chi_transform(u)
    print(f"Chi_A({u}) = {chi_val:.5f}")