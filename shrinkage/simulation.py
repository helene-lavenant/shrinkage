import numpy as np

from scipy.linalg import toeplitz, sqrtm, inv
from scipy.stats import ortho_group
from scipy.optimize import minimize

from .varma import Varma


class PopulationCovariance:
    """
    Generate synthetic population ("true") covariance matrix C,
    according to select models.
    """
    def __init__(
        self,
        N,
        C_model = 'unit',
        rotate_C = False,
        **kwargs,
    ):
        self.N = N
        self.C_model = C_model
        self.rotate_C = rotate_C

        self.kwargs = kwargs

        self.generate_C()

        self.sqrt_C = sqrtm(self.C).real
        self.inv_C = inv(self.C)
        self.sqrt_inv_C = sqrtm(self.inv_C).real
    

    def generate_C(self):
        """
        Generate C.
        """
        if self.C_model == 'unit':
            self.C = np.eye(self.N)
        
        elif self.C_model == 'clusters':
            self.f_list = self.kwargs.get('f_list')
            self.e_list = self.kwargs.get('e_list')

            assert len(self.f_list) == len(self.e_list) - 1
            
            f_list_full = [int(f * self.N) for f in self.f_list]
            f_list_full += [self.N - sum(f_list_full)]
            
            C_list = [s * [e] for s, e in zip(f_list_full, self.e_list)]
            self.C = np.diag([c for sublist in C_list for c in sublist])
        
        elif self.C_model == 'inverse-Wishart':
            self.kappa = self.kwargs.get('kappa')
            self.seed = self.kwargs.get('seed')
            
            q_IW = 1. / (1. + 2 * self.kappa)
            T_IW = int(self.N / q_IW)

            rng = np.random.default_rng(
                seed = self.seed,
            )
            R = rng.standard_normal(
                size = (self.N , T_IW),
            )
            W = R @ R.T / T_IW
            self.C = (1. - q_IW) * inv(W)
        
        elif self.C_model == 'Kumaraswamy':
            self.condition_number = self.kwargs.get('condition_number')
            self.a = self.kwargs.get('a')
            self.b = self.kwargs.get('b')
            self.seed = self.kwargs.get('seed')
            
            rng = np.random.default_rng(
                seed = self.seed,
            )
            k = rng.uniform(
                size = self.N,
            )
            kum = (1. - (1. - k) ** (1 / self.b)) ** (1 / self.a)
            C_eigvals = 1. + (self.condition_number - 1.) * kum
            C_eigvals.sort()
            
            self.C = np.diag(C_eigvals)
        
        elif self.C_model == 'explicit':
            self.C = self.kwargs.get('C')

        else:
            raise Exception('Unknown method to generate C.')
        
        # optionally, rotate such a generated C by an orthogonal similarity transformation

        if self.rotate_C:
            O = ortho_group.rvs(dim = self.N)
            self.C = O @ self.C @ O.T


class AutoCovariance:
    """
    Generate synthetic auto-covariance matrix A,
    according to select models.
    """
    def __init__(
        self,
        T_total,
        A_model = 'unit',
        rotate_A = False,
        **kwargs,
    ):
        self.T_total = T_total
        self.A_model = A_model
        self.rotate_A = rotate_A

        self.kwargs = kwargs

        self.tau = self.kwargs.get("tau", None)

        self.generate_A()

        self.sqrt_A = sqrtm(self.A).real
        self.inv_A = inv(self.A)
        self.sqrt_inv_A = sqrtm(self.inv_A).real
    

    def generate_A(self):
        """
        Generate A.
        """
        if self.A_model == 'unit':
            self.A = np.eye(self.T_total)
        
        elif self.A_model in ['exp-decay', 'VARMA']:
            varma = Varma(
                T = self.T_total,
                **self.kwargs,
            )
            self.a_list = varma.a_list
            self.b_list = varma.b_list
            self.r1 = varma.r1
            self.r2 = varma.r2
            self.A = varma.A
        
        elif self.A_model == '2-exp-decay':
            taus = np.atleast_1d(self.kwargs.get('tau'))
            if len(taus) != 2:
                raise ValueError("tau must be a list or array of two values for 2-exp-decay")
            A1 = toeplitz(np.exp(-np.arange(self.T_total) / taus[0]))
            A2 = toeplitz(np.exp(-np.arange(self.T_total) / taus[1]))

            a_list = self.kwargs.get('a_list', [0.5, 0.5])  # default: equal weights

            self.A = a_list[0]*A1 + a_list[1]*A2
        
        elif self.A_model == 'EWMA':
            self.delta = self.kwargs.get('delta')
            eps = 1. - self.delta / self.T_total
            self.A = np.diag(
                self.T_total * (1 - eps) / (1 - eps ** self.T_total) * eps ** np.arange(self.T_total)
            )
        
        elif self.A_model == 'explicit':
            self.A = self.kwargs.get('A')
        
        else:
            raise Exception('Unknown method to generate A.')
        
        if self.rotate_A:
            O = ortho_group.rvs(dim = self.T_total)
            self.A = O @ self.A @ O.T
    
    def fit_tau(self, data):
        # Estimate optimal tau (or tau1 and tau2) from the data

        if self.A_model == '2_exp_decay':
            empirical_cov = np.cov(data.T)
            initial_guess = [0.7, 0.3]
            result = minimize(self._loss_taus, initial_guess, args=(empirical_cov,), method='Nelder-Mead')
            self.tau_fit = result.x[0]
            self.A = self._2_exp_decay(self.tau_fit[0], self.tau_fit[1], empirical_cov.shape[0])

        elif self.A_model == 'exp_decay':
            empirical_cov = np.cov(data.T)
            initial_guess = 0.5
            result = minimize(self._loss_tau, initial_guess, args=(empirical_cov,), method='Nelder-Mead')
            self.tau_fit = result.x[0]
            self.A = self._exp_decay(self.tau_fit, empirical_cov.shape[0])

    @staticmethod
    def _exp_decay(tau, size):
        return toeplitz(np.exp(-np.arange(size)/tau))

    @staticmethod
    def _2_exp_decay(tau1, tau2, size):
        A1 = toeplitz(np.exp(-np.arange(size)/tau1))
        A2 = toeplitz(np.exp(-np.arange(size)/tau2))
        return A1 + A2
    
    @staticmethod
    def _loss_tau(tau, empirical_cov):
        if tau<= 0: 
            return np.inf
        A_model = AutoCovariance._exp_decay(tau, empirical_cov[0])
        return np.linalg.norm(empirical_cov-A_model, ord='fro') #Frobenius norm
    
    @staticmethod
    def _loss_taus(taus, empirical_cov):
        tau1, tau2 = taus
        if tau1 <= 0 or tau2<= 0: 
            return np.inf
        A_model = AutoCovariance._2_exp_decay(tau1, tau2, empirical_cov.shape[0])
        return np.linalg.norm(empirical_cov-A_model, ord='fro')

class DataMatrix:
    """
    Retrieve a data matrix Y of shape N x T_total:
      - either generated from a given model,
        based on the true (population) covariance matrix C,
        and the autocorrelation matrix A;
      - or loaded from an external dataset.
    """
    def __init__(
        self,
        method,
        **kwargs,
    ):
        self.method = method
        self.kwargs = kwargs

        if self.method == 'sandwich':
            self.prepare_sandwich()
            self.simulate_Y_sandwich()
        
        elif self.method == 'recurrence (warm start)' and self.kwargs.get('A_model') in ['exp-decay', 'VARMA']:
            self.prepare_sandwich()
            self.simulate_Y_recurrence_warm_start()
        
        elif self.method == 'recurrence (fixed start)' and self.kwargs.get('A_model') in ['exp-decay', 'VARMA']:
            self.prepare_sandwich()
            self.simulate_Y_recurrence_fixed_start()

        elif self.method == 'load':
            self.Y = self.kwargs.get('Y')
            self.N, self.T_total = self.Y.shape
        
        else:
            raise Exception('Unknown method to create the data matrix Y.')

        assert self.Y.shape == (self.N, self.T_total)
    

    def prepare_sandwich(self):
        """
        Calculate the cross-covariance matrix C,
        and the auto-covariance matrix A,
        based on given kwargs,
        for the use in a "sandwich" simulation model.
        """
        self.population_covariance = PopulationCovariance(**self.kwargs)
        self.auto_covariance = AutoCovariance(**self.kwargs)

        self.N = self.population_covariance.N
        self.T_total = self.auto_covariance.T_total

    
    def simulate_Y_sandwich(self):
        """
        Simulate synthetic data Y of shape N x T_total
        by first simulating an array X
        of IID random variables from a given distribution,
        then "sandwich"-ing it with square roots of C and A.
        """
        self.dist = self.kwargs.get('dist', 'Gaussian')
        self.seed = self.kwargs.get('seed')

        rng = np.random.default_rng(
            seed = self.seed,
        )
        if self.dist == 'Gaussian':
            X = rng.standard_normal(
                size = (self.N, self.T_total),
            )
        elif self.dist == 'Student-t':
            df = self.kwargs.get('df')
            X = rng.standard_t(
                df = df,
                size = (self.N, self.T_total),
            )
        else:
            raise Exception('Unknown distribution.')
        
        self.Y = self.population_covariance.sqrt_C @ X @ self.auto_covariance.sqrt_A
    

    def simulate_Y_recurrence_warm_start(self):
        """
        Simulate synthetic data Y of shape N x T_total
        by a given recurrence relation.
        """
        if self.kwargs.get('A_model') in ['exp-decay', 'VARMA']:
            self.dist = self.kwargs.get('dist', 'Gaussian')
            self.seed = self.kwargs.get('seed')
            self.warm_start = self.kwargs.get('warm_start', 1000)

            T_full = self.T_total + self.warm_start * (self.auto_covariance.r1 + self.auto_covariance.r2)

            Y = np.zeros(shape = (self.N, T_full))

            rng = np.random.default_rng(
                seed = self.seed,
            )
            if self.dist == 'Gaussian':
                eps = self.population_covariance.sqrt_C @ rng.standard_normal(
                    size = (self.N, T_full),
                )
            elif self.dist == 'Student-t':
                df = self.kwargs.get('df')
                eps = self.population_covariance.sqrt_C @ rng.standard_t(
                    df = df,
                    size = (self.N, T_full),
                )
            else:
                raise Exception('Unknown distribution.')

            for t in range(self.auto_covariance.r2, T_full):
                Y[:, t] = (
                    self.auto_covariance.a_list[::-1] * eps[:, (t - self.auto_covariance.r2):(t + 1)]
                ).sum(axis = 1)
            
            for t in range(self.auto_covariance.r1, T_full - self.auto_covariance.r2):
                Y[:, t] += (
                    self.auto_covariance.b_list[::-1] * Y[:, (t - self.auto_covariance.r1):t]
                ).sum(axis = 1)
            
            self.Y = Y[:, -self.T_total:]
        
        else:
            raise Exception('Unknown model to simulate Y by a recurrence relation.')


    def simulate_Y_recurrence_fixed_start(self):
        """
        Simulate synthetic data Y of shape N x T_total
        by a given recurrence relation,
        starting from a fixed value at time zero.
        """
        if self.kwargs.get('A_model') in ['exp-decay', 'VARMA']:
            self.dist = self.kwargs.get('dist', 'Gaussian')
            self.seed = self.kwargs.get('seed')
            self.Y0 = self.kwargs.get('Y0')
            assert len(self.Y0) == self.N

            Y = np.zeros(shape = (self.N, self.T_total))
            Y[:, 0] = self.Y0

            rng = np.random.default_rng(
                seed = self.seed,
            )
            if self.dist == 'Gaussian':
                eps = self.population_covariance.sqrt_C @ rng.standard_normal(
                    size = (self.N, self.T_total),
                )
            elif self.dist == 'Student-t':
                df = self.kwargs.get('df')
                eps = self.population_covariance.sqrt_C @ rng.standard_t(
                    df = df,
                    size = (self.N, self.T_total),
                )
            else:
                raise Exception('Unknown distribution.')

            for t in range(1, self.T_total):
                eps_restr = eps[:, max(t - self.auto_covariance.r2, 0):(t + 1)]
                a_list_rev_restr = self.auto_covariance.a_list[::-1][:eps_restr.shape[1]]
                
                Y_restr = Y[:, max(t - self.auto_covariance.r1, 0):t]
                b_list_rev_restr = self.auto_covariance.b_list[::-1][:Y_restr.shape[1]]
                
                Y[:, t] = (a_list_rev_restr * eps_restr).sum(axis = 1) + (b_list_rev_restr * Y_restr).sum(axis = 1)
            
            self.Y = Y
        
        else:
            raise Exception('Unknown model to simulate Y by a recurrence relation.')