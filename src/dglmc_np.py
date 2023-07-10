import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from ula_sampler import Ula
from mala_sampler import MALA


class Dglmc:

    def __init__(self, MCMC, grad_U, theta0, z0, N, gamma, rho):
        self.MCMC, self.grad_U = MCMC, grad_U
        self.theta, self.z = theta0, z0
        self.gamma, self.rho, self.N = gamma, rho, N
        self.Q_inv = 1 / np.sum([1 / rho_i for rho_i in rho]) * np.eye(len(theta0))

    def z_update(self):
        for i, n in enumerate(self.N):
            V = lambda x: - (self.grad_U[i](x) + (x - self.theta) / self.rho[i])
            mcmc_update = self.MCMC(self.gamma[i])
            mcmc_update.log_grad = V
            for k in range(n):
                self.z[i] = mcmc_update.step(self.z[i])

    def theta_update(self):
        mu = self.get_mu()
        self.theta = ss.multivariate_normal.rvs(mean=mu, cov=self.Q_inv)

    def step(self):
        self.z_update()
        self.theta_update()

    def get_mu(self):
        return np.dot(self.Q_inv, np.sum([z_i / self.rho[i] for i, z_i in enumerate(self.z)], axis=0))


class Gaussian:

    def __init__(self, mu, cov):
        self.mu, self.cov = mu, cov
        self.cov_inv = np.linalg.inv(cov)
        self.norm_cst = (2 * np.pi * np.linalg.det(cov)) ** (len(mu) / 2)

    def __call__(self, theta):
        theta_mu = (theta - self.mu)
        return np.exp(- theta_mu.dot(self.cov_inv).dot(theta_mu) / 2) / self.norm_cst

    def minus_grad_log(self, theta):
        return np.dot(self.cov_inv, theta - self.mu)


if __name__ == '__main__':
    # fix the seed for reproductibility
    np.random.seed(20)
    # define the dimension
    dim = 1
    #
    mu, cov = np.zeros(dim), np.identity(dim)
    # set the number of workers
    num_workers = 4
    #
    gauss = Gaussian(mu, num_workers * cov)
    # initialize the parameters
    theta0 = np.zeros(dim)
    #
    z0 = list()
    # define the number of iterations executed by each worker
    N = np.random.randint(1, 2, num_workers)
    # contain the potentials of each workers
    grad_U = list()
    # contain the tolerance and the step size parameters of each worker
    rho = .5 * np.ones(num_workers)
    gamma = np.zeros_like(rho)
    #
    for i in range(num_workers):
        z0.append(np.zeros(dim))
        grad_U.append(gauss.minus_grad_log)
        gamma[i] = 1 / (N[i] * (1 / rho[i] + 1 / (np.linalg.eigvalsh(cov)[0] * num_workers)))
    # define the Gibbs sampler
    GS = Dglmc(Ula, grad_U, theta0, z0, N, gamma, rho)
    # define the number of iterations
    mc_iter = 2000
    # theta_list stored the iterates
    theta_list = []
    for i in range(mc_iter):
        GS.step()
        theta_list.append(GS.theta)
    # Display the results
    plt.hist(theta_list, 50, density=True)
    X = np.linspace(-3, 3, 500)
    plt.plot(X, [ss.multivariate_normal.pdf(x, mean=mu, cov=cov) for x in X])
    plt.grid('True')
    plt.show()
