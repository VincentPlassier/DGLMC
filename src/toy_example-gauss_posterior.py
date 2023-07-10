import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import os
from functools import partial
from ula_sampler import ULA
from sgld_sampler import SGLD
from gibbs_sampler import GibbsSampler
from kernel_kde import KDE
from effective_sample_size import ESS
from diagnostic_test import KSD

import time
year, month, day, hour, min = map(str, time.strftime("%Y %m %d %H %M").split())
date_ = '-{}-{}-{}-{}h{}min'.format(year, month, day, hour, min)

## Generate the dataset
# set random seed for reproducibility
np.random.seed(20)
# defined the dimension
dim = 2
# set the number of workers
num_workers = 20
# define the number of samples
n_samples = 10000
# define the center of the distribution
mu = np.zeros(dim)
#
sigma0 = .5 * np.eye(dim) + .5 * np.ones((dim, dim))
sigma0_inv = np.linalg.inv(sigma0)
#
sigma1 = 1.5 * np.eye(dim) - .5 * np.ones((dim, dim))
sigma1_inv = np.linalg.inv(sigma1)
# X contains the Gaussian samples
X = ss.multivariate_normal.rvs(mean=mu, cov=sigma1, size=n_samples)
#
X_split = np.array_split(X, num_workers)

## Define the Gaussian posterior class


class GaussianPosterior:

    def __init__(self, X, mu, sigma0, sigma1):
        self.X = X
        self.mu = mu
        self.sigma0_inv, self.sigma1_inv = np.linalg.inv(sigma0), np.linalg.inv(sigma1)

    def sigma_posterior(self):
        self.sigma_post = np.linalg.inv(self.sigma0_inv + len(self.X) * self.sigma1_inv)

    def mu_posterior(self):
        self.mu_post = self.sigma_post.dot(self.sigma1_inv.dot(self.X.sum(axis=0)) + self.sigma0_inv.dot(self.mu))


## Define a fonction to display the contour of Gaussian densities
def gauss_draw(mu, sigma, low = -1, high = 1):
    # compute a space grid
    Zx, Zy = np.meshgrid(np.linspace(low, high, num=100), np.linspace(low, high, num=100))
    Z = np.stack((Zx, Zy)).T.reshape(-1, 2)
    # define the considered pdf
    def pdf(x): return ss.multivariate_normal.pdf(x, mean=mu, cov=sigma)
    # Y contains the values of the density calculated
    Y = np.fromiter(map(pdf, Z), dtype='float').reshape(Zx.shape).T
    # plot the contours
    CS = plt.contour(Zx, Zy, Y, levels=1, linestyles='dashed')
    # plt.clabel(CS, inline=1, fontsize=4)
    # display the center
    plt.plot(mu[0], mu[1], '*', color='cyan', markersize=6)

## Draw the contours of the Gaussians and determine the posterior laws
# we stock the gradients in grad_U
grad_U = []
def grad_u(X, x):
    return np.dot(sigma0_inv, x - mu) / num_workers + sigma1_inv.dot(len(Xi) * x - X.sum(axis=0))
#
for Xi in X_split:
    grad_U.append(partial(grad_u, Xi))
    gauss_post = GaussianPosterior(Xi, mu, num_workers * sigma0, sigma1)
    gauss_post.sigma_posterior()
    gauss_post.mu_posterior()
    sigma_post = gauss_post.sigma_post
    mu_post = gauss_post.mu_post
    gauss_draw(mu_post, sigma_post, low=-.4, high=.4)

## Global Parameters
# initializes the regressor
theta0 = np.zeros(2)
# mcmc iterations
mc_iter = 20000
# burn-in period
T_burn_in = 2000

## Set the SGLD Sampler
# define the lipschitz constant of Ui
L = 1 / np.linalg.eigvalsh(sigma0)[0] + len(X) / np.linalg.eigvalsh(sigma1)[0]
# define the SGLD sampler
sgld = SGLD(grad_U, tau=1/L, batch_size=1)
# define the list containing the iterates
theta = theta0
theta_sgld = [theta]
# launch the sampler
for i in range(mc_iter):
    theta = sgld.step(theta)
    theta_sgld.append(theta)
theta_sgld = np.squeeze(theta_sgld)[T_burn_in:]

## Set the Gibbs Sampler
# defined the number of iterations executed by each worker
N = 1 * np.ones(num_workers, dtype=int)  # np.random.randint(1, 4, num_workers)
# define the lipschitz constant of Ui
L = 1 / np.linalg.eigvalsh(sigma0)[0] + len(X) / np.linalg.eigvalsh(sigma1)[0]
# defined the regularization parameters rho
rho = .01 * np.ones(num_workers)  # / L
gamma = np.zeros_like(rho)
# initialized the parameters
A = []
z0 = []
for i in range(num_workers):
    A.append(np.eye(dim))
    z0.append(np.zeros(dim))
    gamma[i] = 1 / (1 / rho[i] + L)  # rho[i] / N[i]
# defined the Gibbs sampler
GS = GibbsSampler(ULA, grad_U, A, theta0, z0, N, gamma, rho)
# define the list containing the iterates
theta = theta0
theta_GS = [theta]
# launch the sampler
for i in range(mc_iter):
    GS.step()
    theta_GS.append(GS.theta)
theta_GS = np.squeeze(theta_GS)[T_burn_in:]
# display the contour of the kernel density estimated from the Gibbs Sampler
# sns.kdeplot(x=theta_GS[:, 0], y=theta_GS[:, 1], fill=True)
sns.histplot(x=theta_GS[:, 0], y=theta_GS[:, 1], kde=True)
# kde = KDE(data=theta_GS, sigma=.1)
# kde.smooth_grid(num=50, levels=1)

## Displays the results
# displays the samples obtained with SGLD
# plt.plot(theta_sgld[:, 0], theta_sgld[:, 1], '.', color='blue', markersize=2, label='SGLD')
# displays the samples obtained with the Gibbs Sampler
# plt.plot(theta_GS[:, 0], theta_GS[:, 1], '*', color='g', markersize=2, label='Gibbs Sampler')
# display the samples
# xmin, xmax, ymin, ymax = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
# plt.axis([xmin - .5, xmax + .5, ymin - .5, ymax + .5])
# plt.plot(X[:, 0], X[:, 1], '*', markersize=4)
# create a grey background
# greys = np.full((100, 100, 3), 200, dtype=np.uint8)
# plt.imshow(greys)
# display the mean mu estimated with the Gibbs Sample
mu_GS = np.mean(theta_GS, axis=0)
plt.plot(mu_GS[0], mu_GS[1], '.', color='red', markersize=12, label='mu estimate')
# display the true mu
plt.plot(mu[0], mu[1], '*', color='black', markersize=12, label='mu true')
# display the contour
# plt.grid('True')
plt.axis('off')
plt.autoscale()
plt.title('N={0:.1E}, gamma={1:.1E}, rho={2:.1E}'.format(N[0], gamma[0], rho[0]))
plt.legend()
plt.savefig('../figures/' + os.path.basename(__file__)[:-3] + date_ \
            + '-N={0:.1E}, gamma={1:.1E}, rho={2:.1E}'.format(N[0], gamma[0], rho[0]) + '.png')
plt.show()
