import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
import time
import argparse
from tqdm import tqdm
from functools import partial
from dglmc_sampler import Dglmc
from fald_sampler import Fald
from ula_sampler import Ula

year, month, day, hour, min = map(str, time.strftime("%Y %m %d %H %M").split())
date = '-{}-{}-{}-{}h{}min'.format(year, month, day, hour, min)

## To run on the clusters
# Save the user choice of settings
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
parser.add_argument('-m', '--mc_iter', default=5, type=int, help='set the number of Monte Carlo iterations')
parser.add_argument('-t', '--t_burn_in', default=-1, type=int, help='set the number of burning steps')
parser.add_argument('-N', '--N', default=1, type=int, help='set the number of local updates before communication')
parser.add_argument('-e', '--eta', default=.01, type=float, help='set the Fald time step')
parser.add_argument('-r', '--rho', default=.1, type=float, help='set the Dglmc tolerance')
parser.add_argument('-g', '--gamma', default=2.5 * 1e-4, type=float, help='set the Dglmc time step')
args = parser.parse_args()

# Define the file title
title = '-seed={0}-mc_iter={1}-t_burn_in={2}-N={3}-eta={4:.1E}-rho={5:.1E}-gamma={6:.1E}'.format(args.seed,
                                                                                                 args.mc_iter,
                                                                                                 args.t_burn_in, args.N,
                                                                                                 args.eta, args.rho,
                                                                                                 args.gamma)
title = title.replace(".", "_")  # to load the image on LaTex
# Path to save the results
path_workdir = '/workdir/plassierv/21-fl_comparison'
path_figures = path_workdir + '/figures/' + os.path.basename(__file__)[:-3] + title
path_variables = path_workdir + '/variables/' + os.path.basename(__file__)[:-3] + title

# Create the directory if it does not exist
os.makedirs(path_workdir + '/figures', exist_ok=True)
os.makedirs(path_workdir + '/variables', exist_ok=True)

# Start the timer
startTime = time.time()

## Generate the dataset

# set random seed for reproducibility
np.random.seed(20)
# defined the dimension
dim = 2
# set the number of workers
num_workers = 10
# define the number of samples
n_samples = 1000
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
# Homogeneous dataset
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
        self.mu_post = self.sigma_post.dot(self.sigma1_inv.dot(np.sum(self.X, axis=0)) + self.sigma0_inv.dot(self.mu))


## Define a fonction to display the contour of Gaussian densities
def gauss_draw(mu, sigma, low = -1, high = 1, levels = 1, colors = 'grey', color_mu = 'cyan', linestyles = 'solid'):
    # compute a space grid
    Zx, Zy = np.meshgrid(np.linspace(low, high, num=100), np.linspace(low, high, num=100))
    Z = np.stack((Zx, Zy)).T.reshape(-1, 2)

    # define the considered pdf
    def pdf(x): return ss.multivariate_normal.pdf(x, mean=mu, cov=sigma)

    # Y contains the values of the density calculated
    Y = np.fromiter(map(pdf, Z), dtype='float').reshape(Zx.shape).T
    # plot the contours
    CS = plt.contour(Zx, Zy, Y, levels, colors=colors, linestyles=linestyles)
    # plt.clabel(CS, inline=1, fontsize=4)
    # display the center
    plt.plot(mu[0], mu[1], '*', color=color_mu, markersize=6)


## Draw the contours of the Gaussians and determine the posterior laws

# we stock the gradients in grad_U
grad_U = list()


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
    gauss_draw(mu_post, sigma_post, low=-.6, high=.6, linestyles='dashed')

## Draw the contours of the true posterior
gauss_post = GaussianPosterior(X, mu, sigma0, sigma1)
gauss_post.sigma_posterior()
gauss_post.mu_posterior()
sigma_post = gauss_post.sigma_post
mu_post = gauss_post.mu_post
gauss_draw(mu_post, sigma_post, low=-.6, high=.6, levels=3, colors='red', color_mu='khaki')

## Global Parameters
# initialize the regressor
theta0 = np.zeros(dim)
# mcmc iterations
mc_iter = args.mc_iter
# burn-in period
t_burn_in = args.t_burn_in if args.t_burn_in != -1 else args.mc_iter // 10

## Set the DG-LMC Sampler
N = args.N * np.ones(num_workers, dtype=int)
# define the lipschitz constant of U
L = 1 / np.linalg.eigvalsh(sigma0)[0] + n_samples / np.linalg.eigvalsh(sigma1)[0]
print("1 / L = %s" % (1 / L))
# define the tolerance parameter
rho = args.rho * np.ones(num_workers)  # rho = np.ones(num_workers) / (5 * L)
# define the step size parameter
gamma = np.zeros(num_workers)
for i in range(num_workers):
    gamma[i] = args.gamma  # 1 / (N[i] * (1 / rho[i] + L))
#
z0 = list()
for i in range(num_workers):
    z0.append(np.zeros(dim))
# define the DG-LMC sampler
dglmc = Dglmc(Ula, grad_U, theta0, z0, N, gamma, rho)
# define the list containing the iterates
theta = theta0
theta_dglmc = [theta]
# launch the sampler
for i in tqdm(range(mc_iter)):
    dglmc.step()
    theta_dglmc.append(dglmc.theta)
theta_dglmc = np.squeeze(theta_dglmc)[t_burn_in:]
# display the contour of the kernel density estimated from the SGLD sampler
# sns.kdeplot(x=theta_dglmc[:, 0], y=theta_dglmc[:, 1], fill=True)
# kde = KDE(data=theta_dglmc, sigma=.1)
# kde.smooth_grid(num=80)
plt.hist2d(theta_dglmc[:, 0], theta_dglmc[:, 1], bins=20, cmap='Blues')

# Displays the DG-LMC results
plt.axis('off')
plt.autoscale()
plt.title('DgLmc' + title)
plt.savefig(path_figures + '-DgLmc.pdf', bbox_inches='tight')
plt.show()

## Set the Fald Sampler
N = args.N * np.ones(num_workers, dtype=int)
# define the stepsize parameter
eta = args.eta  # 1 / L
# set the temperature
tau = 1
# define the Fald sampler
fald = Fald(grad_U, theta0, N, eta, tau)
# define the list containing the iterates
theta = theta0
theta_fald = [theta]
# launch the sampler
for i in tqdm(range(mc_iter)):
    fald.step()
    theta_fald.append(fald.theta)
theta_fald = np.squeeze(theta_fald)[t_burn_in:]
# display the contour of the kernel density estimated from the Fald sampler
plt.hist2d(theta_fald[:, 0], theta_fald[:, 1], bins=20, cmap='Blues')

## Displays the Fald results
plt.axis('off')
plt.autoscale()
plt.title('Fald' + title)
plt.savefig(path_figures + '-Fald.pdf', bbox_inches='tight')
plt.show()

## Compute the auto-correlation function
# Estimate the mean and the variance parameters for DgLmc
mean_dglmc = np.mean(theta_dglmc, axis=0)
theta_dglmc_center = theta_dglmc - mean_dglmc
var_dglmc = np.mean(theta_dglmc_center ** 2)
# Estimate the mean and the variance parameters for Fald
mean_fald = np.mean(theta_fald, axis=0)
theta_fald_center = theta_fald - mean_fald
var_fald = np.mean(theta_fald_center ** 2)
# Compute the autocorrelations
autocorrelation_dglmc = np.ones(mc_iter - t_burn_in + 1)
autocorrelation_fald = np.ones(mc_iter - t_burn_in + 1)
for t in range(1, mc_iter - t_burn_in + 1):
    autocorrelation_dglmc[t] = np.mean(theta_dglmc_center[t:] * theta_dglmc_center[:-t]) / var_dglmc
    autocorrelation_fald[t] = np.mean(theta_fald_center[t:] * theta_fald_center[:-t]) / var_fald
# Print a score on the autocorrelation coefficients, lower the better
print("Fald:", eta, np.abs(autocorrelation_fald).mean())
print("DG-LMC:", gamma[0], np.abs(autocorrelation_dglmc).mean())
# Display the autocorrelations
plt.axis('on')
plt.grid('True')
plt.plot(autocorrelation_dglmc, label='DG-LMC')
plt.plot(autocorrelation_fald, label='FALD')
plt.xlabel('Iteration')
plt.ylabel('Autocorrelation')
# plt.title('Autocorrelation comparison' + title)
plt.legend()
plt.savefig(path_figures + '-autocorrelation.pdf', bbox_inches='tight')
plt.show()

# Compute the execution time
executionTime = time.time() - startTime
# Save important statistics
file = open(path_figures + '.txt', 'a')
file.write(
    f"\nDate = {date}, \nTime = {executionTime}, \nmc_iter = {mc_iter}, \nN = {N}, \neta = {eta},"
    f" \nrho = {rho}, \ngamma = {gamma}")
file.close()  # to change the file access mode
