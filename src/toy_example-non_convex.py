import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as ss
import seaborn as sns
import os
from functools import partial
from ula_sampler import ULA
from gibbs_sampler import GibbsSampler

import time
year, month, day, hour, min = map(str, time.strftime("%Y %m %d %H %M").split())
date_ = ''  # -{}-{}-{}-{}h{}min'.format(year, month, day, hour, min)

## Define the main parameters
# fix the seed for reproductibility
np.random.seed(30)
# defined the dimension
dim = 1
# defined the number of iterations
mc_iter = 40000
# burn-in period
T_burn_in = 20000
# set the number of workers
num_workers = 2
# defined the number of iterations executed by each worker
N = 1 * np.ones(num_workers, dtype=int)

## Define the Gaussian Mixture
# define the number of samples
n_samples = 100
# define the number of clusters for the Gaussian Mixture
num_clusters = 2
# define the angular
rho = 2 * np.pi / num_clusters * np.arange(num_clusters)
# define the barycenters of the clusters
barycenters = 1 * np.cos(rho)
# define the weights
w = np.ones(num_clusters) + np.random.uniform(size=num_clusters)
w /= w.sum()
# X will contain the Gaussian Mixture samples
X = np.zeros((n_samples, dim))
# Sigma will contain the covariances
Sigma = []
for i in range(num_clusters):
    cov = np.random.uniform(.1, .8)
    Sigma.append(cov)
# generate some samples
ind = np.random.choice(np.arange(num_clusters), size=n_samples, p=w)
for (i, j) in enumerate(ind):
    X[i] = ss.multivariate_normal.rvs(mean=barycenters[j], cov=Sigma[j])
# define the Gaussian Mixture density
def pdf_gm(x):
    return np.sum([w[i] * ss.multivariate_normal.pdf(x, barycenters[i], Sigma[i]) for i in range(num_clusters)])

## Set the Gibbs Sampler
# initialized the parameters
theta0 = np.zeros(dim)
z0 = []
# defined the matrix A = [A_1,...,A_b]
A = []
# set the Gibbs Sampler parameters
rho = .04 * np.ones(num_workers)
gamma = np.zeros_like(rho)
for i in range(num_workers):
    z0.append(np.zeros(dim))
    A.append(np.identity(dim))
    gamma[i] = 1. * rho[i] / N[i]

# define the first gradient
w1, mu1, sigma1_inv = w[0], barycenters[0], 1 / Sigma[0]
U1 = lambda x: np.dot(x - mu1, np.dot(sigma1_inv, x - mu1))
grad_U1 = lambda x: np.dot(sigma1_inv, x - mu1)
# define the second gradient
w2, mu2, sigma2_inv = w[1], barycenters[1], 1 / Sigma[1]
V = lambda x: np.dot(x - mu2, np.dot(sigma2_inv, x - mu2))
grad_V = lambda x: np.dot(sigma2_inv, x - mu2)
# we compute grad_V for SGLD
grad_U2 = lambda x: (grad_V(x) - grad_U1(x)) / (1 + w1 / w2 * np.exp(V(x) - U1(x)))
# we stock the gradients in grad_U
grad_U = [grad_U1, grad_U2]

# defined the Gibbs sampler
GS = GibbsSampler(ULA, grad_U, A, theta0, z0, N, gamma, rho)
# theta_list stored the iterates
theta_GS = []
z_list = []
for i in range(mc_iter):
    GS.step()
    theta_GS.append(GS.theta)
    z_list.append(GS.z)
theta_GS = np.squeeze(theta_GS)[T_burn_in:]
z_list = np.squeeze(z_list)[T_burn_in:].T

## Displays the results
fig = plt.figure()
# display the Gaussian Mixture density
xx = np.linspace(-5, 5, num=200)
yy = np.fromiter(map(pdf_gm, xx), dtype=float)
plt.axis([-4, 4, 0, .5])
plt.plot(xx, yy, label='ground truth')
# displays the mcmc samples
sns.kdeplot(theta_GS, fill=True, common_norm=False, color='orange')
# animation function
figure_GS = plt.plot([0], [0], [], [], 'ro', markersize=10, label='theta')
def init():
    figure_GS[1].set_data([], [])
    return (figure_GS)
def animate(list, i):
    figure_GS[1].set_data(list[i], pdf_gm(list[i]))
    return (figure_GS)
# calling the animation function
animate_gs = partial(animate, theta_GS)
anim = animation.FuncAnimation(fig, animate_gs, init_func=init, frames=500, interval=30, blit=True, repeat=False)
# for z in z_list:
#     animate_z = partial(animate, z)
#     animation.FuncAnimation(fig, animate_z, init_func=init, frames=500, interval=30, blit=True, repeat=False)
# plt.hist(theta_GS, mc_iter // 60, density=True)
plt.grid('True')
plt.title('N={0:.1E}, gamma={1:.1E}, rho={2:.1E}'.format(N[0], gamma[0], rho[0]))
plt.savefig('../figures/' + os.path.basename(__file__)[:-3] + date_ \
            + '-N={0:.1E}, gamma={1:.1E}, rho={2:.1E}'.format(N[0], gamma[0], rho[0]) + '.png')
plt.legend()
