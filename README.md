# DG-LMC: A turn-key and scalable synchronous distributed MCMC algorithm via Langevin Monte Carlo within Gibbs.

This repository contains the code to reproduce the experiments in the paper *DG-LMC: A turn-key and scalable synchronous distributed MCMC algorithm via Langevin Monte Carlo within Gibbs* by Vincent Plassier, Maxime Vono, Alain Durmus and Eric Moulines.

## Requirements

We use provide a `requirements.txt` file that can be used to create a conda
environment to run the code in this repo:
```bash
$ conda create --name <env> --file requirements.txt
```

Example set-up using `pip`:
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Abstract

Performing reliable Bayesian inference on a big data scale is becoming a keystone in the modern era of machine learning. A workhorse class of methods to achieve this task are Markov chain Monte Carlo (MCMC) algorithms and their design to handle distributed datasets has been the subject of many works. However, existing methods are not completely either reliable or computationally efficient. In this paper, we propose to fill this gap in the case where the dataset is partitioned and stored on computing nodes within a cluster under a master/slaves architecture. We derive a user-friendly centralized distributed MCMC algorithm with provable scaling in high-dimensional settings. We illustrate the relevance of the proposed methodology on both synthetic and real data experiments. [Paper](http://proceedings.mlr.press/v139/plassier21a/plassier21a.pdf)
