#!/usr/bin/env python
# coding: utf-8

import os
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.tools_dl import accuracy_model, client_solver


def schedule(epoch, rho=5e-5, gamma=5e-6):  # todo: to del
    if epoch == 0:
        rho *= 2 ** 6
        gamma *= 2 ** 6
        print('rho, gamma =', rho, gamma)
    elif epoch == 100:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    elif epoch == 300:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    elif epoch == 500:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    elif epoch == 600:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    elif epoch == 800:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    elif epoch == 1000:
        rho *= 1 / 2
        gamma *= 1 / 2
        print('rho, gamma =', rho, gamma)
    return rho, gamma


def init_net_clients(client_data, net_fn, param_optim, load_init = True, save_init = True, path_fn = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    # list of the neural networks on each worker
    net_clients = []
    print("path_fn", path_fn)
    for i, data in tqdm(enumerate(client_data)):
        net_client = copy.deepcopy(net_fn).to(device)
        path_client = os.path.join(path_fn, f'client_{i}')
        if load_init and path_fn is not None and os.path.exists(path_client):
            # load the model of the client
            net_client.load_state_dict(torch.load(path_client))
        else:
            # perform some SGD steps
            model_state_dict = client_solver(data, net_client, **param_optim)
            if save_init and path_fn is not None:
                # store the model of the client
                torch.save(model_state_dict, path_client)
        # store the neural network
        net_clients.append(copy.deepcopy(net_client).to(device))
    return net_clients


class Dglmc:

    def __init__(self, trainloader_init, net_fn, param_optim, tolerance_params, load_init = True, save_init = True,
                 path_fn = None):
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Define the harmonic average of the tolerance parameters
        self.tolerance_params_hm = 1 / torch.sum(1 / tolerance_params).to(self.device)
        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()  # reduction='sum'
        # Load the saved networks
        self.net_fn = net_fn.to(self.device)
        # Copy the deep neural network on each worker
        self.net_clients = init_net_clients(trainloader_init, net_fn, param_optim, load_init, save_init, path_fn)
        # Store the statistics
        self.save_dict = {"accuracies_test": [], "mse_relative": []}

    def net_clients_update(self, trainloader, weight_decay, num_local_updates, tolerance_params, step_sizes,
                           batch_sizes):
        running_loss = 0.
        correct = 0
        total = 0
        for it, (inputs, targets) in enumerate(trainloader):
            inputs = torch.transpose(inputs, dim0=0, dim1=1)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            for i, (net, batch) in enumerate(zip(self.net_clients, inputs)):
                # the i-th worker do N_i local updates
                if it >= num_local_updates[i]:
                    continue
                net.zero_grad()
                outputs = net(batch).to(self.device)  # pk net en couleur ?
                loss = self.criterion(outputs, targets[:, i])
                for param in net.parameters():
                    loss += weight_decay / self.num_data * torch.norm(param) ** 2
                # compute the gradient of loss with respect to all Tensors with requires_grad=True
                loss.backward()
                running_loss += loss.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += targets[:, i].size(0)
                correct += (predicted == targets[:, i]).sum().item()
                # disable gradient calculation to reduce memory consumption
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        # perform the ULA step
                        param.data.add_(step_sizes[i] / tolerance_params[i] * (
                                self.net_fn.state_dict()[name] - param) - self.num_data * step_sizes[
                                            i] * param.grad.data + torch.sqrt(2 * step_sizes[i]) * torch.randn(
                            param.shape).to(self.device))
        print('--- Train --- Accuracy = %.3f, Loss = %.3f' % (100 * correct / total, running_loss))

    @torch.no_grad()
    def net_fn_update(self, tolerance_params):
        # Parameter updates
        for name, param_fn in self.net_fn.named_parameters():
            mu = torch.zeros_like(param_fn)
            for i, net in enumerate(self.net_clients):
                param_z = dict(net.named_parameters())[name]
                mu += param_z / tolerance_params[i]
            param_fn.data.copy_(self.tolerance_params_hm * mu + torch.sqrt(self.tolerance_params_hm) * torch.randn(param_fn.shape).to(self.device))

    def save_results(self, testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples):
        # add the new predictions with the previous ones
        if epoch >= t_burn_in and (
                epoch - t_burn_in) % thinning == 0 and save_samples and path_save_samples is not None:
            # calculate some statistics
            acc_test = accuracy_model(self.net_fn, testloader, self.device)
            torch.save(self.net_fn.state_dict(), path_save_samples + '/%s' % epoch)
            # Save the parameters of each worker
            for i, net_client in enumerate(self.net_clients):
                path_client = os.path.join(path_save_samples, f'client_{i}')
                # store the model of the client
                torch.save(net_client.state_dict(), path_client)
            # save the accuracy
            self.save_dict["accuracies_test"].append(acc_test)
            # print the statistics
            print("--- Test --- Epoch: {}, Test accuracy: {}\n".format(epoch + 1, acc_test))

    def run(self, trainloader, testloader, num_iter, weight_decay = 5, num_local_updates = np.ones(10, dtype=int),
            tolerance_params = .02 * torch.ones(10), step_sizes = .005 * torch.ones(10),
            batch_sizes = 64 * np.ones(10, dtype=int), t_burn_in = 0, thinning = 1, epoch_init = -1,
            save_samples = False, path_save_samples = None):
        self.num_data = sum([len(data[1]) for data in trainloader])
        print('\nnum_data =', self.num_data)
        for epoch in tqdm(range(epoch_init + 1, epoch_init + 1 + num_iter)):
            self.net_clients_update(trainloader, weight_decay, num_local_updates, tolerance_params, step_sizes,
                                    batch_sizes)
            self.net_fn_update(tolerance_params)
            self.save_results(testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples)
        return self.net_fn.state_dict(), self.save_dict
