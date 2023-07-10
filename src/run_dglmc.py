#!/usr/bin/env python
# coding: utf-8

import os
import copy
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from dglmc_dl import Dglmc
from utils import load_dataset
from models.lenet5 import LeNet5
from pytorchcv.model_provider import get_model
from torchvision import transforms
from distutils.util import strtobool
from torch._C import default_generator
from torch.utils.data import DataLoader, TensorDataset
from utils.tools_dl import predictions
from utils.metrics import agreement, total_variation_distance
from utils.generate_imbalanced_dataset import imbalanced_dataset
from utils.uncertainties_tools import PostNet, confidence, ECE, BS, Predictive_entropy, AUC, accuracy_confidence, \
    calibration_curve

# Save the user choice of settings
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet20_frn_swish", choices=["resnet20_frn_swish"],
                    help='set the model name')
parser.add_argument('-n', '--num_iter', default=50, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "SVHN"],
                    help='set the dataset')
parser.add_argument('-i', '--imbalance', default='False', help='if True we generate imbalanced datasets')
parser.add_argument('--proportion', default=None, type=float, help='set the imbalanced parameter')
parser.add_argument('-l', '--num_local_updates', default=None, type=int, help='set the number of local updates')
parser.add_argument('-r', '--tolerance_params', default=1e-04, type=float, help='set the tolerance parameter')
parser.add_argument('-g', '--step_sizes', default=1e-05, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=64, type=int, help='set the mini-batch size')
parser.add_argument('-w', '--weight_decay', default=5, type=float, help='set the l2 regularization parameter')
parser.add_argument('-b', '--t_burn_in', default=0, type=int, help='set the burn in period')
parser.add_argument('-t', '--thinning', default=1, type=int, help='set the thinning')
parser.add_argument('--save_samples', default='True', help='if True we save the samples')
parser.add_argument('--ngpu', default=1, type=int, help='setl the number of gpus')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# Define the title to save the results
title = str()
for key, value in {'d_': args.dataset_name, 'proportion_': args.proportion, 'l_': args.num_local_updates,
                   'r_': args.tolerance_params, 'g_': args.step_sizes}.items():  # 'n_': args.num_iter,
    title += '-' + key + str(value)

# Print the local torch version
print(f"Torch Version {torch.__version__}")

print("os.path.abspath(__file__) =\n\t", os.path.abspath(__file__))
path = os.path.abspath('..')

# Save the path to store the data
if '/gpfs/users/plassierv' in path:
    path_workdir = '/workdir/plassierv/fald_results'
else:
    path_workdir = '/home/cloud/workdir/fald_results'
path_dataset = path_workdir + '/../dataset'
path_figures = path_workdir + '/figures'
path_variables = path_workdir + '/variables'
path_stats = path_variables + '/dglmc' + title
path_txt = path_variables + "/dglmc_text" + title + '.txt'
path_save_samples = path_variables + '/samples-dglmc' + title

# Create the directory if it does not exist
save_samples = strtobool(args.save_samples)
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)
if save_samples:
    os.makedirs(path_save_samples, exist_ok=True)

# Set random seed for reproducibility
seed_np = args.seed if args.seed != -1 else None
seed_torch = args.seed if args.seed != -1 else default_generator.seed()
np.random.seed(seed_np)
torch.manual_seed(seed_torch)
torch.cuda.manual_seed(seed_torch)

# Start the timer
startTime = time.time()

# Load the function associated with the chosen dataset
dataset = getattr(torchvision.datasets, args.dataset_name)

# Define the transformation
normalize = [0.1307, 0.3081] if args.dataset_name == 'MNIST' else [(0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010)]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalize),
])

# Number of worker to distribute the data
num_workers = 10

# Define the parameter of the dataset
params_train = {"root": path_dataset, "train": True, "transform": transform, "download": True}
params_test = {"root": path_dataset, "train": False, "transform": transform}

# Define the datasets
trainset = dataset(**params_train)
split_size = len(trainset) // num_workers + (1 if len(trainset) % num_workers != 0 else 0)
trainloader_init = DataLoader(trainset, split_size, shuffle=False)
if strtobool(args.imbalance):
    print("\n--- Generate imbalanced datasets ---\n")
    trainloader_init = imbalanced_dataset(trainloader_init, num_workers, args.proportion)

# Mofidy the shape of the data to save time during the training stage
inputs = None
targets = None
for data in trainloader_init:
    x = torch.unsqueeze(data[0], dim=1)
    y = torch.unsqueeze(data[1], dim=1)
    if inputs is None:
        length = len(y)
        inputs, targets = x, y
    else:
        length = min(length, len(y))
        inputs = torch.cat((inputs[:length], x[:length]), dim=1)
        targets = torch.cat((targets[:length], y[:length]), dim=1)

# Define the loader
trainset = TensorDataset(inputs, targets)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True)

# Define the testloader
batch_size_test = 500
testset = dataset(**params_test)
testloader = DataLoader(testset, batch_size_test, shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Load the saved networks
pretrained = True
net = LeNet5() if args.dataset_name == 'MNIST' else get_model(args.model, pretrained=pretrained)

# We start from the last sample
last_checkpoint = True
epoch_init = -1
t_burn_in = args.t_burn_in
if last_checkpoint:
    fmax = -1
    for f in os.listdir(path_save_samples):
        if 'client' in f:
            continue
        if int(f) > fmax:
            fmax = int(f)
    if fmax > -1:
        epoch_init = fmax
        t_burn_in = args.thinning
        print(f'--- Load the last checkpoint: {fmax} ---\n ')
        net.load_state_dict(torch.load(os.path.join(path_save_samples, str(fmax))))

# Handle multi-gpu if desired
net.to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    net = nn.DataParallel(net, list(range(args.ngpu)))

# Define the number of update on each worker
num_local_up = np.infty if args.num_local_updates is None else args.num_local_updates
num_local_updates = num_local_up * torch.randint(low=1, high=2, size=(num_workers,)).to(device)

# Define the tolerance parameter
tolerance_params = args.tolerance_params * torch.ones(num_workers).to(device)

# Define the step size parameter
step_sizes = args.step_sizes * torch.ones(num_workers).to(device)

# Define the mini-batch size
batch_sizes = args.batch_size * np.ones(num_workers, dtype=int)

# Define the optimizer used for the warm-start
param_optim = {'num_epochs': 10, 'batch_size': 64, 'lr': .01, 'momentum_decay': 0.9, 'weight_decay': 0.}  # todo : 250

# Define the DGLMC sampler
model = Dglmc(trainloader_init, net, param_optim, tolerance_params, load_init=True,
              path_fn=path_save_samples)  # todo: change here

# Performs num_iter iterations of DGLMC
model_state_dict, save_stats = model.run(trainloader, testloader, args.num_iter, args.weight_decay / num_workers,
                                         num_local_updates, tolerance_params, step_sizes, batch_sizes,
                                         args.t_burn_in, args.thinning, epoch_init, save_samples,
                                         path_save_samples=path_save_samples)

# Define the new testloader
del trainset, testset, trainloader, testloader
if args.dataset_name == 'CIFAR10':
    dataset = getattr(load_dataset, "load_" + args.dataset_name)

# Load the datasets
batch_size_test = 500
testset = dataset(**params_test)
testloader = DataLoader(testset, batch_size_test, shuffle=False)

# Store the targets
targets = None
for inputs, labels in testloader:
    if targets is None:
        targets = labels
    else:
        targets = torch.cat((targets, labels))
targets = targets.numpy()

# Compute the predictions
burn_in_preds = 1000  # todo: 6000
all_probs = None
for it, f in enumerate(os.listdir(path_save_samples)):
    if 'client' in f:
        continue
    if int(f) < burn_in_preds:
        print("Skip " + f)
        continue
    path = os.path.join(path_save_samples, f)
    all_probs_test = predictions(testloader, net, path).cpu().numpy()
    preds = np.argmax(all_probs_test, axis=1)
    print(f'Iter = {it + 1}, keep ' + f + f', accuracy = {np.round(100 * np.mean(preds == targets), 1)}')
    if all_probs is None:
        all_probs = all_probs_test
    else:
        all_probs = (it * all_probs + all_probs_test) / (it + 1)
preds = np.argmax(all_probs, axis=1)
final_acc = np.mean(preds == targets)
print('--- Final accuracy =', final_acc)

# End the timer
executionTime = time.time() - startTime
print("Execution time =", executionTime)

# Load the HMC reference predictions
if args.dataset_name == 'CIFAR10':
    with open(path_dataset + '/cifar10_probs.csv', 'r') as fp:
        reference = np.loadtxt(fp)

    # Now we can compute the metrics
    dglmc_agreement = agreement(all_probs, reference)
    dglmc_total_variation_distance = total_variation_distance(all_probs, reference)

    # Print the scores
    print("Agreement =", dglmc_agreement, "Total variation =", dglmc_total_variation_distance)

# Save the results
save_dict = vars(args)
save_dict["execution time"] = executionTime
if args.dataset_name == 'CIFAR10':
    save_dict["dglmc_agreement"] = dglmc_agreement
    save_dict["dglmc_total_variation_distance"] = dglmc_total_variation_distance
with open(path_txt, 'w') as f:
    f.write('\t--- DGLMC ---\n\n')
    for key, value in save_dict.items():
        f.write('%s:%s\n' % (key, value))
# save_dict["model_state_dict"] = model_state_dict
save_dict["all_probs"] = all_probs

# Save the statistics
if os.path.exists(path_stats):
    saved_dict = torch.load(path_stats)
    save_dict.update(saved_dict)
torch.save(save_dict, path_stats)

# Compute other statistics
ytest = np.loadtxt(path_dataset + '/cifar10_test_y.csv').astype(
    int) if args.dataset_name == 'CIFAR10' else testset.targets.numpy()

# Compute the accuracy in function of p(y|x)>tau
tau_list = np.linspace(0, 1, num=100)
accuracies, misclassified = confidence(ytest, all_probs, tau_list)

# Compute the Expected Calibration Error (ECE)
ece = ECE(all_probs, ytest, num_bins=20)

# Compute the Brier Score
bs = BS(ytest, all_probs)

# Compute the accuracy - confidence
acc_conf = accuracy_confidence(all_probs, ytest, tau_list, num_bins=20)

# Compute the calibration curve
cal_curve = calibration_curve(all_probs, ytest, num_bins=20)

# Save the statistics
save_dict["ytest"] = ytest
save_dict["tau_list"] = tau_list
save_dict["all_probs"] = all_probs
save_dict["accuracies"] = accuracies
save_dict["calibration_curve"] = cal_curve
save_dict["accuracy_confidence"] = acc_conf
torch.save(save_dict, path_stats)

# todo: delete
import matplotlib.pyplot as plt

plt.plot(tau_list, acc_conf)
plt.savefig(path_figures + '/acc_conf-dglmc' + title + '.pdf', bbox_inches='tight')
plt.plot(cal_curve[1], cal_curve[0] - cal_curve[1])
plt.savefig(path_figures + '/cal_curve-dglmc' + title + '.pdf', bbox_inches='tight')
# todo: end to del

# Compute the Negative Log Likelihood (NLL)
sample_list = []
for f in os.listdir(path_save_samples):
    if 'client' in f:
        continue
    net.load_state_dict(torch.load(os.path.join(path_save_samples, f)))
    net.to(device)
    sample_list.append(copy.deepcopy(net))
entropy_dict, nll = Predictive_entropy(ytest, all_probs, PostNet(sample_list), transform, args.dataset_name,
                                       path_dataset)

# Compute the Area Under the Curve (AUC)
auc = AUC(entropy_dict["Dataset"], entropy_dict["OOD dataset"])

# Save the statistics
save_dict["entropy_dict"] = entropy_dict
torch.save(save_dict, path_stats)

# Store the ECE, BS, NNL, AUC
file = open(path_txt, 'a')
file.write(f"\nFinal accuracy = {final_acc}, \nECE = {ece}, \nBS = {bs}, \nNLL = {nll}, \nAUC = {auc}")
file.close()  # to change the file access mode
