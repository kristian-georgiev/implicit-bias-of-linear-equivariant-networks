import torch
from torchvision import datasets, transforms
import e2cnn
from e2cnn import nn as e2nn
import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import argparse

# local imports
from src.nn_training import Group, plot_schatten_norm_sums_and_loss
from src.nn_training import g_net, conv_net, fc_net
from src.nn_training import relu_g_net_pool, relu_conv_net_pool, relu_fc_net_pool

from src.utils import get_training_dataframes, postprocess

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='DEBUG')
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--arch_type', choices=['FC', 'CNN', 'GCNN'])
parser.add_argument('--net_type', choices=['linear', 'relu_pool'])
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
print(args.experiment_name)

lr = 3e-6

if args.arch_type == 'FC':
    lr = 1e-4
    if args.net_type == 'linear':
        nets = {'FC': fc_net}
    else:
        nets = {'FC': relu_fc_net_pool}
elif args.arch_type == 'GCNN':
    if args.net_type == 'linear':
        nets = {"G-CNN": g_net}
    else:
        nets = {"G-CNN": relu_g_net_pool}
else:
    if args.net_type == 'linear':
        nets = {"CNN": conv_net}
    else:
        nets = {"CNN": relu_conv_net_pool}

transform_comp = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((28, 28)),
                                     transforms.Normalize((0.1307,), (0.3081,))])
mnist_data = datasets.MNIST('./data', transform=transform_comp)

batch_size = 50
data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=20)

group = Group('MNIST')
force_train = False

dfs = get_training_dataframes(args.experiment_name, force_train)

dfs = plot_schatten_norm_sums_and_loss(nets, group, dataloader=data_loader,
                                       postprocess_fn=postprocess,
                                       N=args.N, epochs=args.epochs,
                                       cuda=True, dfs=dfs,
                                       exp_name=args.experiment_name, lr=lr)

with open(f'data/training/{args.experiment_name}.pickle', 'wb') as f:
    pickle.dump(dfs, f, protocol=pickle.HIGHEST_PROTOCOL)
