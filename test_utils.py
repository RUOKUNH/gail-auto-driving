import pdb

# import gym
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random


class PNet(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(PNet, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        layer_num = len(hidden_size)
        for i in range(layer_num):
            layers.append(torch.nn.Linear(last_size, hidden_size[i]))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        # pdb.set_trace()
        mean, var = torch.chunk(self._net(inputs), 2, dim=-1)
        # dist = Categorical(prob)
        mean = torch.tanh(mean)
        var = torch.nn.functional.softplus(var)
        dist = Normal(mean, var)
        return dist


class PNet2(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PNet2, self).__init__()
        self.net_dims = [256, 128, 64, 32]
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(
                torch.nn.Sequential(
                torch.nn.Linear(last_dim, self.net_dims[i]),
                # torch.nn.Dropout(0.5),
                torch.nn.ReLU())
            )
            last_dim = self.net_dims[i]
        self.layers = torch.nn.ModuleList(*layers)

        self.output = torch.nn.Linear(self.net_dims[-1], action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.var = torch.nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, states):
        x = torch.FloatTensor(states)
        # x = self.input(states)
        for layer in self.layers:
            x = layer(x)
        # mean, var = torch.chunk(self.output(x), 2, dim=-1)
        # var = F.softplus(var)
        # if var.nim
        # var = torch.cat([torch.diag(var[i]).view(1, 2, 2) for i in range(len(var))])

        mean = self.output(x)

        # mean = torch.tanh(mean) * 2
        # var = torch.tanh(self.var) * 0.1
        # var = torch.diag(self.var)

        # if mean.ndim > 1:
        #     mean[:, 0] = torch.tanh(mean[:, 0])
        #     mean[:, 1] = torch.tanh(mean[:, 1]) * 0.1
        # else:
        #     mean[0] = torch.tanh(mean[0])
        #     mean[1] = torch.tanh(mean[0]) * 0.1
        #
        # var = torch.FloatTensor([[5e-2, 0], [0, 25e-4]])
        # dist = torch.distributions.MultivariateNormal(mean, var)

        # return dist
        return mean


class VNet(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(VNet, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num-1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._net(inputs)