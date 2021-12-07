import pdb

import numpy as np

import matplotlib.pyplot as plt
from utils import *
import time
import torch
from torch import nn


def init_weights_xavier(m):
    pdb.set_trace()
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class FIT(nn.Module):
    def __init__(self, vector_size):
        super().__init__()
        hidden = [128] * 7
        self.hidden = hidden
        self.input = nn.Sequential(
                # nn.BatchNorm1d(vector_size),
                nn.Linear(vector_size, hidden[0]),
            )

        self.layers = nn.ModuleList()
        for i in range(len(hidden)-1):
            self.layers.append(
                nn.Sequential(
                    # nn.LayerNorm(hidden[i]),
                    nn.ReLU(),
                    nn.Linear(hidden[i], hidden[i + 1]),
                )
            )

        self.output = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        # x = x / 10
        x = self.input(x)
        for layer in self.layers:
            x = layer(x) + x
            # x = layer(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        # x = x-0.5
        return x


weight = np.array([-1.6541483 ,  1.6964793 ,  0.7340574 , -3.096751  , -4.365757  ,
        2.4427834 , -1.2998891 ,  4.6308403 ,  1.7422681 , -0.0795083 ,
        4.081893  ,  4.086707  ,  0.36737156,  4.3161373 ,  3.6109133 ,
       -3.6766458 , -4.107183  ,  1.5030646 ,  1.4355726 , -1.0142331 ,
        4.6946507 , -2.3798106 , -0.45307064, -3.716114  ,  0.9551339 ,
       -2.2365117 ,  0.7316961 , -1.4348145 ,  1.5554662 ,  3.955039  ,
        2.3997731 , -3.1243215 ,  3.4392948 , -1.5711551 ,  3.5670576 ,
       -3.357282  ,  4.3879614 ,  0.67608166, -1.375834  , -4.318674  ],
      dtype=np.float32)
weight = torch.from_numpy(weight)


def func(x):
    x = x*10
    x = torch.FloatTensor(x)
    x = torch.sum(weight * x, dim=1)
    x = torch.tanh(x) + x ** 2 / 100
    x = torch.exp(x) - 3
    x = torch.sigmoid(x)
    x[x >= 0.5] = 1
    x[x < 0.5] = 0
    # x *= 10

    return x


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train():
    vector_size = 40
    model = FIT(vector_size)
    iter = 10000
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    loss = []
    model.train()
    data = torch.from_numpy(np.random.random((4000, vector_size)).astype(np.float32) * 4 - 2)
    # np.random.seed()
    for k in range(400):
        new_data = torch.from_numpy(np.random.random((400, vector_size)).astype(np.float32) * 4 - 2)
        data = torch.cat([data, new_data])[-4000:]
        label = func(data)
        np.random.seed()
        for i in range(100):
            idx = torch.from_numpy(np.random.randint(0, 4000, 100).astype(np.int64))
            input = data[idx]
            output = model(input).reshape(-1)
            # pdb.set_trace()
            target = label[idx]
            idx_0 = (target == 0)
            idx_1 = (target == 1)
            output_0 = output[idx_0]
            output_1 = output[idx_1]
            _loss = torch.nn.functional.binary_cross_entropy(
                output_1, torch.ones_like(output_1)
            ) + torch.nn.functional.binary_cross_entropy(
                output_0, torch.zeros_like(output_0)
            )
            if torch.isnan(_loss):
                continue
            # _loss = torch.nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            loss.append(float(_loss))
        # pdb.set_trace()
        print(f'iter {k} loss {np.mean(loss[-50:])}')
    with torch.no_grad():
        x = torch.zeros(10000, 40)
        x[:, 0] = torch.from_numpy(np.linspace(-2, 2, 10000).astype(np.float32))
        y = func(x)
        pred = model(x).reshape(-1)
        # pdb.set_trace()
        pred = list(pred)
        plt.plot(np.linspace(-2, 2, 10000).astype(np.float32), pred)
        plt.plot(np.linspace(-2, 2, 10000).astype(np.float32), y)
        plt.show()
        plt.plot(np.arange(len(loss)-50), loss[50:])
        plt.show()


if __name__ == '__main__':
    train()

