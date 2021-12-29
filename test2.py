import pdb
from net import *
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
        hidden = [128] * 4
        self.hidden = hidden
        # self.input = nn.Sequential(
        #         # nn.BatchNorm1d(vector_size),
        #         nn.Linear(vector_size, hidden[0]),
        #     )

        self.layers = nn.ModuleList()
        last_size = vector_size
        for i in range(len(hidden)):
            self.layers.append(
                nn.Sequential(
                    # nn.LayerNorm(hidden[i]),
                    nn.Linear(last_size, hidden[i]),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                )
            )
            last_size = hidden[i]

        self.output = nn.Linear(hidden[-1], 2)

    def forward(self, x):
        # x = x / 10
        # x = self.input(x)
        for layer in self.layers:
            # x = layer(x) + x
            x = layer(x)
        x = self.output(x)
        # x = torch.tanh(x)
        # x = torch.sigmoid(x)
        # x = x-0.5
        return x


weight = np.array([-1.6541483 ,  1.6964793 ,  0.7340574 , -3.096751  , -4.365757  ,
        2.4427834 , -1.2998891 ,  4.6308403 ,  1.7422681 , -0.0795083 ,
        4.081893  ,  4.086707  ,  0.36737156,  4.3161373 ,  3.6109133 ,
       -3.6766458 , -4.107183  ,  1.5030646 ,  1.4355726 , -1.0142331 ,
        4.6946507 , -2.3798106 , -0.45307064, -3.716114  ,  0.9551339 ,
       -2.2365117 ,  0.7316961 , -1.4348145 ,  1.5554662 ,  3.955039  ,
        2.3997731 , -3.1243215 ,  3.4392948 , -1.5711551 ,  3.5670576 ,
       -3.357282  ,  4.3879614 ,  0.67608166, -1.375834  , -4.318674,-1.6541483 ,  1.6964793 ,  0.7340574 , -3.096751  , -4.365757  ,
        2.4427834 , -1.2998891 ,  4.6308403 ,  1.7422681 , -0.0795083 ,
        4.081893  ,  4.086707  ,  0.36737156,  4.3161373 ,  3.6109133 ,
       -3.6766458 , -4.107183  ,  1.5030646 ,  1.4355726 , -1.0142331 ,
        4.6946507 , -2.3798106 , -0.45307064, -3.716114  ,  0.9551339 ,
       -2.2365117 ,  0.7316961 , -1.4348145 ,  1.5554662 ,  3.955039  ,
        2.3997731 , -3.1243215 ,  3.4392948 , -1.5711551 ,  3.5670576 ,
       -3.357282  ,  4.3879614 ,  0.67608166, -1.375834  , -4.318674   ],
      dtype=np.float32)
weight1 = torch.tensor([weight[:59]])
weight2 = torch.tensor([weight[-59:]])
weight_ = torch.cat([weight1, weight2]).reshape(-1, 2)


def func(x):
    x = x*10
    x = torch.FloatTensor(x)
    x = x @ weight_
    x = torch.tanh(x) + x ** 2 / 100
    x = torch.exp(x) - 3
    x = torch.sigmoid(x)
    # x[x >= 0.5] = 1
    # x[x < 0.5] = 0
    # x *= 100
    # x /= 10

    return x


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train():
    vector_size = 59
    model = FIT(vector_size)
    iter = 10000
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    loss = []
    for k in range(50000):
        np.random.seed()
        model.train()
        for i in range(1):
            input = torch.from_numpy(np.random.random((256, vector_size)).astype(np.float32) * 2 - 1)
            input[:, -20:] = torch.from_numpy(np.random.random((256, 20)).astype(np.float32))
            input[:, :29] = torch.from_numpy(np.random.random((256, 29)).astype(np.float32)*-1)
            output = model(input)
            target = func(input)
            # pdb.set_trace()
            # pdb.set_trace()
            # target = label[idx]
            # idx_0 = (target == 0)
            # idx_1 = (target == 1)
            # output_0 = output[idx_0]
            # output_1 = output[idx_1]
            # _los = torch.nn.functional.binary_cross_entropy(
            #     output_1, torch.ones_like(output_1)
            # ) + torch.nn.functional.binary_cross_entropy(
            #     output_0, torch.zeros_like(output_0)
            # )s
            _loss = torch.nn.functional.mse_loss(output, target)
            if torch.isnan(_loss) or torch.isinf(_loss):
                continue
            if not torch.isfinite(_loss):
                continue
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            loss.append(float(_loss))
        print(f'iter {k} loss {np.mean(loss[-50:])}')
        if (k+1) % 2000 == 0:
            model.eval()
            with torch.no_grad():
                x = torch.zeros(10000, vector_size)
                x[:, 0] = torch.from_numpy(np.linspace(-1, 1, 10000).astype(np.float32))
                # x[:, 1:] = torch.from_numpy(np.random.random(vector_size-1).astype(np.float32)*0.5-0.25)
                y = func(x)
                pred = model(x)
                # pdb.set_trace()
                # pred = list(pred)
                # pdb.set_trace()
                plt.plot(np.linspace(-1, 1, 10000).astype(np.float32), pred[:, 0])
                plt.plot(np.linspace(-1, 1, 10000).astype(np.float32), y[:, 0])
                plt.show()
                plt.plot(np.arange(len(loss)-50), loss[50:])
                plt.show()


def show_net():
    state_dim = 23
    action_dim = 2
    x = torch.zeros((1000, state_dim+action_dim))
    x[:, -10] = torch.from_numpy(np.linspace(0, 1, 1000).astype(np.float32)*2-1)
    # x[:, -2] = x[:, 0]
    # x[:, -1] = x[:, 0]
    d = Discriminator(state_dim, action_dim)
    v = ValueNetwork(state_dim)
    pi = PolicyNetwork(state_dim, action_dim)
    exp = ''
    model_path = 'model' + exp + '.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    d.load_state_dict(model['disc_net'])
    v.load_state_dict(model['value_net'])
    # pi.load_state_dict(model['action_net'])
    d.eval()
    v.eval()
    pi.eval()
    output = d(x[:, :state_dim], x[:, -2:]).detach()
    # output = v(x[:, :state_dim]).detach()
    plt.plot(np.linspace(0, 1, 1000).astype(np.float32)*2-1, list(output))
    plt.show()


if __name__ == '__main__':
    train()
    # show_net()

