import pdb

import gym
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from ppo import PPO

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

env_name = "CartPole-v0"
env = gym.make(env_name)

max_timesteps = 1000
gamma = 0.99


class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_num=2):
        super(Net, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num-1):
            layers.append(torch.nn.Linear(last_size, 16))
            layers.append(torch.nn.ReLU())
            last_size = 16
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._net(inputs)


class PNet(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_num=2):
        super(PNet, self).__init__()
        # Use torch.nn.ModuleList or torch.nn.Sequential For multiple layers
        layers = []
        last_size = input_size
        for i in range(layer_num-1):
            layers.append(torch.nn.Linear(last_size, 16))
            layers.append(torch.nn.ReLU())
            last_size = 16
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        # pdb.set_trace()
        probs = torch.softmax(self._net(inputs), dim=-1)
        dist = Categorical(probs)
        return dist


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo = PPO(PNet, state_dim, action_dim, n_step=1, synchronize_steps=1, mini_batch=10)
vnet = Net(state_dim, 1)
optimizer = torch.optim.Adam(vnet.parameters())

episode_num = 2000
max_timesteps = 1000

reward_log = []
for episode in range(episode_num):

    obs = []
    act = []
    dones = []
    rwd = []
    next_obs = []
    log_probs = []

    state = env.reset()
    env.seed(1)
    random.seed(1)
    # pdb.set_trace()
    state = torch.tensor([state])
    # obs.append(state)

    for timestep in range(max_timesteps):
        # pdb.set_trace()
        dist = ppo.pnet(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # pdb.set_trace()

        next_state, r, done, _ = env.step(action.numpy().reshape(-1)[0])

        obs.append(state)
        next_obs.append(torch.tensor([next_state]))
        act.append(action)
        rwd.append(r)
        dones.append(done)
        log_probs.append(log_prob)

        state = torch.tensor([next_state])
        if done:
            break
    reward_log.append(np.sum(rwd))
    # pdb.set_trace()
    obs = torch.cat(obs).reshape(-1, state_dim)
    act = torch.cat(act).reshape(-1, 1)
    rwd = torch.tensor(rwd).reshape(-1)
    log_probs = torch.tensor(log_probs).reshape(-1)
    next_obs = torch.cat(next_obs).reshape(-1, state_dim)
    # pdb.set_trace()
    rwd = (rwd - torch.mean(rwd)) / (torch.std(rwd) + 1e-8)
    dones = torch.tensor(dones).int().reshape(-1)
    # pdb.set_trace()
    adv = rwd + gamma * (1 - dones) * vnet(next_obs).reshape(-1) - vnet(obs).reshape(-1)
    # pdb.set_trace()
    td_target = rwd + gamma * vnet(next_obs).reshape(-1) * (1-dones)
    # pdb.set_trace()
    critic_loss = torch.mean(F.mse_loss(vnet(obs).reshape(-1), td_target.detach()))
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()

    # old_dist = ppo.pnet(obs)

    ppo.update(log_probs, adv, obs, next_obs, act, gamma, dones, rwd, vnet, epoches=10)


    if episode % 100 == 0:
        print(np.mean(reward_log[-10:]))
        print(critic_loss)





