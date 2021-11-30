import os
import json
import pdb

from traffic_simulator import TrafficSim
from gail import GAIL
import torch
from net import *
from utils import *
from torch import FloatTensor
import argparse

def main():
    exp = 'exp18'
    env = TrafficSim(["scenarios/ngsim"])
    print('env created')
    state_dim = 20
    action_dim = 2

    pi = PolicyNetwork(state_dim, action_dim)
    model_path = 'model'+exp+'.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    pi.load_state_dict(model['action_net'])
    # d = Discriminator(state_dim, action_dim)
    # d.load_state_dict(model['disc_net'])
    v = ValueNetwork(state_dim)
    v.load_state_dict(model['value_net'])

    while True:
        obs = []
        acts = []
        ob = env.reset()
        done = False
        step = 0
        while not done and step <= 1000:
            step += 1
            ob = make_obs(ob)
            ob = make_obs_2(ob)
            # ob = ob[:6]
            act = pi(ob)
            # pdb.set_trace()
            # mean = act.mean.detach().numpy()
            # act = mean
            # pdb.set_trace()
            act = act.sample()
            # print(mean, act)
            act = list(act.cpu().numpy())
            obs.append(ob)
            acts.append(act)

            ob, _, done, _ = env.step(act)
            # pdb.set_trace()

        # obs = torch.FloatTensor(obs)
        # acts = torch.FloatTensor(acts)
        # # print(torch.log(d(obs, acts)))
        # print(d(obs, acts))
        # print(v(obs))
        # pdb.set_trace()


if __name__ == '__main__':
    main()
