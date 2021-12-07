import os
import json
import pdb
import time

from traffic_simulator import TrafficSim
from gail_trpo import GAIL
import torch
from net import *
from utils import *
from torch import FloatTensor
import argparse

def main():
    exp = 'exp42'
    env = TrafficSim(["../scenarios/ngsim"])
    state_dim = 34
    action_dim = 2

    pi = PolicyNetwork(state_dim, action_dim)
    model_path = 'bestmodel'+exp+'.pth'
    # model_path = 'model' + exp + '.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    pi.load_state_dict(model['action_net'])
    d = Discriminator(state_dim, action_dim)
    d.load_state_dict(model['disc_net'])
    v = ValueNetwork(state_dim)
    v.load_state_dict(model['value_net'])
    pi.eval()
    d.eval()
    v.eval()
    while True:
        obs = []
        acts = []
        rwds = []
        _obs = []
        ob = env.reset()
        done = False
        step = 0
        while not done and step <= 1000:
            step += 1
            ob = expert_collector3(ob)
            _obs.append(ob)
            ob = feature2(ob)
            act = pi(ob)
            act = act.sample()
            act = list(act.cpu().numpy())
            obs.append(ob)
            acts.append(act)

            ob, r, done, _ = env.step(act)
            rwds.append(r)

        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(acts)
        print(-torch.log(1-d(obs, acts)))
        print(v(obs))
        # print(acts)
        pdb.set_trace()


if __name__ == '__main__':
    main()
