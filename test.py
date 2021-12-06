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
    # pdb.set_trace()
    pi.load_state_dict(model['action_net'])
    d = Discriminator(state_dim, action_dim)
    d.load_state_dict(model['disc_net'])
    v = ValueNetwork(state_dim)
    v.load_state_dict(model['value_net'])
    # pdb.set_trace()
    # lane_index = []
    pi.eval()
    d.eval()
    v.eval()
    while True:
        # pi = PolicyNetwork(state_dim, action_dim)
        obs = []
        acts = []
        rwds = []
        _obs = []
        # t = time.time()
        ob = env.reset()
        # pdb.set_trace()
        done = False
        step = 0
        while not done and step <= 1000:
            # pdb.set_trace()
            # print(time.time() - t)
            # t = time.time()
            step += 1
            # pdb.set_trace()
            ob = expert_collector2(ob)
            # pdb.set_trace()
            # print(time.time()-t)
            # t = time.time()
            _obs.append(ob)
            ob = feature2(ob)
            # print(time.time()-t)
            # t = time.time()
            act = pi(ob)
            act = act.sample()
            act = list(act.cpu().numpy())
            # print(time.time()-t)
            # t = time.time()
            obs.append(ob)
            acts.append(act)

            ob, r, done, _ = env.step(act)
            # print(time.time()-t)
            # t = time.time()
            # lane_index.append(ob.ego_vehicle_state.lane_index)
            rwds.append(r)

        # print(time.time() - t)
        # t = time.time()
        # pdb.set_trace()
        # lane_index = list(np.unique(lane_index))
        # print(lane_index)

        obs = torch.FloatTensor(obs)
        acts = torch.FloatTensor(acts)
        print(-torch.log(1-d(obs, acts)))
        # print(rwds)
        # print(d(obs, acts))
        print(v(obs))
        print(acts)
        pdb.set_trace()


if __name__ == '__main__':
    main()