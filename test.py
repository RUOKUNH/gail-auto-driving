# import pickle as pkl
import os
# import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
# import time
# from multiagent_traffic_simulator import MATrafficSim
# from traffic_simulator import TrafficSim
# from gail_trpo import GAIL
# import torch
# from net import *
# from utils import *
# from torch import FloatTensor
# import argparse
# from BC import BC
# from torch.utils.tensorboard import SummaryWriter


def example_ma():
    env = MATrafficSim(['../scenarios/ngsim'], 20)
    while True:
        state = env.reset()
        done = {}
        for agent_id in state.keys():
            done[agent_id] = False
        while False in done.values():
            action = {}
            for agent_id in state.keys():
                if done[agent_id]:
                    continue
                action[agent_id] = np.array([.0, .0])
            next_state, r, done, _ = env.step(action)
            if len(next_state) != len(state):
                pdb.set_trace()
            state = next_state


def example():
    env = TrafficSim(["../scenarios/ngsim"], envision=True)
    while True:
        ob = env.reset()
        done = False
        step = 0
        while not done and step <= 1000:
            step += 1
            act = np.array([0., 0.])

            if step > 20:
                pdb.set_trace()
            ob, r, done, _ = env.step(act)
            ob1 = expert_collector3(ob)
            ob2 = feature17(ob1)


def test():
    feature = feature15
    state_dim = 67
    descriptor = None
    pnet = PolicyNetwork(state_dim, 2)
    agent = BC(pnet, torch.device('cpu'))
    agent.load_model('bc-feature15/model.pth')
    env = TrafficSim(["../scenarios/ngsim"], envision=True)
    expert_state = []
    expert_action = []
    expert_path = './expert_data_feature16.pkl'
    pkf = open(expert_path, 'rb')
    while True:
        try:
            record = pkl.load(pkf)
            expert_obs = record['observation']
            if descriptor:
                expert_obs = [descriptor(ob) for ob in expert_obs]
            expert_act = record['actions']
            expert_state += expert_obs
            expert_action += expert_act
        except EOFError:
            pkf.close()
            break
    expert_state = torch.tensor(expert_state).float()
    expert_action = torch.tensor(expert_action).float()
    normalize_weight = torch.mean(expert_state, dim=0)
    normalize_std = torch.std(expert_state, dim=0)
    while True:
        state = env.reset()
        state = expert_collector3(state)
        state = feature(state)
        if descriptor:
            state = descriptor(state)
        state = torch.tensor(state).float()
        state = (state - normalize_weight) / normalize_std
        done = False
        while not done:
            action = agent.action(state).detach()
            action /= 10
            next_state, r, done, _ = env.step(action.numpy())
            state = feature(expert_collector3(next_state))
            if descriptor:
                state = descriptor(state)
            state = torch.tensor(state).float()
            state = (state - normalize_weight) / normalize_std
        pdb.set_trace()


def read(log_path):
    # log_path = 'gail-feature16-criticp1.5/train_log.txt'
    mdist = []
    exp_score =[]
    gen_score = []
    with open(log_path, 'r') as f:
        while True:
            x = f.readline().split(' ')
            if x == ['']:
                break
            if x[0].split('-')[0] != 'gail':
                continue
            for i in range(len(x)):
                if x[i] == 'exp_score':
                    exp_score.append(float(x[i+1]))
                if x[i] == 'gen_score':
                    gen_score.append(float(x[i+1]))
                if x[i] == 'mdist':
                    mdist.append(float(x[i+1]))

    return np.array(mdist), np.array(exp_score), np.array(gen_score)


def curve():
    if not os.path.exists('result_plot'):
        os.mkdir('result_plot')
    save_path = 'result_plot'
    log1 = 'gail-feature16-nopenalty/train_log.txt'
    log2 = 'gail-feature16-criticp1.5/train_log.txt'
    mdist1, exp1, gen1 = read(log1)
    mean_dist1 = np.array([np.mean(mdist1[max(i-1,0):i+2]) for i in range(len(mdist1))])
    low_dist1 = np.array([np.min(mdist1[max(i-1,0):i+2]) for i in range(len(mdist1))])
    high_dist1 = np.array([np.max(mdist1[max(i-1,0):i + 2]) for i in range(len(mdist1))])
    mdist2, exp2, gen2 = read(log2)
    mean_dist2 = np.array([np.mean(mdist2[max(i-1,0):i + 2]) for i in range(len(mdist2))])
    low_dist2 = np.array([np.min(mdist2[max(i-1,0):i + 2]) for i in range(len(mdist2))])
    high_dist2 = np.array([np.max(mdist2[max(i-1,0):i + 2]) for i in range(len(mdist2))])
    L = len(mdist1)

    plt.figure(1, (6, 4))
    plt.plot(np.arange(L) * 4096, mean_dist1[:L], label='origin', color='navy')
    plt.fill_between(np.arange(L) * 4096, low_dist1[:L], high_dist1[:L], alpha=0.2, color='navy')
    plt.plot(np.arange(L) * 4096, mean_dist2[:L], label='balanced penalty', color='orange')
    plt.fill_between(np.arange(L) * 4096, low_dist2[:L], high_dist2[:L], alpha=0.2, color='orange')
    plt.xlabel('timesteps')
    plt.ylabel('mean distance(20 cars)')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(save_path + '\mdist_comp.png')
    plt.close()

    plt.figure(1, (6, 4))
    plt.plot(np.arange(L) * 4096, exp1[:L], label='origin', color='navy')
    plt.plot(np.arange(L) * 4096, gen1[:L], color='navy')
    plt.plot(np.arange(L) * 4096, exp2[:L], label='balanced penalty', color='orange')
    plt.plot(np.arange(L) * 4096, gen2[:L], color='orange')
    plt.xlabel('timesteps')
    plt.ylabel('critic scores(0~1)')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(save_path + '\critic_score.png')
    plt.close()


if __name__ == '__main__':
    # example_ma()
    # example()
    # board()
    # test()
    curve()
