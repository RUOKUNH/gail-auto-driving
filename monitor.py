import pickle as pkl
import os
import json
import pdb
import time
from multiagent_traffic_simulator import MATrafficSim
from traffic_simulator import TrafficSim
from gail_trpo import GAIL
import torch
from net import *
from utils import *
from torch import FloatTensor
import argparse
from BC import BC
from train import evaluate


def monitor(expert_path, path, feature, state_dim, action_dim, descriptor=None):
    pnet = PolicyNetwork(state_dim, action_dim*2)
    agent = BC(pnet, torch.device('cpu'))
    env = MATrafficSim(['../scenarios/ngsim'], 10)
    expert_state = []
    expert_action = []
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
    normalize_weight = torch.mean(expert_state, dim=0)
    normalize_std = torch.std(expert_state, dim=0)
    while not os.path.exists(f'{path}/model.pth'):
        time.sleep(1)
        print('waiting')
    epoch = agent.load_model(f'{path}/model.pth')
    last_modified_time = time.ctime(os.stat(f'{path}/model.pth').st_mtime)
    x_distance_log = []
    success_log = []
    # collect = 0
    while True:
        dlog, slog = evaluate(env, normalize_weight, normalize_std, agent, feature)
        x_distance_log += dlog
        success_log += slog
        if np.mean(x_distance_log) > 100:
            dlog, slog = evaluate(env, normalize_weight, normalize_std, agent, feature)
            x_distance_log += dlog
            success_log += slog
            agent.save_model(path=f'{path}/model_{round(np.mean(x_distance_log), 0)}_{round(np.mean(success_log), 3)}.pth', epoch=epoch)
        if time.ctime(os.stat(f'{path}/model.pth').st_mtime) > last_modified_time:
            epoch = agent.load_model(f'{path}/model.pth')
            last_modified_time = time.ctime(os.stat(f'{path}/model.pth').st_mtime)
            x_distance_log = []
            success_log = []
            print('##########')
            print(f'Epoch{epoch+1}')
            print('##########')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help='monitor path',
                        type=str)
    args = parser.parse_args()
    path = args.path
    feature = feature15
    state_dim = 65
    action_dim = 2
    expert_path = './expert_data_feature16.pkl'
    monitor(expert_path, path, feature, state_dim, action_dim)