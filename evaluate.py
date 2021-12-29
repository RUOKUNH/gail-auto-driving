import pickle as pkl
import os
import json
import pdb
import time
from gail_ppo import GAIL_PPO
from traffic_simulator import TrafficSim
from gail_trpo import GAIL
import torch
from net import *
from utils import *
from torch import FloatTensor
import argparse
from BC import BC
from multiagent_traffic_simulator import MATrafficSim
from fixed_traffic_simulator import FixedMATrafficSim


def evaluate(agent, feature, descriptor=None):
    expert_state = []
    expert_action = []
    pkf = open(expert_path, 'rb')
    while True:
        try:
            record = pkl.load(pkf)
            expert_obs = record['observation']
            if descriptor:
                expert_obs = [descriptor(obs) for obs in expert_obs]
            expert_act = record['actions']
            expert_state += expert_obs
            expert_action += expert_act
        except EOFError:
            pkf.close()
            break
    expert_state = torch.tensor(expert_state).float()
    normalize_weight = torch.mean(expert_state, dim=0)
    normalize_std = torch.std(expert_state, dim=0)
    # pdb.set_trace()
    agent_num = 10
    with open("test_ids.pkl", "rb") as f:
        eval_vehicle_ids = pkl.load(f)

    eval_vehicle_groups = []
    for idx in range(len(eval_vehicle_ids) - agent_num):
        eval_vehicle_groups.append(eval_vehicle_ids[idx: idx + agent_num])
    # env = MATrafficSim(['../scenarios/ngsim'], agent_num, envision=True)
    env = FixedMATrafficSim(['../scenarios/ngsim'], agent_num, eval_vehicle_groups)
    x_distance_log = []
    success_log = []
    # vehicle_id = ['3197', '3205', '3228', '3246', '3238', '3242', '3252', '3260', '3244', '3265']    with open(os.path.join(BASE_DIR, "test_ids.pkl"), "rb") as f:


    # eval_vehicle_groups = eval_vehicle_groups[:10]


    while len(x_distance_log) < 2051:
        # pdb.set_trace()
        state = env.reset()
        _xdlog = []
        x_start = {}
        step = 0
        finish = 0
        while finish < agent_num:
            if len(state) == 0:
                break
            step += 1
            for agent_id in state.keys():
                if agent_id not in x_start.keys():
                    x_start[agent_id] = state[agent_id].ego_vehicle_state.position[0]
            if descriptor:
                _state = [descriptor(feature(expert_collector3(state[agent_id]))) for agent_id in state.keys()]
            else:
                _state = [feature(expert_collector3(state[agent_id])) for agent_id in state.keys()]
            _state = torch.tensor(_state).float()
            _state = (_state - normalize_weight) / normalize_std
            _action = agent.action(_state)
            action = {}
            idx = -1
            for agent_id in state.keys():
                idx += 1
                action[agent_id] = _action[idx].detach().numpy()
            next_state, r, done, _ = env.step(action)
            # for agent_id in done.keys():
            #     if done[agent_id]:
            #         finish += 1
            #         x_end = next_state[agent_id].ego_vehicle_state.position[0]
            #         x_distance_log.append(x_end - x_start[agent_id])
            #         _xdlog.append(x_end - x_start[agent_id])
            #         if next_state[agent_id].events.reached_goal:
            #             print(env.agentid_to_vehid[agent_id])
            #             success_log.append(1)
            #         else:
            #             success_log.append(0)
            if descriptor:
                _next_state = [descriptor(feature(expert_collector3(next_state[id]))) for id in state.keys()]
            else:
                _next_state = [feature(expert_collector3(next_state[id])) for id in state.keys()]
            _next_state = torch.tensor(_next_state).float()
            _next_state = (_next_state - normalize_weight) / normalize_std
            _done = [done[id] for id in state.keys()]

            state = {}
            for agent_id in next_state.keys():
                if done[agent_id]:
                    continue
                state[agent_id] = next_state[agent_id]

        # print(f"totmdist {np.mean(x_distance_log)} totsrate {np.mean(success_log)} "
        #       f"mdist {np.mean(_xdlog)} mind {np.min(_xdlog)}"
        #       f"{len(x_distance_log)} {len(success_log)}")
        pdb.set_trace()


if __name__ == '__main__':
    expert_path = './expert_data_feature16.pkl'
    # exp_path = 'bc-feature16'
    exp_path = 'gail-feature16-criticp1.5'
    state_dim = 65
    action_dim = 2
    feature = feature15
    descriptor = feature15_descriptor
    # descriptor = None
    agent = GAIL_PPO(state_dim, action_dim, None, policy_net=[512, 256, 128, 64])
    agent.load_model(f"{exp_path}/model/model_707.pth")
    # agent.load_model(f"{exp_path}/model/model_213.pth")
    # pnet = PolicyNetwork(state_dim, 4, [256, 128, 64, 64, 32])
    # agent = BC(pnet)
    # agent.load_model(f"{exp_path}/model.pth")
    evaluate(agent, feature, descriptor)