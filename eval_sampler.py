import pdb
from net import *
from BC import BC
from tqdm import tqdm
import numpy as np
import multiprocessing
from collections import defaultdict
import torch
from utils import *
from gail_ppo import GAIL_PPO
import pickle as pkl


def single_env_rollout(env_ctor_func, agent, feature, args, descriptor):
    expert_path = './expert_data_feature16.pkl'
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
    def get_action(obs, agent, feature):
        state = descriptor(feature(expert_collector3(obs)))
        state = torch.tensor(state).float()
        state = (state - normalize_weight) / normalize_std
        action = agent.action(state)
        return action.reshape(-1).detach().numpy()
    env, eval_group_num = env_ctor_func()
    paths = defaultdict(list)
    print(">>> Generating trajs...")
    for _ in tqdm(range(eval_group_num)):
        state = env.reset()
        vehicle_ids = env.vehicle_id
        path = {v_id: defaultdict(list) for v_id in vehicle_ids}

        done_n = {"__all__": False}
        while not done_n["__all__"]:
            if args.model == 'single':
                act_n = {
                    a_id: get_action(state[a_id], agent, feature)
                    for a_id in state.keys()
                    if not done_n.get(a_id, False)
                }
            else:
                _state = [descriptor(feature(expert_collector3(state[agent_id]))) for agent_id in state.keys()]
                _state = torch.tensor(_state).float()
                _state = (_state - normalize_weight) / normalize_std
                _action = agent.action(_state)
                action = {a_id: _action[i].detach().numpy()
                          for i, a_id in enumerate(state.keys())}
                act_n = {
                    a_id: action[a_id]
                    for a_id in state.keys()
                    if not done_n.get(a_id, False)
                }
            next_state, rew_n, done_n, info_n = env.step(act_n)

            for a_id, info in info_n.items():
                v_id = info["vehicle_id"]
                if a_id in state.keys():
                    path[v_id]["observations"].append(state[a_id].ego_vehicle_state.position[:2])
                    path[v_id]["actions"].append(act_n[a_id])
                    path[v_id]["next_observations"].append(next_state[a_id].ego_vehicle_state.position[:2])
                    path[v_id]["rewards"].append(rew_n[a_id])
                    path[v_id]["dones"].append(done_n[a_id])
                    path[v_id]["infos"].append(info_n[a_id])

            state = next_state

        for v_id in vehicle_ids:
            if len(path[v_id]) > 0:
                paths[v_id].append(path[v_id])

    return paths


class ParallelPathSampler:
    def __init__(
        self,
        env_ctor_func_list,
        agent,
        feature,
        descriptor,
        args,
    ):
        self.env_ctor_func_list = env_ctor_func_list
        self.agent = agent
        self.feature = feature
        self.descriptor = descriptor
        self.args = args

    def collect_samples(self):

        paths = single_env_rollout(self.env_ctor_func_list[0], self.agent, self.feature, self.args, self.descriptor)

        return paths
