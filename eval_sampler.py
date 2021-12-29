import pdb

from tqdm import tqdm
import numpy as np
import multiprocessing
from collections import defaultdict
import torch
from utils import *
from gail_ppo import GAIL_PPO
import pickle as pkl


expert_path = './expert_data_feature16.pkl'
exp_path = 'gail-feature16-criticp1.5'
state_dim = 65
action_dim = 2
feature = feature15
descriptor = feature15_descriptor
agent = GAIL_PPO(state_dim, action_dim, None, policy_net=[512, 256, 128, 64])
agent.load_model(f"{exp_path}/model/model_707.pth")
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


def get_action(obs):
    state = descriptor(feature(expert_collector3(obs)))
    state = torch.tensor(state).float()
    state = (state - normalize_weight) / normalize_std
    action = agent.action(state)
    return action.reshape(-1).detach().numpy()


def single_env_rollout(env_ctor_func):
    env, eval_group_num = env_ctor_func()
    paths = defaultdict(list)
    print(">>> Generating trajs...")
    for _ in tqdm(range(eval_group_num)):
        state = env.reset()
        vehicle_ids = env.vehicle_id
        path = {v_id: defaultdict(list) for v_id in vehicle_ids}

        done_n = {"__all__": False}
        while not done_n["__all__"]:
            # act_n = {
            #     a_id: get_action(obs_n[a_id])
            #     for a_id in obs_n.keys()
            #     if not done_n.get(a_id, False)
            # }
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

        # print("{} finished".format(vehicle_ids))
        for v_id in vehicle_ids:
            if len(path[v_id]) > 0:
                paths[v_id].append(path[v_id])
        # pdb.set_trace()

    return paths


class ParallelPathSampler:
    def __init__(
        self,
        env_ctor_func_list,
    ):
        self.env_ctor_func_list = env_ctor_func_list

    def collect_samples(self):
        # worker_num = len(self.env_ctor_func_list)
        # queue = multiprocessing.Queue()
        # workers = []
        # for i in range(worker_num):
        #     worker_args = (i, queue, self.env_ctor_func_list[i])
        #     workers.append(multiprocessing.Process(target=single_env_rollout, args=worker_args))
        #
        # for worker in workers:
        #     worker.start()

        # paths = {}
        # for _ in workers:
        #     pid, _paths = queue.get()
        #     paths = {**paths, **_paths}

        paths = single_env_rollout(self.env_ctor_func_list[0])

        return paths
