import pdb

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from utils import *
from net import *
from ppo import PPO
import matplotlib.pyplot as plt
import pickle as pkl
import random
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import os
import torch.nn.functional as F
from threading import Lock
from traffic_simulator import TrafficSim
from multiagent_traffic_simulator import MATrafficSim


class GAIL_PPO:
    def __init__(
            self,
            state_dim,
            action_dim,
            args,
            policy_net=None,
            alim=True
    ):
        if policy_net is None:
            policy_net = [512, 256, 128, 64]
        self.train_param = {
            # basic info
            'policy_net': policy_net,
            'value_net': [512, 256, 128, 64],
            'critic_net': [512, 256, 128, 64],
            'state_dim': state_dim,
            'action_dim': action_dim,

            # learning rate
            'policy_lr': 5e-5,
            'value_lr': 5e-5,
            'critic_lr': 5e-5,

            # optimizer and loss param
            'max_kl': 0.01,
            'beta': 1,
            'gamma': 0.99,
            'max_step': 1000,
            'batch_size': 4096,
            'critic_mini_batch': 4096,
            'mini_batch': 1024,
            'policy_mini_epoch': 5,
            'critic_mini_epoch': 5,
            'penalty': True,
            'line_search': False,

            'critic_penalty': True,
            'penalty_weight': 3,

            'action_limit': alim,
        }
        self.state_dim = state_dim
        self.action_dim = action_dim

        policy_lr = self.train_param['policy_lr']
        value_lr = self.train_param['value_lr']
        critic_lr = self.train_param['critic_lr']

        lr = [policy_lr, value_lr, critic_lr]

        self.policy = PolicyNetwork(state_dim, action_dim*2, self.train_param['policy_net'], self.train_param['action_limit'])
        self.value = ValueNetwork(state_dim, self.train_param['value_net'])
        self.target_value = ValueNetwork(state_dim, self.train_param['value_net'])
        self.critic = Discriminator(self.state_dim, self.action_dim, self.train_param['critic_net'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.ppo = PPO(self.train_param, lr,
                       self.policy, self.value, self.target_value, state_dim, action_dim, n_step=1)

        self.args = args

        self.agent_num = 20
        self.env = env = MATrafficSim(['../scenarios/ngsim'], self.agent_num)

    def normalize(self, state):
        return (state - self.normalize_weight) / self.normalize_std

    def collect(self, batch_size, mini_batch, max_step, feature, descriptor):
        states = []
        next_states = []
        acts = []
        dones = []
        log_probs = []
        x_distance_log = []
        success_log = []
        self.critic.eval()
        self.ppo.policy.eval()
        self.ppo.target_value.eval()
        while len(states) < batch_size:
            state = self.env.reset()
            x_start = {}
            step = 0
            finish = 0
            while finish < self.agent_num and step < max_step:
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
                _state = self.normalize(_state)
                _action = self.ppo.action(_state)
                _, log_prob = self.ppo.dist(_state, _action)
                action = {}
                idx = -1
                for agent_id in state.keys():
                    idx += 1
                    action[agent_id] = _action[idx].detach().numpy()
                next_state, r, done, _ = self.env.step(action)
                for agent_id in done.keys():
                    if done[agent_id]:
                        finish += 1
                        x_end = next_state[agent_id].ego_vehicle_state.position[0]
                        x_distance_log.append(x_end - x_start[agent_id])
                        if next_state[agent_id].events.reached_goal:
                            success_log.append(1)
                        else:
                            success_log.append(0)
                if descriptor:
                    _next_state = [descriptor(feature(expert_collector3(next_state[id]))) for id in state.keys()]
                else:
                    _next_state = [feature(expert_collector3(next_state[id])) for id in state.keys()]
                _next_state = torch.tensor(_next_state).float()
                _next_state = self.normalize(_next_state)
                _done = [done[id] for id in state.keys()]
                states += _state.tolist()
                acts += _action.detach().tolist()
                next_states += _next_state.tolist()
                log_probs += log_prob.reshape(-1).detach().tolist()
                dones += _done

                state = {}
                for agent_id in next_state.keys():
                    if done[agent_id]:
                        continue
                    state[agent_id] = next_state[agent_id]
        rewards = -torch.log(1 - self.critic(torch.tensor(states), torch.tensor(acts))).reshape(-1).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        idx = np.arange(len(states)//mini_batch*mini_batch).astype(np.long)
        random.shuffle(idx)
        idx = torch.from_numpy(idx)
        rewards = rewards.reshape(-1, 1)[idx]
        states = torch.tensor(states)[idx]
        next_states = torch.tensor(next_states)[idx]
        acts = torch.tensor(acts)[idx]
        log_probs = torch.tensor(log_probs).reshape(-1, 1)[idx]
        dones = torch.tensor(dones).int().reshape(-1, 1)[idx]
        batch = {"state": states, "action": acts,
                 "log_prob": log_probs,
                 "next_state": next_states, "done": dones,
                 "reward": rewards, }
        adv = self.ppo.compute_adv(batch, 0.99)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["adv"] = adv
        return batch, x_distance_log, success_log

    def action(self, state):
        return self.ppo.action(state)

    def train(self, expert_path, feature, descriptor=None):
        if not os.path.exists(f'{self.args.exp}'):
            os.mkdir(f'{self.args.exp}')
        if not os.path.exists(f'{self.args.exp}/model'):
            os.mkdir(f'{self.args.exp}/model')
        log_path = f"{self.args.exp}/train_log.txt"
        SHA_TZ = timezone(
            timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)

        # Train Log
        with open(log_path, 'a') as log:
            log.write(f"\n\n\n{time.asctime(utc_now.astimezone(SHA_TZ).timetuple())}")
            log.write(f"\ntrain_params:")
            for k, v in self.train_param.items():
                log.write(f"\n{k}:{v}")
        reward_log = []
        srate_log = []
        dloss_log = []
        aloss_log = []
        vloss_log = []
        exp_score_log = []
        gen_score_log = []

        episode_num = 10000
        max_step = self.train_param['max_step']
        gamma = self.train_param['gamma']
        batch_size = self.train_param['batch_size']
        mini_batch = self.train_param['mini_batch']
        critic_mini_batch = self.train_param['critic_mini_batch']

        expert_buffer = []
        pkf = open(expert_path, 'rb')
        while True:
            try:
                record = pkl.load(pkf)
                expert_obs = record['observation']
                if descriptor:
                    expert_obs = [descriptor(obs) for obs in expert_obs]
                expert_act = record['actions']
                expert_buffer.append((expert_obs, expert_act))
            except EOFError:
                pkf.close()
                break
        expert_state = []
        expert_action = []
        for expert_obs, expert_act in expert_buffer:
            expert_state += expert_obs
            expert_action += expert_act
        expert_state = torch.tensor(expert_state).float()
        expert_action = torch.tensor(expert_action).float()
        self.normalize_weight = torch.mean(expert_state, dim=0)
        self.normalize_std = torch.std(expert_state, dim=0)
        expert_state = (expert_state - self.normalize_weight) / self.normalize_std
        t = time.time()
        for episode in range(episode_num):
            # Generate interactive data
            batch, dlog, slog = self.collect(batch_size, mini_batch, max_step, feature, descriptor)
            with open(f'{self.args.exp}/reward_log.txt', 'a') as log:
                log.write(f"{dlog}")
                log.write('\n')
                log.write(f"{slog}")
                log.write('\n')
            reward_log.append(np.mean(dlog))
            srate_log.append(np.mean(slog))
            self.save_model(f"{self.args.exp}/model/model_{episode}.pth")

            self.critic.train()
            self.ppo.policy.train()
            self.ppo.target_value.train()
            aloss = 0
            vloss = 0
            dloss = 0

            # Update Policy
            for _ in range(self.train_param['policy_mini_epoch']):
                _vloss, _aloss = self.ppo.update(batch, gamma, mini_batch)
                aloss += _aloss
                vloss += _vloss

            # Update Critic
            for _ in range(self.train_param['critic_mini_epoch']):
                expert_idx = np.arange(len(expert_state)).astype(np.long)
                random.shuffle(expert_idx)
                expert_idx = torch.from_numpy(expert_idx)
                _exp_score = []
                _gen_score = []
                _batch_size = len(batch["state"])
                shuffle_idx = np.arange(_batch_size).astype(np.long)
                random.shuffle(shuffle_idx)
                shuffle_idx = torch.from_numpy(shuffle_idx)
                for _iter in range(batch_size // critic_mini_batch):
                    idx = torch.arange(_iter * mini_batch, (_iter + 1) * mini_batch)
                    gen_idx = shuffle_idx[idx]
                    exp_idx = expert_idx[idx]
                    gen_state = batch["state"][gen_idx]
                    gen_action = batch["action"][gen_idx]
                    exp_state = expert_state[exp_idx]
                    exp_action = expert_action[exp_idx]
                    exp_scores = self.critic(exp_state, exp_action)
                    gen_scores = self.critic(gen_state, gen_action)
                    _exp_score += exp_scores.detach().tolist()
                    _gen_score += gen_scores.detach().tolist()
                    critic_loss = torch.nn.functional.binary_cross_entropy(
                        exp_scores, torch.ones_like(exp_scores)
                    ) + torch.nn.functional.binary_cross_entropy(
                        gen_scores, torch.zeros_like(gen_scores)
                    )
                    balance_penalty = self.train_param['penalty_weight'] * torch.mean((exp_scores - gen_scores) ** 2)
                    if self.train_param['critic_penalty']:
                        critic_loss += balance_penalty
                    if not (torch.isnan(critic_loss) or torch.isinf(critic_loss)):
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                    dloss += critic_loss.detach()

            dloss_log.append(dloss)
            aloss_log.append(aloss)
            vloss_log.append(vloss)
            exp_score_log.append(np.mean(_exp_score))
            gen_score_log.append(np.mean(_gen_score))
            log_info = f"{self.args.exp} episode {episode} " + \
                       "critic loss %.2f actor loss %.2f value loss %.2f " % (dloss, aloss, vloss) + \
                       "mdist %.1f srate %.3f " % (reward_log[-1], srate_log[-1]) + \
                       "exp_score %.2f gen_score %.2f " % (np.mean(_exp_score), np.mean(_gen_score)) + \
                       f"time {time.time()-t}"

            print_info = f"{self.args.exp} episode {episode} " + \
                         "critic loss %.2f "% dloss + \
                         "mdist %.1f " % reward_log[-1] + \
                         "exp_score %.2f gen_score %.2f " % (np.mean(_exp_score), np.mean(_gen_score))

            print(print_info)
            with open(log_path, 'a') as log:
                log.write("\n"+log_info)
            t = time.time()

            plt.plot(np.arange(len(dloss_log)), dloss_log)
            plt.savefig(f'{self.args.exp}/dloss.png')
            plt.close()
            plt.plot(np.arange(len(vloss_log)), vloss_log)
            plt.savefig(f'{self.args.exp}/vloss.png')
            plt.close()
            plt.plot(np.arange(len(aloss_log)), aloss_log)
            plt.savefig(f'{self.args.exp}/aloss.png')
            plt.close()
            plt.plot(np.arange(len(reward_log)), reward_log)
            plt.savefig(f'{self.args.exp}/reward.png')
            plt.close()
            plt.plot(np.arange(len(srate_log)), srate_log)
            plt.savefig(f'{self.args.exp}/srate.png')
            plt.close()
            plt.plot(np.arange(len(exp_score_log)), exp_score_log)
            plt.plot(np.arange(len(gen_score_log)), gen_score_log)
            plt.savefig(f'{self.args.exp}/critic_score.png')
            plt.close()

    def save_model(self, path):
        state = {
            'policy': self.ppo.policy.state_dict(),
            'value': self.ppo.value.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))
        self.ppo.policy.load_state_dict(state['policy'])
