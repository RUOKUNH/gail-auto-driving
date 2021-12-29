import pdb

import numpy as np
from utils import *
import torch
from torch import FloatTensor
import random
import torch.nn.functional as F


class PPO:
    def __init__(self, train_param, lr,
                 PolicyNet, ValueNet, targetValueNet,
                 state_dim, action_dim, n_step=5):
        self.train_param = train_param
        self.beta = self.train_param['beta']
        self.max_kl = train_param['max_kl']
        self.policy = PolicyNet
        self.value = ValueNet
        self.target_value = targetValueNet
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr[0])
        self.value_optimizer = torch.optim.Adam(self.target_value.parameters(), lr=lr[1])
        self.n_step = n_step
        self.action_dim = action_dim
        self.state_dim = state_dim

    def compute_adv(self, batch, gamma):
        s = batch["state"]
        r = batch["reward"].reshape(-1, 1)
        s1 = batch["next_state"]
        done = batch["done"].reshape(-1, 1)
        with torch.no_grad():
            adv = r + gamma * (1 - done) * self.target_value(s1) - self.target_value(s)
        return adv

    def L_clip(self, adv, ratio, old_dist, obs, epsilon=0.2):
        rated_adv = torch.zeros(len(adv))
        _grad = False
        for idx, (r, v) in enumerate(zip(ratio, adv)):
            if v >= 0:
                rated_adv[idx] = min(r, 1 + epsilon) * v
            else:
                rated_adv[idx] = max(r, 1 - epsilon) * v
        loss = torch.mean(-rated_adv)
        if self.train_param['penalty']:
            dist = self.policy(obs)
            kld = kl_divergence(dist, old_dist, self.action_dim, require_grad=True)
            loss -= self.beta * kld
        return loss

    def rescale_and_line_search(self, old_dist, new_param, old_param, obs):
        d_param = new_param - old_param
        alpha = 1.0
        for _ in range(10):
            new_param = old_param + alpha * d_param
            set_params(self.policy, new_param)
            dist = self.policy(obs)
            kld = kl_divergence(dist, old_dist, self.action_dim)
            if kld < self.max_kl:
                return
            else:
                alpha *= 0.7
        set_params(self.policy, old_param)
        print('step too large')

    def action(self, state):
        dist = self.policy(state)
        return dist.sample()

    def dist(self, state, action=None):
        _dist = self.policy(state)
        if action is None:
            action = _dist.sample()
        log_prob = _dist.log_prob(action.reshape(_dist.sample().shape))
        return _dist, log_prob

    def soft_update(self, source, target, tau=0.01):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch, gamma, mini_batch, kld_limit=False, value_only=False):
        batch_size = len(batch["state"])
        shuffle_idx = np.arange(batch_size).astype(np.long)
        random.shuffle(shuffle_idx)
        shuffle_idx = torch.from_numpy(shuffle_idx)
        aloss = 0
        vloss = 0
        for _iter in range(batch_size // mini_batch):
            idx = torch.arange(_iter*mini_batch, (_iter+1)*mini_batch)
            idx = shuffle_idx[idx].long()
            s = batch["state"][idx]
            a = batch["action"][idx]
            r = batch["reward"][idx].reshape(-1, 1)
            s1 = batch["next_state"][idx]
            adv = batch["adv"][idx]
            done = batch["done"][idx].reshape(-1, 1)
            old_log_prob = batch["log_prob"][idx].reshape(-1, 1)

            td_target = r.float() + gamma * self.target_value(s1) * (1 - done)
            value_loss = torch.mean(F.mse_loss(self.target_value(s), td_target.detach()))
            if not (torch.isnan(value_loss) or torch.isinf(value_loss)):
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
            vloss += value_loss.detach()

            current_dist = self.policy(s)
            log_prob = current_dist.log_prob(a.reshape(current_dist.sample().shape)).reshape(-1, 1)
            ratio = torch.exp(log_prob - old_log_prob.detach()).reshape(-1)
            adv = adv.reshape(-1)
            old_param = get_flat_params(self.policy).clone()
            dist_entropy = torch.mean(current_dist.entropy())
            actor_loss = self.L_clip(adv, ratio, old_dist=current_dist, obs=s)
            if self.train_param['use_entropy']:
                actor_loss -= dist_entropy * 0.01
            if self.train_param['l2_norm']:
                actor_loss += self.train_param['l2_norm_weight'] * torch.mean(get_flat_params(self.policy))
            if not (torch.isnan(actor_loss) or torch.isinf(actor_loss)):
                if not value_only:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
            new_param = get_flat_params(self.policy)
            if self.train_param['line_search']:
                self.rescale_and_line_search(current_dist, new_param, old_param, s)
            new_dist = self.policy(s)
            kld = kl_divergence(new_dist, current_dist, self.action_dim)
            if self.train_param['penalty']:
                if kld < self.max_kl / 1.5:
                    self.beta /= 2
                if kld > self.max_kl * 1.5:
                    self.beta *= 2

            aloss += actor_loss.detach()

            self.soft_update(self.value, self.target_value, 0.01)
            self.value.load_state_dict(self.target_value.state_dict())

        return vloss, aloss

    def print_param(self):
        policy_param = get_flat_params(self.policy)
        value_param = get_flat_params(self.value)
        print(torch.max(policy_param))
        print(torch.max(value_param))

    def get_pnet(self):
        return self.policy