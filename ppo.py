import pdb

import numpy as np
from utils import *
import torch
from torch import FloatTensor
import random
from net import *


class PPO:
    def __init__(self, state_dim, action_dim, config, n_step=5, synchronize_steps=1, mini_batch=100):
        self.collect_pnet = PolicyNetwork(state_dim, action_dim)
        self.pnet = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.pnet.parameters())
        self.synchronize_steps = synchronize_steps
        self.synchronize_step = 0
        self.mini_batch = mini_batch
        self.n_step = n_step
        self.config = config

    def L_clip(self, advs, ratio, epsilon=0.2):
        rated_advs = 0
        _grad = False
        for i in range(len(advs)):
            if advs[i] >= 0:
                if ratio[i] > 1+epsilon:
                    rated_advs += (1+epsilon) * advs[i]
                else:
                    rated_advs += ratio[i] * advs[i]
                    _grad = True
            else:
                if ratio[i] < 1-epsilon:
                    rated_advs += (1-epsilon) * advs[i]
                else:
                    rated_advs += ratio[i] * advs[i]
                    _grad = True
        rated_advs /= len(advs)
        return -rated_advs, _grad

    def update(self, obs, acts, gamma, vnet, dnet):
        self.synchronize_step += 1
        ######### update ########
        _update = False
        update_data = []
        vnet.eval()
        dnet.eval()
        for i in range(len(obs)):
            ob = obs[i]
            if len(ob) <= self.n_step:
                continue
            act = acts[i]
            ob, act = FloatTensor(ob), FloatTensor(act)
            costs = torch.log(dnet(ob, act)).squeeze().detach()
            rwds = -1 * costs
            curr_val = vnet(ob).detach().view(-1)
            nnext_val = curr_val[self.n_step:]
            curr_val = curr_val[:len(curr_val)-self.n_step]
            advs = gamma**self.n_step*nnext_val - curr_val
            for k in range(self.n_step):
                advs += gamma**k * rwds[k:k+len(curr_val)]
            for k in range(len(advs)):
                update_data.append((list(ob[k]), list(act[k]), advs[k]))
        random.shuffle(update_data)
        st = 0
        self.pnet.train()
        while st < len(update_data):
            ed = st + self.mini_batch
            if ed > len(update_data):
                ed = len(update_data)
            batch_data = update_data[st:ed]
            _advs = []
            _obs = []
            _acts = []
            for _ob, _act, _adv in batch_data:
                _obs.append(_ob)
                _acts.append(_act)
                _advs.append(_adv)
            _obs, _acts = FloatTensor(_obs), FloatTensor(_acts)
            dist = self.pnet(_obs)
            old_dist = self.collect_pnet(_obs)
            _ratio = torch.exp(dist.log_prob(_acts)
                               - old_dist.log_prob(_acts).detach())
            loss, _grad = self.L_clip(_advs, _ratio)
            # dist_causal_entropy = self.config['lambda_'] * (-1 * self.pnet(_obs).log_prob(_acts)).mean()
            # loss -= dist_causal_entropy
            if _grad:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _update = True
            st = ed
        #########################
        if self.synchronize_step >= self.synchronize_steps:
            self.collect_pnet.load_state_dict(self.pnet.state_dict())
            self.synchronize_step = 0

        return self.collect_pnet, _update

    def get_pnet(self):
        return self.pnet