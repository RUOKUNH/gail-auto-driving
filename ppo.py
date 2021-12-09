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
        self.action_dim = action_dim
        self.state_dim = state_dim

    def L(self, dist, old_dist, acts, advs):
        adl = (advs * torch.exp(
            dist.log_prob(acts)
            - old_dist.log_prob(acts).detach()
        )).mean()
        return adl

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

    def rescale_and_line_search(self, old_dist, new_param, old_param, obs):
        d_param = new_param - old_param
        alpha = 1.0
        for _ in range(10):
            new_param = old_param + alpha * d_param
            set_params(self.pnet, new_param)
            dist = self.pnet(obs)
            kld = kl_divergence(dist, old_dist, self.action_dim)
            if kld < 0.1:
                return
            else:
                alpha *= 0.7
        set_params(self.pnet, old_param)
        print('step too large')


    def update(self, obs, acts, gamma, vnet, dnet, epoches=5, kld_limit=False):
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
            act[:, -1] *= 10
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
        for epoch in range(epoches):
            random.shuffle(update_data)
            st = 0
            self.pnet.train()
            self.collect_pnet.eval()
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
                if len(_obs) < 2:
                    break
                _obs, _acts = FloatTensor(_obs), FloatTensor(_acts)
                dist = self.pnet(_obs)
                old_dist = self.collect_pnet(_obs)
                _ratio = torch.exp(dist.log_prob(_acts) - old_dist.log_prob(_acts).detach())
                loss, _grad = self.L_clip(_advs, _ratio)
                # dist_causal_entropy = self.config['lambda_'] * (-1 * self.pnet(_obs).log_prob(_acts)).mean()
                # loss -= dist_causal_entropy
                if not _grad:
                    st = ed
                    continue
                if torch.isnan(loss) or torch.isinf(loss):
                    st = ed
                    continue
                old_param = get_flat_params(self.pnet)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                new_param = get_flat_params(self.pnet)
                if kld_limit:
                    self.rescale_and_line_search(dist, new_param, old_param, _obs)
                st = ed
        #########################
        if self.synchronize_step >= self.synchronize_steps:
            self.collect_pnet.load_state_dict(self.pnet.state_dict())
            self.synchronize_step = 0
            synchronize = 1
        else:
            synchronize = 0

        return synchronize

    def get_pnet(self):
        return self.pnet