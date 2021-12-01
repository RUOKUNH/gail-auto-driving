import pdb

import torch

from utils import *
from net import *
from trpo import TRPO
import matplotlib.pyplot as plt
import pickle as pkl
import random
# from tqdm import trange, tqdm
import time
import os
import torch.nn.functional as F


class GAIL:
    def __init__(
            self,
            state_dim,
            action_dim,
            train_config,
            args
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim)

        self.trpo = TRPO(train_config['kld_thre'], action_dim, state_dim)
        self.args = args

    def train(self, env, expert_path, render=False):
        if self.args.con:
            model = torch.load('model' + self.args.exp + '.pth')
            self.pi.load_state_dict(model['action_net'])
            self.v.load_state_dict(model['value_net'])
            self.d.load_state_dict(model['disc_net'])
            with open('scores.txt', 'rb') as pkf:
                scores_dict = pkl.load(pkf)
                score_list = scores_dict[self.args.exp]
        else:
            score_list = []

        train_iter = self.train_config['train_iter']
        max_step = self.train_config['max_step']
        lambda_ = self.train_config['lambda_']
        gamma = self.train_config['gamma']

        opt_d = torch.optim.Adam(self.d.parameters())
        opt_v = torch.optim.Adam(self.v.parameters())

        expert_buffer = []
        pkf = open(expert_path, 'rb')
        while True:
            try:
                expert_buffer.append(pkl.load(pkf))
            except EOFError:
                pkf.close()
                break
        random.shuffle(expert_buffer)
        batch_idx = 0
        batch_size = 50
        _iter = 0
        t = time.time()
        while _iter < train_iter:
            try:
                # old_params = [get_flat_params(self.pi).detach(),
                #               get_flat_params(self.v).detach(),
                #               get_flat_params(self.d).detach(),]
                _iter += 1

                # Generate interactive data
                obs = []
                acts = []
                rwds = []
                gammas = []
                speeds = []
                step = 0
                ob = env.reset()
                done = False
                while step < max_step and not done:
                    ob = make_obs(ob)
                    # ob = ob[:6]
                    ob = make_obs_2(ob)
                    speeds.append(ob[1])
                    act = self.pi(ob).sample()
                    act = list(act.cpu().numpy())

                    obs.append(ob)
                    acts.append(act)

                    ob, r, done, _ = env.step(act)

                    rwds.append(r)  # real rewards, not used to train
                    gammas.append(gamma ** step)

                    step += 1

                t1 = time.time() - t
                t = time.time()

                score_list.append(np.sum(rwds))
                obs = FloatTensor(obs)
                acts = FloatTensor(acts)
                gammas = torch.FloatTensor(gammas)

                # Get expert data batch
                if batch_idx + batch_size - 1 > len(expert_buffer):
                    batch_buffer = expert_buffer[batch_idx:]
                    batch_idx = 0
                else:
                    batch_buffer = expert_buffer[batch_idx:batch_idx + batch_size]
                    batch_idx += batch_size

                for record in batch_buffer:
                    # GT-update Value net
                    self.d.eval()
                    costs = torch.log(self.d(obs, acts)).squeeze().detach()
                    # costs += speed_penalty
                    esti_rwds = -1 * costs
                    dist_rwds = esti_rwds * gammas
                    gts = torch.FloatTensor([sum(dist_rwds[i:]) / gammas[i]
                                             for i in range(dist_rwds.size()[0])])
                    self.v.train()
                    esti_v = self.v(obs).view(-1)
                    # loss = torch.mean((gts - esti_v) ** 2)
                    # opt_v.zero_grad()
                    # loss.backward(retain_graph=True)
                    # opt_v.step()

                    # TD Update
                    td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                    loss_v = F.mse_loss(esti_v[:-1], td_v)
                    opt_v.zero_grad()
                    loss_v.backward(retain_graph=True)
                    opt_v.step()

                    # update D net
                    self.d.train()
                    exp_obs = record['observation']
                    exp_acts = record['actions']
                    exp_obs = [make_obs_2(ob) for ob in exp_obs]
                    exp_obs = FloatTensor(exp_obs)
                    # exp_obs = exp_obs[:, :6]
                    # pdb.set_trace()
                    exp_acts = FloatTensor(exp_acts)

                    # score expert close to 0
                    exp_scores = self.d(exp_obs, exp_acts)
                    # score generator close to 1
                    gen_scores = self.d(obs, acts)

                    loss_d = torch.nn.functional.binary_cross_entropy(
                        exp_scores, torch.zeros_like(exp_scores)
                    ) + torch.nn.functional.binary_cross_entropy(
                        gen_scores, torch.ones_like(gen_scores)
                    )
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()

                t2 = time.time() - t
                t = time.time()

                # TRPO-update Action net
                # Get advantage values
                self.d.eval()
                costs = torch.log(self.d(obs, acts)).squeeze().detach()
                esti_rwds = -1 * costs
                self.v.eval()
                curr_val = self.v(obs).detach().view(-1)
                next_val = curr_val[1:]
                curr_val = curr_val[:-1]
                # pdb.set_trace()
                advs = esti_rwds[:-1] \
                       + gamma * next_val \
                       - curr_val
                # pdb.set_trace()
                dist_causal_entropy = (-1 * self.pi(obs).log_prob(acts)).mean()
                grad_dist_causal_entropy = get_flat_grads(
                    dist_causal_entropy, self.pi
                )

                flag = self.trpo.step(
                    self.pi,
                    obs[:-1],
                    acts[:-1],
                    advs,
                    extra_gradient=-grad_dist_causal_entropy * lambda_
                )

                if flag == -1:
                    print('fail trpo-update')
                t3 = time.time() - t
                t = time.time()

                print(
                    f"{self.args.exp}, reward at iter {_iter}: step{step}, ",
                    "score: %.2f, m_speed: %.2f" % (np.sum(rwds), np.mean(speeds)),
                    "time: %.2f, %.2f, %.2f" % (t1, t2, t3),
                    "last heading %.4f" % obs[-1, 0],
                    "d_loss %.3f, v_loss %.3f" % (loss_d, loss_v)
                )

                if _iter % 100 == 0:
                    plt.plot(np.arange(len(score_list)), score_list)
                    plt.savefig('rwd' + self.args.exp + '.png')
                    plt.close()
                    state = {'action_net': self.pi.state_dict(),
                             'value_net': self.v.state_dict(),
                             'disc_net': self.d.state_dict()}
                    torch.save(state, 'model' + self.args.exp + '.pth')
                    if not os.path.exists('scores.txt'):
                        score_dict = {self.args.exp: score_list}
                    else:
                        with open('scores.txt', 'rb') as pkf:
                            score_dict = pkl.load(pkf)
                        score_dict[self.args.exp] = score_list
                    with open('scores.txt', 'wb') as pkf:
                        pkl.dump(score_dict, pkf)

            except KeyboardInterrupt:
                exit()

        return self.pi, self.v, self.d
