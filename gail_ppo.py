import pdb

import torch

from utils import *
from net import *
from ppo import PPO
import matplotlib.pyplot as plt
import pickle as pkl
import random
import time
import os
import torch.nn.functional as F
from threading import Lock
from traffic_simulator import TrafficSim


class GAIL_PPO:
    def __init__(
            self,
            state_dim,
            action_dim,
            train_config,
            args,
            collectors=5,
            synchronize=1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_config = train_config

        self.ppo = PPO(state_dim, action_dim, train_config, synchronize_steps=synchronize)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim)

        self.args = args

        self.collectors = collectors

        self.env = TrafficSim(["../scenarios/ngsim"], envision=False)

    def collect(self, obs, acts, rwds, gammas, steps, speeds, max_step, pnet, gamma, feature):
        obs1 = []
        acts1 = []
        rwds1 = []
        gammas1 = []
        speed = []
        step = 0
        ob = self.env.reset()
        done = False
        while step < max_step and not done:
            # pdb.set_trace()
            _speed = ob.ego_vehicle_state.speed
            ob = expert_collector3(ob)
            ob = feature(ob)
            speed.append(ob[1])
            act = pnet(ob).sample()
            act = list(act.cpu().numpy())
            if _speed <= 0:
                act[0] = max(0, act[0])

            obs1.append(ob)
            acts1.append(act)

            ob, r, done, _ = self.env.step(act)

            rwds1.append(r)
            gammas1.append(gamma ** step)

            step += 1

        # self.collector_lock.acquire()
        obs.append(obs1)
        acts.append(acts1)
        rwds.append(rwds1)
        gammas.append(gammas1)
        steps[0] += step
        speeds += speed
        # self.collector_lock.release()

    def train(self, expert_path, feature, kld_limit, epoch):
        if not os.path.exists(f'{self.args.exp}'):
            os.mkdir(f'{self.args.exp}')
        best_score = -np.inf
        # if self.args.con:
        #     model = torch.load('model' + self.args.exp + '.pth')
        #     self.ppo.pnet.load_state_dict(model['action_net'])
        #     self.ppo.collect_pnet.load_state_dict(model['action_net'])
        #     self.v.load_state_dict(model['value_net'])
        #     self.d.load_state_dict(model['disc_net'])
        #     with open('scores.txt', 'rb') as pkf:
        #         scores_dict = pkl.load(pkf)
        #         score_list = scores_dict[self.args.exp]
        # else:
        #     score_list = []
        score_list = []
        d_loss_list = []
        v_loss_list = []
        d_value_list = []

        train_iter = self.train_config['train_iter']
        max_step = self.train_config['max_step']
        lambda_ = self.train_config['lambda_']
        gamma = self.train_config['gamma']

        opt_d = torch.optim.Adam(self.d.parameters())
        opt_v = torch.optim.Adam(self.v.parameters())

        expert_buffer = []
        # obs, act
        # train_buffer = [[], []]
        # max_train_buffer = 2000
        # # beta = 0.99
        # beta = 1
        gen_obs_buffer = torch.Tensor([])
        gen_act_buffer = torch.Tensor([])
        gen_obs_buffer2 = []
        gen_act_buffer2 = []
        collect_obs_buffer = []
        collect_act_buffer = []
        pkf = open(expert_path, 'rb')
        while True:
            try:
                record = pkl.load(pkf)
                expert_obs = record['observation']
                expert_act = record['actions']
                # expert_obs = [feature(ob) for ob in expert_obs]
                expert_buffer.append((expert_obs, expert_act))
            except EOFError:
                pkf.close()
                break
        random.shuffle(expert_buffer)
        # pdb.set_trace()
        print('finish loading data')
        batch_idx = 0
        _iter = 0
        t = time.time()
        while _iter < train_iter:
            _iter += 1
            # Generate interactive data
            obs2 = []
            acts2 = []
            rwds2 = []
            gammas2 = []
            speeds = []
            _step = [0]
            collects = 0
            self.ppo.collect_pnet.eval()
            while collects < self.collectors:
                try:
                    self.collect(obs2, acts2, rwds2, gammas2, _step, speeds, max_step, self.ppo.collect_pnet, gamma,
                                 feature)
                    collects += 1
                except:
                    continue

            t1 = time.time() - t
            t = time.time()

            score_list.append(np.mean([np.sum(rwd) for rwd in rwds2]))

            for rwd in rwds2:
                rwd = np.array(rwd)
                last_steps = min(20, len(rwd))
                rwd[-last_steps:] -= np.linspace(0, np.sqrt(5), last_steps) ** 2

            # pdb.set_trace()

            # Dagger
            # while len(train_buffer[0]) >= max_train_buffer:
            #     pop_idx = random.randint(0, len(train_buffer[0])-1)
            #     train_buffer[0].pop(pop_idx)
            #     train_buffer[1].pop(pop_idx)
            # sample_num = 10
            # expert_ratio = beta ** (_iter - 1)
            # expert_samples = int(np.ceil(sample_num * expert_ratio))
            # generation_samples = min(sample_num - expert_samples, self.collectors)
            # for i in range(expert_samples):
            #     sample = expert_buffer[batch_idx]
            #     exp_obs = sample['observation']
            #     exp_obs = [feature(ob) for ob in exp_obs]
            #     train_buffer[0].append(exp_obs)
            #     train_buffer[1].append(sample['actions'])
            #     batch_idx += 1
            #     if batch_idx >= len(expert_buffer):
            #         batch_idx = 0
            # sample_count = 0
            # for i in range(self.collectors):
            #     if sample_count == generation_samples:
            #         break
            #     # if np.sum(rwds2[i]) > 150:
            #     train_buffer[0].append(obs2[i])
            #     train_buffer[1].append(acts2[i])
            #     sample_count += 1
            # buffer_update_record.append(sample_count + expert_samples)
            # generation_obs = []
            # generation_act = []
            # for i in range(self.collectors):
            #     generation_obs.append(torch.FloatTensor(obs2[i]))
            #     generation_act.append(torch.FloatTensor(acts2[i]))
            # generation_rwd = [torch.FloatTensor(rwd) for rwd in rwds2]
            # train_samples = min(len(train_buffer[0]), 10)
            # expert_sample_idx = np.random.randint(0, len(train_buffer[0]), train_samples)
            # expert_obs = []
            # expert_act = []
            # for i in expert_sample_idx:
            #     expert_obs += train_buffer[0][i]
            #     expert_act += list(train_buffer[1][i])

            # Update Dnet

            for i in range(self.collectors):
                gen_obs_buffer = torch.cat([gen_obs_buffer, torch.Tensor(obs2[i])])
                gen_act_buffer = torch.cat([gen_act_buffer, torch.Tensor(acts2[i])])
                gen_obs_buffer2.append(obs2[i])
                gen_act_buffer2.append(acts2[i])
            collect_act_buffer += acts2
            collect_obs_buffer += obs2
            if len(gen_obs_buffer) > 5000:
                gen_obs_buffer = gen_obs_buffer[-5000:]
                gen_act_buffer = gen_act_buffer[-5000:]
            if len(gen_obs_buffer2) > 500:
                gen_obs_buffer2 = gen_obs_buffer2[-500:]
                gen_act_buffer2 = gen_act_buffer2[-500:]

            exp_obs_buffer = []
            exp_act_buffer = []
            while len(exp_obs_buffer) < 5000:
                obs, act = expert_buffer[np.random.randint(len(expert_buffer))]
                # obs = [feature(ob) for ob in obs]
                exp_obs_buffer += obs
                exp_act_buffer += act
            exp_obs_buffer = torch.Tensor(exp_obs_buffer)
            exp_act_buffer = torch.Tensor(exp_act_buffer)
            iters = min(len(gen_obs_buffer) // 500 + 1, 25)
            d_loss = 0
            np.random.seed()
            self.d.train()
            for _ in range(iters):
                gen_idx = torch.from_numpy(np.random.randint(0, len(gen_obs_buffer), 200).astype(np.int64))
                exp_idx = torch.from_numpy(np.random.randint(0, len(exp_obs_buffer), 200).astype(np.int64))
                expert_obs = exp_obs_buffer[exp_idx].clone()
                expert_obs = exp_obs_buffer[exp_idx].clone()
                expert_act = exp_act_buffer[exp_idx].clone()
                gen_obs = gen_obs_buffer[gen_idx].clone()
                gen_act = gen_act_buffer[gen_idx].clone()
                expert_act[:, -1] *= 10
                gen_act[:, -1] *= 10
                exp_scores = self.d(expert_obs, expert_act)
                gen_scores = self.d(gen_obs, gen_act)
                loss_d = torch.nn.functional.binary_cross_entropy(
                    exp_scores, torch.ones_like(exp_scores)
                ) + torch.nn.functional.binary_cross_entropy(
                    gen_scores, torch.zeros_like(gen_scores)
                )
                if torch.isnan(loss_d) or torch.isinf(loss_d):
                    continue
                d_loss += loss_d
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            d_loss /= iters
            d_loss *= 1000
            d_loss_list.append(d_loss)

            # TD-update Value net
            self.d.eval()
            self.v.train()
            v_loss = 0
            iters = min(25, len(gen_obs_buffer2) // 10 + 1)
            gen_idx = np.random.randint(0, len(gen_obs_buffer2), iters)
            exp_idx = np.random.randint(0, len(expert_buffer), iters)
            d_value_buffer = []
            for it in range(iters):
                gen_obs = torch.FloatTensor(gen_obs_buffer2[gen_idx[it]])
                gen_act = torch.FloatTensor(gen_act_buffer2[gen_idx[it]])
                gen_act[:, -1] *= 10
                costs = torch.log(1 - self.d(gen_obs, gen_act)).squeeze().detach()
                esti_rwds = -1 * costs
                if not (torch.isnan(torch.mean(esti_rwds)) or torch.isinf(torch.mean(esti_rwds))):
                    d_value_buffer.append(torch.mean(esti_rwds).detach().numpy())
                # take real reward in use
                # esti_rwds = 0.8 * esti_rwds + 0.2 * generation_rwd[i]
                esti_v = self.v(gen_obs).view(-1)
                td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                loss_v = F.mse_loss(esti_v[:-1], td_v)
                if not (torch.isnan(loss_v) or torch.isinf(loss_v)):
                    opt_v.zero_grad()
                    loss_v.backward()
                    opt_v.step()
                    v_loss += loss_v

                exp_obs, exp_act = expert_buffer[exp_idx[it]]
                # exp_obs = [feature(ob) for ob in exp_obs]
                exp_obs, exp_act = torch.FloatTensor(exp_obs), torch.FloatTensor(exp_act)
                exp_act[:, -1] *= 10
                costs = torch.log(1 - self.d(exp_obs, exp_act)).squeeze().detach()
                esti_rwds = -1 * costs
                esti_v = self.v(exp_obs).view(-1)
                td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                loss_v = F.mse_loss(esti_v[:-1], td_v)
                if not (torch.isnan(loss_v) or torch.isinf(loss_v)):
                    opt_v.zero_grad()
                    loss_v.backward()
                    opt_v.step()
                    v_loss += loss_v

            if len(d_value_buffer) > 0:
                d_value_list.append(np.mean(d_value_buffer))

            v_loss /= (iters * 2)
            v_loss *= 1000
            v_loss_list.append(v_loss)

            # PPO update Action net
            if _iter <= 10:
                synchronize = self.ppo.update(collect_obs_buffer, collect_act_buffer, gamma, self.v, self.d,
                                              kld_limit=True, epoches=epoch)
            else:
                synchronize = self.ppo.update(collect_obs_buffer, collect_act_buffer, gamma, self.v, self.d,
                                              kld_limit=False, epoches=epoch)

            if synchronize:
                collect_act_buffer = []
                collect_obs_buffer = []

            rwds2 = [np.sum(rwd) for rwd in rwds2]

            plt.plot(np.arange(len(score_list)), score_list)
            plt.savefig(f'{self.args.exp}/rwd.png')
            plt.close()
            plt.plot(np.arange(len(d_loss_list)), d_loss_list)
            plt.savefig(f'{self.args.exp}/dloss.png')
            plt.close()
            plt.plot(np.arange(len(v_loss_list)), v_loss_list)
            plt.savefig(f'{self.args.exp}/vloss.png')
            plt.close()
            plt.plot(np.arange(min(100, len(d_loss_list[-100:]))), d_loss_list[-100:])
            plt.savefig(f'{self.args.exp}/newdloss.png')
            plt.close()
            plt.plot(np.arange(min(100, len(v_loss_list[-100:]))), v_loss_list[-100:])
            plt.savefig(f'{self.args.exp}/newvloss.png')
            plt.close()
            plt.plot(np.arange(len(d_value_list)), d_value_list)
            plt.savefig(f'{self.args.exp}/dvalue.png')
            plt.close()

            # state = {'action_net': self.ppo.get_pnet().state_dict(),
            #          'value_net': self.v.state_dict(),
            #          'disc_net': self.d.state_dict()}
            # saved_model.append(state)
            # if len(saved_model) > 20:
            #     saved_model.pop(0)
            # if _iter > 30 and np.mean(score_list[-5:]) < 60 and self.args.revert:
            #     # revert
            #     revert_count += 1
            #     print("############")
            #     print("revert")
            #     print("############")
            #     score_list = score_list[:-20]
            #     train_buffer[0] = train_buffer[0][:-np.sum(buffer_update_record[-20:])]
            #     train_buffer[1] = train_buffer[1][:-np.sum(buffer_update_record[-20:])]
            #     _iter -= 20
            #     state = saved_model[0]
            #     self.ppo.pnet.load_state_dict(state['action_net'])
            #     self.ppo.collect_pnet.load_state_dict(state['action_net'])
            #     self.pi = self.ppo.collect_pnet
            #     self.v.load_state_dict(state['value_net'])
            #     self.d.load_state_dict(state['disc_net'])
            #     continue

            if np.mean(score_list[-self.collectors:]) > best_score:
                best_score = np.mean(score_list[-self.collectors:])
                state = {'action_net': self.ppo.get_pnet().state_dict(),
                         'value_net': self.v.state_dict(),
                         'disc_net': self.d.state_dict(),
                         'net_dims': [self.ppo.get_pnet().net_dims,
                                      self.v.net_dims,
                                      self.d.net_dims]}
                torch.save(state, f'{self.args.exp}/bestmodel{self.args.exp}.pth')

            if _iter % 50 == 0:
                state = {'action_net': self.ppo.get_pnet().state_dict(),
                         'value_net': self.v.state_dict(),
                         'disc_net': self.d.state_dict(),
                         'net_dims': [self.ppo.get_pnet().net_dims,
                                      self.v.net_dims,
                                      self.d.net_dims]
                         }
                torch.save(state, f'{self.args.exp}/model{self.args.exp}.pth')
                # if not os.path.exists(f'scores{self.args.exp}.txt'):
                #     score_dict = {self.args.exp: score_list}
                # else:
                #     with open(f'scores{self.args.exp}.txt', 'rb') as pkf:
                #         score_dict = pkl.load(pkf)
                #     score_dict[self.args.exp] = score_list
                # with open(f'scores{self.args.exp}.txt', 'wb') as pkf:
                #     pkl.dump(score_dict, pkf)

            t4 = time.time() - t
            t = time.time()

            print(
                f"{self.args.exp}, iter {_iter}: step{_step[0]}, ",
                "score: %.2f, %.2f, %.2f, " % (np.mean(rwds2), np.max(rwds2), np.min(rwds2)),
                "m_speed: %.2f, " % np.mean(speeds),
                "time: %.2f %.2f" % (t1, t4),
                "dloss %.3f, vloss %.3f, dvalue %.3f" % (d_loss, v_loss, d_value_list[-1]),
            )
