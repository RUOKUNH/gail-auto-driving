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
            _heading = ob.ego_vehicle_state.heading
            ob = expert_collector3(ob)
            ob = feature(ob)
            speed.append(ob[1])
            act = pnet(ob).sample()
            act = list(act.cpu().numpy())
            # if _speed <= 0:
            #     act[0] = max(0, act[0])
            # if _heading <= -0.1:
            #     act[1] = max(0, act[1])
            # if _heading >= 0.1:
            #     act[1] = min(0, act[1])
            act[1] = 0

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

    def train(self, expert_path, feature, kld_limit, epoch, d_iters, v_iters, beta):
        if not os.path.exists(f'{self.args.exp}'):
            os.mkdir(f'{self.args.exp}')
        best_score = -np.inf
        score_list = []
        upper_score_list = []
        lower_score_list = []
        d_loss_list = []
        v_loss_list = []
        d_value_list = []

        train_iter = self.train_config['train_iter']
        max_step = self.train_config['max_step']
        lambda_ = self.train_config['lambda_']
        gamma = self.train_config['gamma']

        opt_d = torch.optim.Adam(self.d.parameters(), eps=1e-4)
        opt_v = torch.optim.Adam(self.v.parameters(), eps=1e-4)

        expert_buffer = []
        beta = beta
        gen_obs_buffer = torch.Tensor([])
        gen_act_buffer = torch.Tensor([])
        gen_obs_buffer2 = []
        gen_act_buffer2 = []
        collect_obs_buffer = []
        collect_act_buffer = []
        coach_buffer = []
        gen_coach_buffer = []
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
            upper_score_list.append(np.max([np.sum(rwd) for rwd in rwds2]))
            lower_score_list.append(np.min([np.sum(rwd) for rwd in rwds2]))

            for rwd in rwds2:
                rwd = np.array(rwd)
                last_steps = min(20, len(rwd))
                rwd[-last_steps:] -= np.linspace(0, np.sqrt(5), last_steps) ** 2

            # Update Dnet
            _rwds = [np.sum(rwd) for rwd in rwds2]
            # Coaching Dagger
            idx = np.argmax(_rwds)
            if _rwds[idx] > 100:
                gen_coach_buffer.append((obs2[idx], acts2[idx]))
                obs2.pop(idx)
                acts2.pop(idx)
            if len(gen_coach_buffer) > 50:
                gen_coach_buffer = gen_coach_buffer[-50:]
            d_value_buffer = []
            for i in range(len(obs2)):
                # gen_obs_buffer = torch.cat([gen_obs_buffer, torch.Tensor(obs2[i])])
                # gen_act_buffer = torch.cat([gen_act_buffer, torch.Tensor(acts2[i])])
                gen_obs_buffer2.append(obs2[i])
                gen_act_buffer2.append(acts2[i])
                # pdb.set_trace()
                d_vals = self.d(torch.FloatTensor(obs2[i]), torch.FloatTensor(acts2[i]))
                d_val = torch.mean(d_vals)
                if not (torch.isnan(d_val) or torch.isinf(d_val)):
                    d_value_buffer.append(float(d_val))

            coach_buffer = []
            for i in range(50):
                p = beta**_iter
                if np.random.random() > p and len(gen_coach_buffer) > 0:
                    idx = np.random.randint(len(gen_coach_buffer))
                    coach_buffer.append(gen_coach_buffer[idx])
                else:
                    idx = np.random.randint(len(expert_buffer))
                    coach_buffer.append(expert_buffer[idx])
            collect_act_buffer += acts2
            collect_obs_buffer += obs2
            # if len(gen_obs_buffer) > 1500:
            #     gen_obs_buffer = gen_obs_buffer[-1500:]
            #     gen_act_buffer = gen_act_buffer[-1500:]
            if len(gen_obs_buffer2) > 25:  # save 5 iteration gen data
                gen_obs_buffer2 = gen_obs_buffer2[-25:]
                gen_act_buffer2 = gen_act_buffer2[-25:]
            gen_obs_buffer = torch.cat([torch.Tensor(obs) for obs in gen_obs_buffer2])
            gen_act_buffer = torch.cat([torch.Tensor(act) for act in gen_act_buffer2])

            exp_obs_buffer = []
            exp_act_buffer = []
            while len(exp_obs_buffer) < 5000:
                obs, act = coach_buffer[np.random.randint(len(coach_buffer))]
                # obs = [feature(ob) for ob in obs]
                exp_obs_buffer += obs
                exp_act_buffer += act
            exp_obs_buffer = torch.Tensor(exp_obs_buffer)
            exp_act_buffer = torch.Tensor(exp_act_buffer)
            iters = d_iters
            d_loss = 0
            np.random.seed()
            self.d.train()
            for _ in range(iters):
                gen_idx = torch.from_numpy(np.random.randint(0, len(gen_obs_buffer), 200).astype(np.int64))
                exp_idx = torch.from_numpy(np.random.randint(0, len(exp_obs_buffer), 200).astype(np.int64))
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
            n_step = 5
            curr_obs_buffer = []
            nnext_obs_buffer = []
            nstep_discount_rwd_buffer = []
            for bf in range(len(gen_obs_buffer2)):
                gen_obs = torch.FloatTensor(gen_obs_buffer2[bf])
                gen_act = torch.FloatTensor(gen_act_buffer2[bf])
                gen_act[:, -1] *= 10
                costs = torch.log(1 - self.d(gen_obs, gen_act)).squeeze().detach()
                rewards = -1 * costs
                curr_obs_buffer.append(gen_obs[:-n_step, :])
                nnext_obs_buffer.append(gen_obs[n_step:, :])
                discount_rwd = rewards[:-n_step]
                for k in range(1, n_step):
                    discount_rwd += rewards[k: k - n_step] * gamma ** k
                nstep_discount_rwd_buffer.append(discount_rwd)
            for _ in range(15):
                exp_obs, exp_act = coach_buffer[np.random.randint(0, len(coach_buffer))]
                exp_obs, exp_act = torch.FloatTensor(exp_obs), torch.FloatTensor(exp_act)
                exp_act[:, -1] *= 10
                costs = torch.log(1 - self.d(exp_obs, exp_act)).squeeze().detach()
                rewards = -1 * costs
                curr_obs_buffer.append(exp_obs[:-n_step, :])
                nnext_obs_buffer.append(exp_obs[n_step:, :])
                discount_rwd = rewards[:-n_step]
                for k in range(1, n_step):
                    discount_rwd += rewards[k: k - n_step] * gamma ** k
                nstep_discount_rwd_buffer.append(discount_rwd)
            curr_obs_buffer, nnext_obs_buffer, nstep_discount_rwd_buffer = \
                torch.cat(curr_obs_buffer), torch.cat(nnext_obs_buffer), torch.cat(nstep_discount_rwd_buffer)
            curr_obs_buffer = curr_obs_buffer[~torch.isinf(nstep_discount_rwd_buffer)
                                              & ~torch.isnan(nstep_discount_rwd_buffer)]
            nnext_obs_buffer = nnext_obs_buffer[~torch.isinf(nstep_discount_rwd_buffer)
                                                & ~torch.isnan(nstep_discount_rwd_buffer)]
            nstep_discount_rwd_buffer = nstep_discount_rwd_buffer[
                ~torch.isinf(nstep_discount_rwd_buffer) & ~torch.isnan(nstep_discount_rwd_buffer)]

            v_loss = 0
            iters = v_iters
            for it in range(iters):
                train_idx = torch.from_numpy(np.random.randint(0, len(curr_obs_buffer), 200).astype(np.int64))
                curr_obs = curr_obs_buffer[train_idx]
                nnext_obs = nnext_obs_buffer[train_idx]
                nstep_discount_rwd = nstep_discount_rwd_buffer[train_idx]
                curr_val = self.v(curr_obs).view(-1)
                nnext_val = self.v(nnext_obs).view(-1)
                td_val = gamma ** n_step * nnext_val + nstep_discount_rwd
                loss_v = F.mse_loss(curr_val, td_val)
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
            if kld_limit or _iter < 10:
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
            plt.plot(np.arange(len(upper_score_list)), upper_score_list)
            plt.plot(np.arange(len(lower_score_list)), lower_score_list)
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
                                      self.d.net_dims],
                         'action_dim': self.action_dim}
                torch.save(state, f'{self.args.exp}/bestmodel{self.args.exp}.pth')

            if _iter % 50 == 0:
                state = {'action_net': self.ppo.get_pnet().state_dict(),
                         'value_net': self.v.state_dict(),
                         'disc_net': self.d.state_dict(),
                         'net_dims': [self.ppo.get_pnet().net_dims,
                                      self.v.net_dims,
                                      self.d.net_dims],
                         'action_dim': self.action_dim
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
