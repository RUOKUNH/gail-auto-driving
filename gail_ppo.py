import pdb
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
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_config = train_config

        self.ppo = PPO(state_dim, action_dim, train_config)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim)

        self.args = args

        self.collectors = collectors

        self.env = TrafficSim(["../scenarios/ngsim"], envision=False)

    def collect(self, obs, acts, rwds, gammas, steps, speeds, max_step, pnet, gamma):
        obs1 = []
        acts1 = []
        rwds1 = []
        gammas1 = []
        speed = []
        step = 0
        ob = self.env.reset()
        done = False
        while step < max_step and not done:
            ob = expert_collector3(ob)
            ob = feature2(ob)
            speed.append(ob[1])
            act = pnet(ob).sample()
            act = list(act.cpu().numpy())

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

    def train(self, expert_path):
        best_score = -np.inf
        if self.args.con:
            model = torch.load('model' + self.args.exp + '.pth')
            self.ppo.pnet.load_state_dict(model['action_net'])
            self.ppo.collect_pnet.load_state_dict(model['action_net'])
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
        # obs, act
        # train_buffer = [[], []]
        # max_train_buffer = 2000
        # # beta = 0.99
        # beta = 1
        gen_obs_buffer = []
        gen_act_buffer = []
        gen_obs_buffer2 = []
        gen_act_buffer2 = []
        pkf = open(expert_path, 'rb')
        while True:
            try:
                expert_buffer.append(pkl.load(pkf))
            except EOFError:
                pkf.close()
                break
        random.shuffle(expert_buffer)
        exp_obs_buffer = []
        exp_act_buffer = []
        for exp_record in expert_buffer:
            obs = exp_record['observation']
            act = exp_record['actions']
            obs = [feature2(ob) for ob in obs]
            exp_obs_buffer += obs
            exp_act_buffer += list(act)
        batch_idx = 0
        _iter = 0
        t = time.time()
        while _iter < train_iter:
            _iter += 1

            buffer_update_record = []

            # Generate interactive data
            obs2 = []
            acts2 = []
            rwds2 = []
            gammas2 = []
            speeds = []
            _step = [0]
            collects = 0
            self.ppo.pnet.eval()
            while collects < self.collectors:
                try:
                    self.collect(obs2, acts2, rwds2, gammas2, _step, speeds, max_step, self.ppo.pnet, gamma)
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
            #     exp_obs = [feature2(ob) for ob in exp_obs]
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
                gen_obs_buffer += obs2[i]
                gen_act_buffer += acts2[i]
                gen_obs_buffer2.append(obs2[i])
                gen_act_buffer2.append(acts2[i])
            if len(gen_obs_buffer) > 10000:
                gen_obs_buffer = gen_obs_buffer[-10000:]
                gen_act_buffer = gen_act_buffer[-10000:]
            if len(gen_obs_buffer2) > 500:
                gen_obs_buffer2 = gen_obs_buffer2[-500:]
                gen_act_buffer2 = gen_act_buffer2[-500:]

            iters = len(gen_obs_buffer) // 100
            d_loss = 0
            np.random.seed()
            for _ in range(iters):
                gen_idx = np.random.randint(0, len(gen_obs_buffer), 200)
                exp_idx = np.random.randint(0, len(exp_obs_buffer), 200)
                expert_obs = torch.tensor(exp_obs_buffer[exp_idx])
                expert_act = torch.tensor(exp_act_buffer[exp_idx])
                gen_obs = torch.tensor(gen_obs_buffer[gen_idx])
                gen_act = torch.tensor(gen_act_buffer[gen_idx])
                self.d.train()
                exp_scores = self.d(expert_obs, expert_act)
                gen_scores = self.d(gen_obs, gen_act)
                loss_d = torch.nn.functional.binary_cross_entropy(
                    exp_scores, torch.ones_like(exp_scores)
                ) + torch.nn.functional.binary_cross_entropy(
                    gen_scores, torch.zeros_like(gen_scores)
                )
                if torch.isnan(loss_d):
                    continue
                d_loss += loss_d
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            d_loss /= iters

            # TD-update Value net
            self.d.eval()
            self.v.train()
            v_loss = 0
            iters = min(50, len(gen_obs_buffer2))
            gen_idx = np.random.randint(0, len(gen_obs_buffer2), iters)
            exp_idx = np.random.randint(0, len(expert_buffer), iters)
            for it in range(iters):
                gen_obs = torch.tensor(gen_obs_buffer2[gen_idx[it]])
                gen_act = torch.tensor(gen_act_buffer2[gen_idx[it]])
                costs = torch.log(1 - self.d(gen_obs, gen_act)).squeeze().detach()
                esti_rwds = -1 * costs
                # take real reward in use
                # esti_rwds = 0.8 * esti_rwds + 0.2 * generation_rwd[i]
                esti_v = self.v(gen_obs).view(-1)
                td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                loss_v = F.mse_loss(esti_v[:-1], td_v)
                if not torch.isnan(loss_v):
                    opt_v.zero_grad()
                    loss_v.backward()
                    opt_v.step()
                    v_loss += loss_v

                exp_sample = expert_buffer[exp_idx[it]]
                exp_obs = exp_sample['observation']
                exp_act = exp_sample['actions']
                exp_obs = [feature2(ob) for ob in exp_obs]
                exp_obs, exp_act = torch.tensor(exp_obs), torch.tensor(exp_act)
                costs = torch.log(1 - self.d(exp_obs, exp_act)).squeeze().detach()
                esti_rwds = -1 * costs
                esti_v = self.v(exp_obs).view(-1)
                td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                loss_v = F.mse_loss(esti_v[:-1], td_v)
                if not torch.isnan(loss_v):
                    opt_v.zero_grad()
                    loss_v.backward()
                    opt_v.step()
                    v_loss += loss_v

            v_loss /= (iters * 2)

            # PPO update Action net
            _update = self.ppo.update(obs2, acts2, gamma, self.v, self.d)

            rwds2 = [np.sum(rwd) for rwd in rwds2]

            plt.plot(np.arange(len(score_list)), score_list)
            plt.savefig('rwd' + self.args.exp + '.png')
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
                torch.save(state, 'bestmodel' + self.args.exp + '.pth')

            if _iter % 50 == 0:
                state = {'action_net': self.ppo.get_pnet().state_dict(),
                         'value_net': self.v.state_dict(),
                         'disc_net': self.d.state_dict(),
                         'net_dims': [self.ppo.get_pnet().net_dims,
                                      self.v.net_dims,
                                      self.d.net_dims]
                         }
                torch.save(state, 'model' + self.args.exp + '.pth')
                if not os.path.exists(f'scores{self.args.exp}.txt'):
                    score_dict = {self.args.exp: score_list}
                else:
                    with open(f'scores{self.args.exp}.txt', 'rb') as pkf:
                        score_dict = pkl.load(pkf)
                    score_dict[self.args.exp] = score_list
                with open(f'scores{self.args.exp}.txt', 'wb') as pkf:
                    pkl.dump(score_dict, pkf)

            t4 = time.time() - t
            t = time.time()

            print(
                f"{self.args.exp}, iter {_iter}: step{_step[0]}, ",
                "score: %.2f, %.2f, %.2f, " % (np.mean(rwds2), np.max(rwds2), np.min(rwds2)),
                "m_speed: %.2f, " % np.mean(speeds),
                "time: %.2f %.2f" % (t1, t4),
                "d_loss %.3f, v_loss %.3f, " % (d_loss, v_loss),
                f"ppo {_update}"
            )

        return self.ppo.get_pnet(), self.v, self.d
