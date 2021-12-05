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
import threading
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

        self.pi = self.ppo.collect_pnet
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim)

        self.args = args

        self.collectors = collectors

        self.collector_lock = Lock()

    def collect(self, obs, acts, rwds, gammas, steps, speeds, max_step, pnet, gamma):
        obs1 = []
        acts1 = []
        rwds1 = []
        gammas1 = []
        speed = []
        step = 0
        ob = self.env.reset()
        done = False
        # pdb.set_trace()
        while step < max_step and not done:
            ob = expert_collector(ob)
            ob = new_feature_detection(ob)
            speed.append(ob[1])
            act = pnet(ob).sample()
            act = list(act.cpu().numpy())

            obs1.append(ob)
            acts1.append(act)

            ob, r, done, _ = self.env.step(act)

            rwds1.append(r)
            gammas1.append(gamma ** step)

            step += 1
            # print(step)

        # self.collector_lock.acquire()
        obs.append(obs1)
        acts.append(acts1)
        rwds.append(rwds1)
        gammas.append(gammas1)
        steps[0] += step
        speeds += speed
        # self.collector_lock.release()

    def train(self, expert_path, render=False):
        best_score = -np.inf
        self.env = TrafficSim(["../scenarios/ngsim"], envision=False)
        # self.env = TrafficSim(["./ngsim"], envision=False)
        if self.args.con:
            model = torch.load('model' + self.args.exp + '.pth')
            self.ppo.pnet.load_state_dict(model['action_net'])
            self.ppo.collect_pnet.load_state_dict(model['action_net'])
            self.pi =self.ppo.collect_pnet
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
        train_buffer = [[], []]
        max_train_buffer = 2000
        beta = 0.99
        pkf = open(expert_path, 'rb')
        while True:
            try:
                expert_buffer.append(pkl.load(pkf))
            except EOFError:
                pkf.close()
                break
        random.shuffle(expert_buffer)
        batch_idx = 0
        revert_count = 0
        _iter = 0
        t = time.time()
        while _iter < train_iter:
            # try:
            # old_params = [get_flat_params(self.pi).detach(),
            #               get_flat_params(self.v).detach(),
            #               get_flat_params(self.d).detach(),]
            _iter += 1

            saved_model = []
            buffer_update_record = []

            # Generate interactive data
            obs2 = []
            acts2 = []
            rwds2 = []
            gammas2 = []
            speeds = []
            _step = [0]
            collects = 0
            # self.t = time.time()
            while collects < self.collectors:
                try:
                    self.collect(obs2, acts2, rwds2, gammas2, _step, speeds, max_step, self.pi, gamma)
                    collects += 1
                except:
                    continue

            # t1 = time.time() - t
            # t = time.time()

            # score_list += [np.sum(rwd) for rwd in rwds2]
            score_list.append(np.mean([np.sum(rwd) for rwd in rwds2]))

            for rwd in rwds2:
                rwd = np.array(rwd)
                last_steps = min(20, len(rwd))
                rwd[-last_steps:] -= np.linspace(0, np.sqrt(5), last_steps)**2

            # Dagger get train data for discriminator
            # Update Dnet
            while len(train_buffer[0]) >= max_train_buffer:
                pop_idx = random.randint(0, len(train_buffer[0])-1)
                train_buffer[0].pop(pop_idx)
                train_buffer[1].pop(pop_idx)
            sample_num = 10
            expert_ratio = beta ** (_iter - 1)
            expert_samples = int(np.ceil(sample_num * expert_ratio))
            generation_samples = min(sample_num - expert_samples, self.collectors)
            for i in range(expert_samples):
                sample = expert_buffer[batch_idx]
                exp_obs = sample['observation']
                exp_obs = [new_feature_detection(ob) for ob in exp_obs]
                train_buffer[0].append(exp_obs)
                train_buffer[1].append(sample['actions'])
                batch_idx += 1
                if batch_idx >= len(expert_buffer):
                    batch_idx = 0
            sample_count = 0
            for i in range(self.collectors):
                if sample_count == generation_samples:
                    break
                # if np.sum(rwds2[i]) > 150:
                train_buffer[0].append(obs2[i])
                train_buffer[1].append(acts2[i])
                sample_count += 1
            buffer_update_record.append(sample_count + expert_samples)
            generation_obs = []
            generation_act = []
            for i in range(self.collectors):
                generation_obs.append(torch.FloatTensor(obs2[i]))
                generation_act.append(torch.FloatTensor(acts2[i]))
            # generation_obs = FloatTensor(generation_obs)
            # generation_act = FloatTensor(generation_act)
            generation_rwd = [torch.FloatTensor(rwd) for rwd in rwds2]
            train_samples = min(len(train_buffer[0]), 10)
            expert_sample_idx = np.random.randint(0, len(train_buffer[0]), train_samples)
            expert_obs = []
            expert_act = []
            for i in expert_sample_idx:
                # pdb.set_trace()
                expert_obs += train_buffer[0][i]
                expert_act += list(train_buffer[1][i])
            # pdb.set_trace()
            # expert_obs = FloatTensor(expert_obs)
            # expert_act = FloatTensor(expert_act)
            # exp_scores = self.d(expert_obs, expert_act)
            # gen_scores = torch.cat(
            #     [self.d(generation_obs[i], generation_act[i]) for i in range(len(generation_obs))]
            # )

            gen_obs = torch.cat(generation_obs)
            gen_act = torch.cat(generation_act)
            exp_samples = len(expert_obs) // 10 + 1
            gen_samples = len(gen_obs) // 10 + 1
            self.d.train()
            for i in range(10):
                _expert_obs = FloatTensor(expert_obs[i*exp_samples : (i+1)*exp_samples])
                _expert_act = FloatTensor(expert_act[i*exp_samples : (i+1)*exp_samples])
                _gen_obs = gen_obs[i*gen_samples : (i+1)*gen_samples]
                _gen_act = gen_act[i*gen_samples : (i+1)*gen_samples]
                exp_scores = self.d(_expert_obs, _expert_act)
                gen_scores = self.d(_gen_obs, _gen_act)
                loss_d = torch.nn.functional.binary_cross_entropy(
                    exp_scores, torch.zeros_like(exp_scores)
                ) + torch.nn.functional.binary_cross_entropy(
                    gen_scores, torch.ones_like(gen_scores)
                )
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            # TD-update Value net
            self.d.eval()
            self.v.train()
            for i in range(len(generation_obs)):
                costs = torch.log(self.d(generation_obs[i], generation_act[i])+1e-8).squeeze().detach()
                esti_rwds = -1 * costs
                # take real reward in use
                esti_rwds = 0.8 * esti_rwds + 0.2 * generation_rwd[i]

                esti_v = self.v(generation_obs[i]).view(-1)
                td_v = esti_rwds[:-1] + gamma * esti_v[1:]
                loss_v = F.mse_loss(esti_v[:-1], td_v)
                opt_v.zero_grad()
                loss_v.backward()
                opt_v.step()

            # t2 = time.time() - t
            # t = time.time()

            # PPO update Action net
            self.pi, _update = self.ppo.update(obs2, acts2, gamma, self.v, self.d)

            # t3 = time.time() - t
            # t = time.time()

            rwds2 = [np.sum(rwd) for rwd in rwds2]

            plt.plot(np.arange(len(score_list)), score_list)
            plt.savefig('rwd' + self.args.exp + '.png')
            plt.close()

            state = {'action_net': self.ppo.get_pnet().state_dict(),
                     'value_net': self.v.state_dict(),
                     'disc_net': self.d.state_dict()}
            saved_model.append(state)
            if len(saved_model) > 20:
                saved_model.pop(0)
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
                         'disc_net': self.d.state_dict()}
                torch.save(state, 'bestmodel' + self.args.exp + '.pth')

            if _iter % 50 == 0:
                state = {'action_net': self.ppo.get_pnet().state_dict(),
                         'value_net': self.v.state_dict(),
                         'disc_net': self.d.state_dict()}
                torch.save(state, 'model' + self.args.exp + '.pth')
                if not os.path.exists(f'scores{self.args.exp}.txt'):
                    score_dict = {self.args.exp: score_list}
                else:
                    with open(f'scores{self.args.exp}.txt', 'rb') as pkf:
                        score_dict = pkl.load(pkf)
                    score_dict[self.args.exp] = score_list
                with open(f'scores{self.args.exp}.txt', 'wb') as pkf:
                    pkl.dump(score_dict, pkf)

            # except KeyboardInterrupt:
            #     exit()
            t4 = time.time() - t
            t = time.time()

            print(
                f"{self.args.exp}, reward at iter {_iter}: step{_step[0]}, ",
                "score: %.2f, %.2f, %.2f, " % (np.mean(rwds2), np.max(rwds2), np.min(rwds2)),
                "m_speed: %.2f, " % np.mean(speeds),
                "time: %.2f" % t4,
                "d_loss %.3f, v_loss %.3f, " % (loss_d, loss_v),
                f"revert {revert_count}, ppo {_update}"
            )


        return self.ppo.get_pnet(), self.v, self.d
