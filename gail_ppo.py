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

        self.pi = PolicyNetwork(self.state_dim, self.action_dim)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim)

        self.ppo = PPO(self.pi, train_config)

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
        while step < max_step and not done:
            ob = make_obs(ob)
            ob = make_obs_2(ob)
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

    def train(self, expert_path, render=False):
        best_score = -np.inf
        # self.env = TrafficSim(["../scenarios/ngsim"], envision=False, collectors=self.collectors)
        self.env = TrafficSim(["./ngsim"], envision=False)
        if self.args.con:
            model = torch.load('model' + self.args.exp + '.pth')
            self.pi.load_state_dict(model['action_net'])
            self.v.load_state_dict(model['value_net'])
            self.d.load_state_dict(model['disc_net'])
            self.ppo = PPO(self.pi, self.train_config)
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
        train_buffer = ([], [])
        max_train_buffer = 2000
        beta = 0.995
        pkf = open(expert_path, 'rb')
        while True:
            try:
                expert_buffer.append(pkl.load(pkf))
            except EOFError:
                pkf.close()
                break
        random.shuffle(expert_buffer)
        batch_idx = 0
        batch_size = 10
        _iter = 0
        t = time.time()
        while _iter < train_iter:
            # try:
            # old_params = [get_flat_params(self.pi).detach(),
            #               get_flat_params(self.v).detach(),
            #               get_flat_params(self.d).detach(),]
            _iter += 1

            # Generate interactive data
            obs2 = []
            acts2 = []
            rwds2 = []
            gammas2 = []
            speeds = []
            _step = [0]
            # collect_threads = []
            # for i in range(self.collectors):
            #     # init_ob = self.env.reset(i)
            #     th = threading.Thread(target=self.collect,
            #     args=(obs2, acts2, rwds2, gammas2, _step, speeds, i, max_step, self.pi, gamma))
            #     collect_threads.append(th)
            # for th in collect_threads:
            #     th.start()
            # for th in collect_threads:
            #     th.join()
            # for i in range(self.collectors):
            #     self.collect(obs2, acts2, rwds2, gammas2, _step, speeds, i, max_step, self.pi, gamma)
            for i in range(self.collectors):
                self.collect(obs2, acts2, rwds2, gammas2, _step, speeds, max_step, self.pi, gamma)
            t1 = time.time() - t
            t = time.time()

            score_list += [np.sum(rwd) for rwd in rwds2]

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
            expert_ratio = beta ** (_iter - 1)
            expert_samples = int(np.ceil(10 * expert_ratio))
            generation_samples = min(10 - expert_samples, self.collectors)
            for i in range(expert_samples):
                sample = expert_buffer[batch_idx]
                exp_obs = sample['observation']
                exp_obs = [make_obs_2(ob) for ob in exp_obs]
                train_buffer[0].append(exp_obs)
                train_buffer[1].append(sample['actions'])
                batch_idx += 1
                if batch_idx >= len(expert_buffer):
                    batch_idx = 0
            sample_count = 0
            for i in range(self.collectors):
                if sample_count == generation_samples:
                    break
                # if rwds2[i] > 200:
                train_buffer[0].append(obs2[i])
                train_buffer[1].append(acts2[i])
                sample_count += 1
            generation_obs = []
            generation_act = []
            for i in range(self.collectors):
                generation_obs += obs2[i]
                generation_act += acts2[i]
            generation_obs = FloatTensor(generation_obs)
            generation_act = FloatTensor(generation_act)
            generation_rwd = np.concatenate(rwds2)
            expert_sample_idx = np.random.randint(0, len(train_buffer[0]), 10)
            expert_obs = []
            expert_act = []
            for i in expert_sample_idx:
                # pdb.set_trace()
                expert_obs += train_buffer[0][i]
                expert_act += list(train_buffer[1][i])
            expert_obs = FloatTensor(expert_obs)
            expert_act = FloatTensor(expert_act)
            exp_scores = self.d(expert_obs, expert_act)
            gen_scores = self.d(generation_obs, generation_act)
            loss_d = torch.nn.functional.binary_cross_entropy(
                exp_scores, torch.zeros_like(exp_scores)
            ) + torch.nn.functional.binary_cross_entropy(
                gen_scores, torch.ones_like(gen_scores)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            # for i in range(self.collectors):
            #     obs, acts, gammas = obs2[i], acts2[i], gammas2[i]
            #     obs, acts, gammas = FloatTensor(obs), FloatTensor(acts), FloatTensor(gammas)
            #     # update D net
            #     for record in batch_buffer:
            #         self.d.train()
            #         exp_obs = record['observation']
            #         exp_acts = record['actions']
            #         exp_obs = [make_obs_2(ob) for ob in exp_obs]
            #         exp_obs = FloatTensor(exp_obs)
            #         exp_acts = FloatTensor(exp_acts)
            #
            #         # score expert close to 0
            #         exp_scores = self.d(exp_obs, exp_acts)
            #         # score generator close to 1
            #         gen_scores = self.d(obs, acts)
            #
            #         loss_d = torch.nn.functional.binary_cross_entropy(
            #             exp_scores, torch.zeros_like(exp_scores)
            #         ) + torch.nn.functional.binary_cross_entropy(
            #             gen_scores, torch.ones_like(gen_scores)
            #         )
            #         opt_d.zero_grad()
            #         loss_d.backward()
            #         opt_d.step()

            # TD-update Value net
            self.d.eval()
            costs = torch.log(self.d(generation_obs, generation_act)+1e-8).squeeze().detach()
            esti_rwds = -1 * costs
            # esti_rwds = -0.8 * costs + 0.2 * generation_rwd

            self.v.train()
            esti_v = self.v(generation_obs).view(-1)
            td_v = esti_rwds[:-1] + gamma * esti_v[1:]
            loss_v = F.mse_loss(esti_v[:-1], td_v)
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()

            t2 = time.time() - t
            t = time.time()

            # PPO update Action net
            self.pi = self.ppo.update(obs2, acts2, gamma, self.v, self.d)

            t3 = time.time() - t
            t = time.time()

            print(
                f"{self.args.exp}, reward at iter {_iter}: step{_step[0]}, ",
                "score: %.2f, %.2f, m_speed: %.2f" % (np.mean(score_list[-self.collectors:]), np.max(score_list[-self.collectors:]), np.mean(speeds)),
                "time: %.2f, %.2f, %.2f" % (t1, t2, t3),
                "d_loss %.3f, v_loss %.3f" % (loss_d, loss_v)
            )

            plt.plot(np.arange(len(score_list)), score_list)
            plt.savefig('rwd' + self.args.exp + '.png')
            plt.close()
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

        return self.ppo.get_pnet(), self.v, self.d
