import pdb
from net import *
from test_utils import *
from ppo import PPO
import gym

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def trans2tensor(batch):
    for k in batch:
        # print(batch[k])
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device)
        elif isinstance(batch[k][0], torch.Tensor):
            batch[k] = torch.cat(batch[k]).to(device=device)
        else:
            batch[k] = torch.tensor(batch[k], device=device, dtype=torch.float32)

    return batch


class PNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, net_dims):
        super(PNet, self).__init__()
        self.net_dims = net_dims
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i]))
            layers.append(torch.nn.ELU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], action_dim))
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, states):
        mean, var = torch.chunk(self.layers(states), 2, dim=-1)
        var = torch.tanh(var) * 10
        var = F.softplus(var)
        cov_mat = torch.diag_embed(var)
        dist = torch.distributions.MultivariateNormal(mean, cov_mat)

        return dist


class VNet(torch.nn.Module):
    def __init__(self, state_dim, net_dims):
        super(VNet, self).__init__()
        self.net_dims = net_dims
        layers = []
        last_dim = state_dim
        for i in range(len(self.net_dims)):
            layers.append(torch.nn.Linear(last_dim, self.net_dims[i]))
            layers.append(torch.nn.ReLU())
            last_dim = self.net_dims[i]
        layers.append(torch.nn.Linear(self.net_dims[-1], 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, states):
        val = self.layers(states)
        return val


# env_name = "MountainCarContinuous-v0"
env_name = 'Pendulum-v1'
env = gym.make(env_name)

max_timesteps = 1000
gamma = 0.99
train_param = {
    'policy_lr': 1e-3,
    'value_lr': 1e-3,
    'max_kl': 1e-1,
    'beta': 1,
    'penalty': True,
    'line_search': False
}
policy_net = [32]
value_net = [32]
target_value = VNet(3, value_net)
vnet = VNet(3, value_net)
pnet = PNet(3, 2, policy_net)
ppo = PPO(train_param,
          PolicyNet=pnet, ValueNet=vnet, targetValueNet=target_value,
          state_dim=3, action_dim=1, n_step=1)


def train(num_episode=1000, mini_epoch=5, print_every=20):
    rewards_log = []
    episodes_log = []
    batch_size = 1024
    for i_episode in range(num_episode):
        states = []
        acts = []
        rewards = []
        next_states = []
        log_probs = []
        dones = []

        while len(states) < batch_size:
            _rewards = []
            state = env.reset()
            env.seed(1)
            random.seed(1)

            state = torch.tensor([state], device=device, dtype=torch.float32)

            for timestep in range(max_timesteps):
                action = ppo.action(state)
                _, log_prob = ppo.dist(state, action)
                next_state, reward, done, _ = env.step(action.numpy().reshape(-1))

                # collect samples
                states.append(state)
                acts.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                _rewards.append(reward)
                next_states.append(torch.tensor([next_state], dtype=torch.float32))
                dones.append(done)

                state = torch.tensor([next_state], device=device, dtype=torch.float32)
                if done:
                    rewards_log.append(np.sum(_rewards))
                if done or len(states) == batch_size:
                    break
        episodes_log.append(i_episode)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        acts = torch.tensor(acts).reshape(-1, 1)
        log_probs = torch.stack(log_probs).reshape(-1, 1)
        rewards = torch.tensor(rewards).reshape(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = torch.tensor(dones).int().reshape(-1, 1)
        batch = {"state": states, "action": acts,
                 "log_prob": log_probs,
                 "next_state": next_states, "done": dones,
                 "reward": rewards, }
        adv = ppo.compute_adv(batch, gamma)
        batch["adv"] = adv
        for i in range(mini_epoch):
            ppo.update(batch, gamma, mini_batch=128)
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            print("Episode: {}, Reward: {}".format(i_episode + 1, np.mean(rewards_log[-10:])))
    infos = {
        "rewards": rewards_log,
        "episodes": episodes_log
    }
    return infos

def train2(num_episode=1000, mini_epoch=5, print_every=20):
    rewards_log = []
    episodes_log = []
    for i_episode in range(num_episode):
        states = []
        acts = []
        rewards = []
        next_states = []
        log_probs = []
        dones = []

        _rewards = 0
        state = env.reset()
        env.seed(1)
        random.seed(1)

        state = torch.tensor([state], device=device, dtype=torch.float32)

        for timestep in range(max_timesteps):
            action = ppo.action(state)
            _, log_prob = ppo.dist(state, action)
            next_state, reward, done, _ = env.step(action.numpy().reshape(-1))

            # collect samples
            states.append(state)
            acts.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            _rewards += reward
            next_states.append(torch.tensor([next_state], dtype=torch.float32))
            dones.append(done)

            state = torch.tensor([next_state], device=device, dtype=torch.float32)
            if done:
                rewards_log.append(np.sum(_rewards))
                break
        mini_batch = len(states)
        episodes_log.append(i_episode)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        acts = torch.tensor(acts).reshape(-1, 1)
        log_probs = torch.stack(log_probs).reshape(-1, 1)
        rewards = torch.tensor(rewards).reshape(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = torch.tensor(dones).int().reshape(-1, 1)
        batch = {"state": states, "action": acts,
                 "log_prob": log_probs,
                 "next_state": next_states, "done": dones,
                 "reward": rewards, }
        adv = ppo.compute_adv(batch, gamma)
        batch["adv"] = adv
        for i in range(mini_epoch):
            ppo.update(batch, gamma, mini_batch)
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            print("Episode: {}, Reward: {}".format(i_episode + 1, np.mean(rewards_log[-10:])))
    infos = {
        "rewards": rewards_log,
        "episodes": episodes_log
    }
    return infos


ppo_cartpole_infos = train2(2000, 10, print_every=10)


# state = env.reset()
# state = torch.tensor([state], device=device, dtype=torch.float32)
# for time_step in range(max_timesteps):
#     action = ppo.action(state)
#     _, log_prob = ppo.dist(state, action)
#     next_state, reward, done, _ = env.step(action.numpy().reshape(-1))
#     env.render()
#
#     state = torch.tensor([next_state], device=device, dtype=torch.float32)
#     if done:
#         break