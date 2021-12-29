import pdb

import torch.nn.functional as F
import torch


class BC:
    def __init__(self, PolicyNet, device=None):
        self.policy = PolicyNet
        self.w = 1
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters())

    def action(self, state):
        dist = self.policy(state)
        action = dist.sample()
        return action

    def dist(self, state):
        dist = self.policy(state)
        return dist

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def update(self, batch):
        state = batch['state'].float()
        action = batch['action'].detach().float()
        _dist = self.dist(state)
        log_probs = _dist.log_prob(action.reshape(_dist.sample().shape))
        actor_loss = torch.mean(-log_probs)
        # l2_norm_loss = 1e-2 * torch.mean(self.policy.parameters())
        actor_loss1 = actor_loss.detach()
        actor_loss2 = actor_loss1

        # _action = self.action(state)
        # actor_loss1 = F.mse_loss(_action[:, 0], action[:, 0])
        # actor_loss2 = F.mse_loss(_action[:, 1], action[:, 1])
        # actor_loss = actor_loss2 + actor_loss1

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss1.cpu().detach().numpy(), actor_loss2.cpu().detach().numpy()

    def save_model(self, path, epoch):
        state = {'policy': self.policy.state_dict(),
                 'epoch': epoch}
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))
        self.policy.load_state_dict(state['policy'])
        return state['epoch']
