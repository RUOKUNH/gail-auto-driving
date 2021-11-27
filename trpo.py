import pdb
import numpy as np
import torch
from functools import partial
from utils import *


def L(p_net, obs, acts, advs, old_dist):
    dist = p_net(obs)
    # pdb.set_trace()
    adl = (advs * torch.exp(
        dist.log_prob(acts)
        - old_dist.log_prob(acts).detach()
    )).mean()
    # print(adl)
    # pdb.set_trace()
    return adl


def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


# only available for gaussian distributions
# refer https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#
# As the cov mat is diag, so some operations like trace and inverse is achieved by sumation
def kl_divergence(dist, old_dist, action_dim):
    old_mean = old_dist.mean.detach()
    old_cov = old_dist.covariance_matrix.sum(-1).detach()
    mean = dist.mean
    cov = dist.covariance_matrix.sum(-1)
    return 0.5 * ((old_cov / cov).sum(-1)
                  + (((old_mean - mean) ** 2) / cov).sum(-1)
                  - action_dim
                  + torch.log(cov).sum(-1)
                  - torch.log(old_cov).sum(-1)).mean()


def conjugate_gradient(Av_func, b, max_iter=50, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2  # 2-norm

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


class TRPO:
    # kl_thre is the upper threshold of kl_divergence between new and old policy
    # L_func is just the added part beside the original reward value
    def __init__(self, kl_thre, action_dim, state_dim):
        self.kl_thre = kl_thre
        self.cg_damping = 1e-2
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.step_total = 0
        self.update_step = 1

    # compute Hessian matrix
    def Hv(self, p_net, grad_kld, v):
        hessian = get_flat_grads(torch.dot(grad_kld, v), p_net).detach()

        return hessian + self.cg_damping * v

    def rescale_and_linesearch(
            self, g, x, Hx, kld, p_net, obs, acts, advs, grad_kld, max_iter=10, success_ratio=0.1):
        old_params = get_flat_params(p_net).detach()
        # set_params(p_net, old_params)
        old_dist = p_net(obs)
        L_old = L(p_net, obs, acts, advs, old_dist).detach()
        # if self.step_total >= self.update_step:
        #     hessian = torch.cat([get_flat_grads(grad, p_net).view(-1, 1) for grad in grad_kld]
        #                         , axis=1)
        #     try:
        #         _x = torch.linalg.lstsq(hessian, g).solution
        #         beta = torch.sqrt((2 * self.kl_thre) / torch.dot(g, _x))
        #         self.step_total = 0
        #     except:
        #         _x = x
        #         beta = torch.sqrt((2 * self.kl_thre) / torch.dot(x, Hx))
        #         print('pinv failed')
        # else:
        _x = x
        beta = torch.sqrt((2 * self.kl_thre) / torch.dot(x, Hx))

        # pdb.set_trace()

        for _ in range(max_iter):
            new_params = old_params + beta * _x

            set_params(p_net, new_params)
            dist = p_net(obs)
            kld_new = kl_divergence(dist, old_dist, self.action_dim)

            L_new = L(p_net, obs, acts, advs, old_dist).detach()

            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, beta * _x)  # first-order approximation
            ratio = actual_improv / approx_improv

            if ratio > success_ratio \
                    and actual_improv > 0 \
                    and kld_new < self.kl_thre:
                return new_params, 1

            beta *= 0.5

        # print("The line search was failed!")
        return old_params, -1

    # pack params of L_func as a list
    def step(self, p_net, obs, acts, advs, extra_gradient=None):
        self.step_total += 1
        old_dist = p_net(obs)
        dist = p_net(obs)
        # policy gradient
        g = get_flat_grads(L(p_net, obs, acts, advs, old_dist), p_net).detach()
        kld = kl_divergence(dist, old_dist, self.action_dim)
        grad_kld = get_flat_grads(kld, p_net)
        # x = inv(H)*g, by solving from H*s=g
        x = conjugate_gradient(partial(self.Hv, p_net, grad_kld), g).detach()
        # Hx=H*x, used in finding new_params
        Hx = self.Hv(p_net, grad_kld, x)
        # if self.step_total >= self.update_step:
        #     hessian = torch.cat([get_flat_grads(grad, p_net).view(-1, 1) for grad in grad_kld]
        #                         , axis=1)
        #     try:
        #         hessian_inv = torch.linalg.pinv(hessian)
        #         self.step_total = 0
        #     except:
        #         print('failed getting pinv')
        # line search on param ahead of DeltaTheta to prevent step too big
        # if self.step_total > 0:
        new_param, flag = self.rescale_and_linesearch(g, x, Hx, kld, p_net, obs, acts, advs, grad_kld)
        # else:
        #     new_param, flag = self.rescale_and_linesearch(g, x, Hx, kld, p_net, obs, acts, advs,
        #                                                   hessian_inv=hessian_inv)
        if extra_gradient is not None:
            set_params(p_net, new_param + extra_gradient)
        else:
            set_params(p_net, new_param)

        return flag
