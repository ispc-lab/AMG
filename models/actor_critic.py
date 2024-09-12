import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from utils.gen_utils import find_reference, get_feat, SetAtomNum
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.distributions.categorical import Categorical
from models.AMG import AMG
import torch.nn.functional as F
from torch_scatter import scatter_add


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs=obs.to(self.device))
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class AMG_Actor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        self.device = device
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    def _distribution(self, obs):
        logits = self.logits_net(obs.cpu())
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    

class AMG_Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        
    def forward(self, obs):
        value = torch.squeeze(self.v_net(obs.cpu()), -1)
        return value 

    
class AMG_ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                hidden_sizes=(64,64), activation=nn.Tanh, device='cpu'):
        super().__init__()
        self.action_space = action_space
        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        self.pi = AMG_Actor(obs_dim, action_space.n, hidden_sizes, activation, device)
        # build value function
        self.v = AMG_Critic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs=obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs=obs)[0]