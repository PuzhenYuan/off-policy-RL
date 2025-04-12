import torch
from torch import Tensor
from torch.nn import functional as F
from models import Critic, Actor, SoftActor
from copy import deepcopy
import numpy as np
import os
from utils import get_schedule

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class BACAgent:
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, target_update_interval, log_std_min, log_std_max, device):
        
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
        self.actor_net = SoftActor(state_size, action_size, hidden_dim, deepcopy(action_space), log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)
        
        self.tau = tau
        self.gamma = gamma ** nstep
        self.device = device
        self.target_update_interval = target_update_interval

        self.train_step = 0
    
    def __repr__(self):
        return "BACAgent"
    
    def eval(self):
        self.actor_net.eval()
    
    def train(self):
        self.actor_net.train()
    
    @torch.no_grad()
    def get_action(self, state, sample=False):
        action, _ = self.actor_net.evaluate(torch.as_tensor(state).to(self.device), sample)
        return action.cpu().numpy()
    
    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        critic_loss, td_error = self.update_critic(state, action, reward, next_state, done, weights)
        actor_loss = self.update_actor(state)
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target, self.critic_net)
        self.train_step += 1
        return {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'td_error': td_error}

    def update_critic(self, state, action, reward, next_state, done, weights=None):
        next_action, next_log_prob = self.actor_net.evaluate(next_state, sample=True)
        next_Q = self.critic_target(next_state, next_action)
        target_Q = reward + self.gamma * (1 - done) * next_Q
        Q = self.critic_net(state, action)
        critic_loss = torch.mean((Q - target_Q)**2 * (weights if weights is not None else 1))
        td_error = torch.abs(Q - target_Q).detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), td_error.mean().item()

    def update_actor(self, state):
        action, log_prob = self.actor_net.evaluate(torch.as_tensor(state).to(self.device), sample=True)
        Q = self.critic_net(state, action)
        actor_loss = -Q.mean()
        # actor_loss = torch.mean(-Q.detach() * log_prob)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def save(self, name_prefix='best_'):
        os.makedirs('models', exist_ok=True)
        torch.save(self.critic_net.state_dict(), os.path.join('models', name_prefix + '_critic.pt'))
        torch.save(self.actor_net.state_dict(), os.path.join('models', name_prefix + '_actor.pt'))

    def load(self, name_prefix='best_'):
        self.critic_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_critic.pt')))
        self.actor_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_actor.pt')))
        self.critic_target.load_state_dict(self.critic_net.state_dict())