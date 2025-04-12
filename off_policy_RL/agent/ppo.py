import os
import torch
import numpy as np
from torch import Tensor
from copy import deepcopy
from models import Actor, Critic, SoftActor
from utils import get_schedule

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class PPOAgent:
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, clip_param, ppo_epoch, mini_batch_size, eps_schedule, log_std_min, log_std_max, device, target_kl, entropy_coef, value_loss_coef, initial_penalty):
        
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.actor_net = SoftActor(state_size, action_size, hidden_dim, deepcopy(action_space), log_std_min, log_std_max).to(device)
        self.actor_net_old = deepcopy(self.actor_net).to(device)
        
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)
        
        # 超参数设置
        self.gamma = gamma ** nstep   # 用 n 步折扣后的值
        self.tau = tau
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.target_kl = target_kl
        self.penalty_coeff = initial_penalty  # KL 惩罚的初始系数
        
        self.train_step = 0
        self.device = device
        self.epsilon_schedule = get_schedule(eps_schedule)

    def __repr__(self):
        return "PPOAgent"

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def get_action(self, 
            state: Float[np.ndarray, "state_dim"] | Float[np.ndarray, "batch_size state_dim"], 
            sample: bool = False
        ) -> Float[np.ndarray, "action_dim"] | Float[np.ndarray, "batch_size action_dim"]:
        
        state = torch.as_tensor(state).to(self.device)
        action, _ = self.actor_net.evaluate(state, sample)
        return action.cpu().numpy()
    
    def eval(self):
        self.actor_net.eval()

    def train(self):
        self.actor_net.train()

    def update(self, batch, weights=None):
        """
        完成一轮 PPO 更新：
        1. 利用 critic_net(state, action) 计算当前动作 Q 值
        2. 对于每个 (state, action, reward, next_state, done)，在 next_state 下采样动作，
            用 critic_net(next_state, next_action) 得到下状态 Q 值，计算 target = reward + gamma * Q_next * (1 - done)
        3. 优势 advantage = target - Q
        4. 多轮 PPO update，每轮随机打乱 batch（mini_batch_size）
        5. Actor 的 PPO 裁剪目标损失、KL 惩罚以及 Critic 均方误差损失
        6. 更新 Actor 和 Critic 网络，最后更新 actor_net_old 权重
        """
        state, action, reward, next_state, done = batch
        
        # 转换到 device
        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device).float().squeeze(-1)
        next_state = next_state.to(self.device)
        done       = done.to(self.device).float().squeeze(-1)
        
        # 使用actor生成确定性动作来估计状态价值 V(s)
        with torch.no_grad():
            det_action, _ = self.actor_net.evaluate(state, sample=False)
            V_value = self.critic_net(state, det_action).squeeze(-1) # TODO: just approximation
            det_next_action, _ = self.actor_net.evaluate(next_state, sample=False)
            V_next = self.critic_net(next_state, det_next_action).squeeze(-1)
            # 计算状态价值的TD目标
            target = reward + self.gamma * V_next * (1.0 - done)
        
        # 计算当前状态动作对的 Q 值（仍然使用经验中的动作）
        Q_value = self.critic_net(state, action).squeeze(-1)
        # 优势函数定义为 Q(s,a) - V(s)
        advantage = Q_value - V_value
        
        # 分离目标和优势函数，避免梯度传播
        target = target.detach()
        advantage = advantage.detach()
        
        batch_size = state.size(0)
        
        # 多轮 PPO update
        for epoch in range(self.ppo_epoch):
            # 打乱 indices，并分成 mini-batch
            permuted_indices = torch.randperm(batch_size)
            for i in range(0, batch_size, self.mini_batch_size):
                mb_inds = permuted_indices[i: i + self.mini_batch_size]
                mb_state = state[mb_inds]
                mb_action = action[mb_inds]
                mb_advantage = advantage[mb_inds]
                mb_target = target[mb_inds]
                
                # 当前策略评估：计算当前策略下动作对的 log 概率
                _, mb_log_prob = self.actor_net.evaluate(mb_state, sample=True)
                mb_log_prob = mb_log_prob.squeeze(-1)
                # 使用旧策略计算对应的 log 概率
                with torch.no_grad():
                    _, mb_log_prob_old = self.actor_net_old.evaluate(mb_state, sample=True)
                    mb_log_prob_old = mb_log_prob_old.squeeze(-1)
                
                # 计算概率比率
                ratio = torch.exp(mb_log_prob - mb_log_prob_old)
                # PPO clip目标
                surr1 = ratio * mb_advantage
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surr2 = clipped_ratio * mb_advantage
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                
                # 计算 KL 散度（当前策略与旧策略之间的差异）
                kl_div = self.compute_kl(mb_state)
                
                # 计算 critic 均方误差损失：当前 critic 对 (state, action) 的预测与 target 的差异
                mb_Q = self.critic_net(mb_state, mb_action).squeeze(-1)
                critic_loss = torch.mean((mb_target - mb_Q) ** 2)
                
                # 熵奖励（鼓励探索），假设 actor_net 实现 entropy 方法，传入状态信息
                entropy_bonus = torch.mean(self.actor_net.entropy(mb_state))
                
                # 总损失：actor 的 PPO 损失 + KL 惩罚 + critic 均方误差 - 熵奖励
                total_actor_loss = actor_loss + self.penalty_coeff * kl_div - self.entropy_coef * self.epsilon_schedule(self.train_step) * entropy_bonus
                total_critic_loss = self.value_loss_coef * critic_loss
                
                # 优化更新
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                self.critic_optimizer.step()
                
                # 自适应调整惩罚系数，使 KL 散度保持在目标附近
                if kl_div > self.target_kl * 1.5:
                    self.penalty_coeff *= 2.0
                elif kl_div < self.target_kl / 1.5:
                    self.penalty_coeff *= 0.5
        
        # 更新完成后，将当前策略复制到旧策略网络
        self.actor_net_old.load_state_dict(self.actor_net.state_dict())
        self.train_step += 1
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'td_error': advantage.mean().item()}
    
    def compute_kl(self, states):
        """
        计算当前策略与旧策略之间的 KL 散度（假设均为高斯分布
        返回均值(mu)和log_std
        """
        mu, log_std = self.actor_net(states)
        old_mu, old_log_std = self.actor_net_old(states)
        old_mu, old_log_std = old_mu.detach(), old_log_std.detach()
        # 利用高斯分布的KL公式，注意需要保证各维度对应相加后求均值
        # KL(old || new) = log_std_new - log_std_old + ( exp(2*log_std_old) + (old_mu - mu)^2 ) / (2*exp(2*log_std)) - 0.5
        kl = log_std - old_log_std + (torch.exp(2 * old_log_std) + (old_mu - mu) ** 2) / (2.0 * torch.exp(2 * log_std)) - 0.5
        # 对各样本各维度计算均值
        return torch.mean(kl)

    def save(self, name_prefix='best_'):
        os.makedirs('models', exist_ok=True)
        torch.save(self.critic_net.state_dict(), os.path.join('models', name_prefix + '_critic.pt'))
        torch.save(self.actor_net.state_dict(), os.path.join('models', name_prefix + '_actor.pt'))

    def load(self, name_prefix='best_'):
        self.critic_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_critic.pt')))
        self.actor_net.load_state_dict(torch.load(os.path.join('models', name_prefix + '_actor.pt')))
        self.actor_net_old = deepcopy(self.actor_net)