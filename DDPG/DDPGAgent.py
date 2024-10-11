import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Ornstein-Uhlenbeck Noise
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
        
    def reset(self):
        self.state = self.mu.copy()
        
    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        z = self.relu(self.l1(state))
        # tanh activation for continuous action states
        return self.tanh(self.l2(z))

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        z = torch.cat([state, action], dim=1)
        z = self.relu(self.l1(z))
        return self.l2(z)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, replay_buffer_size=1e6, gamma=0.99, tau=0.0001, batch_size=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update factor
        self.batch_size = batch_size
        
        # Actor Network
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.target_actor = Actor(state_dim, hidden_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        # Critic Network
        self.critic = Critic(state_dim, hidden_dim, action_dim)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        # Copy the initial weights to the target networks
        self._update_target_networks(self.target_actor, self.actor, 1.0)
        self._update_target_networks(self.target_critic, self.critic, 1.0)

    def act(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()[0]  # Convert to numpy array
        return action
        
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer.memory) < self.batch_size:
            return
        
        # Sample a random mini-batch of transitions from Replay Buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Bellman update for critic
        y_Q = rewards.unsqueeze(-1) + self.gamma * self.target_critic(next_states, self.target_actor(next_states))
        
        # Update critic by minimizing loss (Mean Squared Error)
        critic_loss = nn.MSELoss()(y_Q, self.critic(states, actions))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor to maximize the value given by the critic
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update for the target networks
        self._update_target_networks(self.target_actor, self.actor, self.tau)
        self._update_target_networks(self.target_critic, self.critic, self.tau)

    def _update_target_networks(self, target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
