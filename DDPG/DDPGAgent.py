import torch
import torch.nn as nn
import torch.optim as optim
import random


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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = (
            state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


class DDPGAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, replay_buffer_size=1e6, gamma=0.99, tau=0.001, batch_size=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma  # discount factor
        self.tau = tau  # learning rate
        self.batch_size = batch_size

        # Actor Network
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.target_actor = Actor(state_dim, hidden_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        # Critic network
        self.critic = Critic(state_dim, hidden_dim, action_dim)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        if self.replay_buffer.capacity < self.batch_size or len(self.replay_buffer.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        y_Q = torch.add(rewards, self.gamma *
                        self.target_critic(next_states, self.target_actor(next_states)))

        critic_loss = nn.MSELoss()(y_Q, self.critic(states, actions))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._update_target_networks(self.target_actor, self.actor, self.tau)
        self._update_target_networks(self.target_critic, self.critic, self.tau)

    def _update_target_networks(self, target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)
