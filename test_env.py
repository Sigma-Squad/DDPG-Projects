import gymnasium as gym
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from DDPG.DDPGAgent import DDPGAgent, OUNoise
import numpy as np
import torch
import torch.optim as optim

#Stock Trading Environment
env = gym.make('stocks-v0', df=STOCKS_GOOGL, window_size=10, frame_bound=(10, 300))

#Parameters
observation, info = env.reset(seed=2024)
state_dim = observation.shape[0] * observation.shape[1]
action_dim = 1  
hidden_dim = 64


batch_size = 64
agent = DDPGAgent(state_dim, hidden_dim, action_dim, batch_size=batch_size)
optimizer_actor = optim.Adam(agent.actor.parameters(), lr=0.001)
optimizer_critic = optim.Adam(agent.critic.parameters(), lr=0.002)


ou_noise = OUNoise(action_dim) # AI suggested me to add it to make the training process "more stable"

# Hyperparameters
M = 100 
T = 200  
gamma = 0.99  
tau = 0.005  
noise_decay = 0.99 
max_profit = -np.inf  

# Training loop
for episode in range(1, M+1):
    observation, info = env.reset(seed=2024)
    state = np.array(observation).reshape(-1)
    ou_noise.reset()  # Reset noise at the start of each episode
    episode_reward = 0
    episode_profit = 0
    
    for t in range(1, T+1):
     
        action = agent.act(state) + ou_noise() * noise_decay
        action = np.clip(action, -1, 1)  
        
       
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state).reshape(-1)
        
       
        agent.remember(state, [action], reward, next_state, terminated or truncated)
        agent.train()
        
       
        state = next_state
        episode_reward += reward
        episode_profit += reward * (gamma ** t)  
        
        if terminated or truncated:
            break

    
    noise_decay *= noise_decay
    
    
    if episode_profit > max_profit:
        max_profit = episode_profit
        print(f"New best profit: {max_profit:.2f} at Episode {episode}")
    
    print(f"Episode {episode}/{M}, Reward: {episode_reward:.2f}, Profit: {episode_profit:.2f}")


plt.cla()
env.unwrapped.render_all()
plt.show()
