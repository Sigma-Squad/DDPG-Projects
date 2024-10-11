import gymnasium as gym
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from DDPG.DDPGAgent import DDPGAgent
import numpy as np
import torch

env = gym.make('stocks-v0',
               df=STOCKS_GOOGL,
               window_size=10,
               frame_bound=(10, 300)
               )

observation, info = env.reset(seed=2024)
state_dim = observation.shape[0] * observation.shape[1]  
action_dim = 1  
hidden_dim = 4
agent = DDPGAgent(state_dim, hidden_dim, action_dim)

M = 45  # episodes
exploration_noise = np.random.normal  # random process for action exploration
noise_scale = 0.1 

for ep in range(M):
    observation, info = env.reset(seed=2024)
    state = observation.flatten()  # Flatten the observation to match agent input
    exploration_noise=np.random.normal
    episode_profit = 0 
    
    while True:
        action = agent.act(state) + exploration_noise(0, noise_scale)  # Adding noise for exploration
        action_tensor = torch.tensor(action, dtype=torch.float32)  
        action = 1 if torch.sigmoid(action_tensor) >= 0.5 else 0  
    
        n_state, reward, terminated, truncated, info = env.step(action)
        n_state = n_state.flatten()  
        agent.remember(state, [action], reward, n_state, terminated or truncated)
        agent.train()    
        state = n_state
        episode_profit += reward 
        done = terminated or truncated
        if done:
            print(f"Episode: {ep + 1}, Total Profit: {episode_profit:.2f}, Info: {info}")
            break  
plt.cla()
env.unwrapped.render_all()
plt.show()
