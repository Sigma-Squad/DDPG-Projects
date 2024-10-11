# Test run

import gymnasium as gym
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from DDPG.DDPGAgent import DDPGAgent, OUNoise
import numpy as np
import torch

env = gym.make('stocks-v0',
        df=STOCKS_GOOGL,
        window_size=10,
        frame_bound=(10, 300)
    )

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

observation ,info= env.reset(seed=2024)
print("Observation shape:", observation.shape)

#Initialization with necessary Params
state_dim = observation.shape[0]*observation.shape[1]
action_dim = 1
hidden_dim = 64
agent = DDPGAgent(state_dim, hidden_dim, action_dim)

M = 55
T = 25

# Reset the environment
# state = np.array(observation).reshape(-1)

torch.autograd.set_detect_anomaly(True)
total_profit = 0
for episode in range(1, M+1):
    
    observation, info = env.reset(seed=2024)
    exploration_noise = OUNoise(size = action_dim)
    state = np.array(observation).reshape(-1)
    
    for t in range(1,T+1):
        
        action = sigmoid(agent.act(state) + exploration_noise())
        action = 1 if action[0] >= 0.5 else 0
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state).reshape(-1)
        agent.remember(state ,[action],reward,next_state,terminated or truncated)
        state = next_state
        total_profit += reward
        agent.train()

        done = terminated or truncated
        if done:
            print("info:", info)
            break
        print(info)

    plt.cla()
    env.unwrapped.render_all()
    plt.show()