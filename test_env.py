import gymnasium as gym
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from DDPG.DDPGAgent import DDPGAgent
import numpy as np

env = gym.make('stocks-v0',
        df=STOCKS_GOOGL,
        window_size=10,
        frame_bound=(10, 300)
    )

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
# print("Action Dim:", action_dim)
action_dim = 1

hidden_dim = 4
ddpg_agent = DDPGAgent(state_dim, hidden_dim, action_dim)

episodes = 2
total_rewards = []

for episode in range(episodes):
    state = env.reset(seed=2024)[0] 
    state = np.array(state).reshape(-1)  
    current_episode_reward = 0
    done = False
    while not done:
        action = ddpg_agent.act(state)
        # print(action[0])
        # if ddpg_agent.act(state)<0.5 :
        #     action1 = 0
        #     action = [0]
        # else:
        #     action1 = 1
        #     action = [1]
        next_state, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        next_state = np.array(next_state).reshape(-1)
        ddpg_agent.remember(state, action, reward, next_state, done)
        ddpg_agent.train()
        state = next_state
        current_episode_reward += reward
        if done:
            print(f"Episode {episode + 1}")
            print("info:", info)
            total_rewards.append(current_episode_reward)
            break

# Plot the results
# plt.plot(total_rewards)
# plt.title('Rewards per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.show()

plt.cla()
env.unwrapped.render_all()
plt.show()