# Test run

import gymnasium as gym
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt

env = gym.make('stocks-v0',
        df=STOCKS_GOOGL,
        window_size=10,
        frame_bound=(10, 300)
    )

observation = env.reset(seed=2024)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        print("info:", info)
        break

plt.cla()
env.unwrapped.render_all()
plt.show()