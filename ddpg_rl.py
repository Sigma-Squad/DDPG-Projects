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
agent = DDPGAgent(20, 4, 1)
state, reward = env.reset()
state = np.array(state).reshape(20)
epochs = 1


def train_test(agent, state, reward, epochs, istrain):

    while True:

        action = 0 if agent.act(state) < 0.5 else 1

        nstate, reward, terminated, truncated, info = env.step(action)
        nstate = np.array(nstate).reshape(20)

        agent.remember(state, [action], reward,
                       nstate, terminated or truncated)
        if istrain:
            agent.train()

        state = nstate
        done = terminated or truncated

        if done and (not epochs or not istrain):
            print("info:", info)
            break
        elif done:
            print(epochs, "Remaining")
            epochs -= 1
            state, reward = env.reset()
            state = np.array(state).reshape(20)


train_test(agent, state, reward, epochs, True)

state, reward = env.reset()
state = np.array(state).reshape(20)

train_test(agent, state, reward, epochs, False)

plt.cla()
env.unwrapped.render_all()
plt.show()
