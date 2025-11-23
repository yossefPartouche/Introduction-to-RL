import gymnasium as gym
import numpy as np
import time

Q = np.load("Q_table.npy")
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = "human")
num_visual_episodes = 5

for ep in range(num_visual_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        print(f"State {state}, Action {action}")
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.2)
    print(f"Episode {ep+1}: reward = {total_reward}")
env.close()
