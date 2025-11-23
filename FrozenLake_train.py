import gymnasium as gym 
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

state, _ = env.reset()

Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.3
gamma = 0.99
epsilon = 0.99

for ep in range(2000):
    state, _ = env.reset()
    done = False

    while not done: 
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * ( reward + gamma*np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
np.save("Q_table.npy", Q)
print(Q)
print("Training finished")

    
