import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
print("Env kwargs:", env.spec.kwargs)

Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.3
gamma = 0.99
epsilon = 1.0

successes = 0

print(env.unwrapped.P[0][2])

for ep in range(2000):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        # epsilon greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        old = Q[state, action]
        target = reward + gamma * np.max(Q[next_state])
        Q[state, action] = old + alpha * (target - old)

        ep_reward += reward
        state = next_state

    # epsilon decay
    epsilon = max(0.05, epsilon * 0.999)

    # track successes
    if ep_reward > 0:
        successes += 1

    if ep % 100 == 0 and ep > 0:
        print(f"Episode {ep} | Success: {successes}/100 | Max Q: {np.max(Q):.4f}")
        successes = 0
np.save("Q_table.npy", Q)
print("Training done")