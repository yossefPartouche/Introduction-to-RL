import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
# C = 0 and D = 1

class PrisonerDillema(gym.Env):

    def __init__(self, states, actions, transitions, rewards, gamma=0.9, theta=1e-6):
        self.states = states
        self.actions = actions
        self.P = transitions
        self.R = rewards
        self.theta = theta
        self.gamma = gamma
        
        self.V = {s : 0.0 for s in states}
        self.V_history = []
        self.pi = {s: np.random.choice(actions) for s in states}
    
    def policy_iteration(self):
        iteration = 0
        while True:
            print(f"\n=== Policy Iteration Step {iteration} ===")
            print(self.gamma)
            print("Current policy:", self.pi)
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print("\nPolicy stabilized! Final policy:", self.pi)
                break
            iteration +=1
        return self.pi, self.V
    
    def policy_evaluation(self):
        while True:
            delta = 0
            V_snapshot = {}
            for s in self.states:
                v_old = self.V[s]
                a = self.pi[s]
                self.V[s] = sum(prob * (reward + self.gamma * self.V[s_next]) for (prob, s_next, reward) in self.P[s][a])
                delta = max(delta, np.abs(v_old- self.V[s]))
            self.V_history.append(self.V.copy())
            if delta < self.theta:
                break
        return self.V
    
    def policy_improvement(self):
        is_policy_stable = True
        for s in self.states:
            old_action = self.pi[s]
            action_vals = { a: sum(prob * (reward + self.gamma * self.V[s_next]) for prob, s_next, reward in self.P[s][a]) for a in self.actions}
            best_action = max(action_vals, key=action_vals.get)
            self.pi[s] = best_action
 
            if best_action != old_action:
                is_policy_stable = False
        return is_policy_stable
    

