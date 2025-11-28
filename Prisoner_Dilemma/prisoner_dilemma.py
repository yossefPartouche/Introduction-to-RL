import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random
import time 
import pygame

class PrisonerDillema(gym.Env):
    def __init__(self):

        self.states = [('C', 'C'), ('D','C')]
        self.actions = ['D', 'C']
        self.P = {s : {a: [] for a in self.actions} for s in self.states}
        self.P[('C', 'C')]['D'] = (1.0, self.states[1], 5)
        self.P[('C', 'C')]['C'] = (1.0, self.states[0], 3)
        self.P[('D', 'C')]['D'] = (1.0, self.states[1], 5)
        self.P[('D', 'C')]['C'] = (1.0, self.states[0], 3)
        self.gamma = 0.9
        #self.gamma = {0.1, 0.5, 0.9, 0.99}
        self.theta = 1e-6
        pi = self.pi = {
            ('C', 'C') : {'C' : 0.5, 'D' : 0.5},
            ('D', 'C') : {'C' : 0.5, 'D' : 0.5},
        }
        V = self.V = {}
    
    def reset(self):
        pi = {s : random.choice(self.actions) for s in self.states}
        V = {s : 0.0 for s in self.states}

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in self.states:
                v_old = self.V[s]
                a = self.pi[s]
                self.V[s] = np.sum(prob * (reward + self.gamma * self.V[s_next]) for (prob, s_next, reward) in self.P[s][a])
                delta = max(delta, np.abs(v_old- self.V[s]))
            if delta < self.theta:
                break
        return self.V