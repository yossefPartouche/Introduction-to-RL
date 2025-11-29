from prisoner_dilemma_env import PrisonerDillema
import matplotlib.pyplot as plt 
import numpy as np

payoff = {(0, 0): 3, (0, 1): 0, (1, 0):5, (1, 1):1}


def m1_alwaysC():
    actions = [0, 1]
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: 0 for a in actions} for s in states}

    for s in states:
        my_prev, opp_prev = s
        for a in actions:
            opp_action = 0
            new_state = (a, opp_action)
            reward = payoff[(a, opp_action)]

            P[s][a] = [(1.0, new_state, reward)]
            R[s][a] = reward
    return states, actions, P, R
    
def m1_alwaysD():
    states = [(0, 0),(0, 1), (1, 0), (1, 1) ]
    actions = [0, 1]
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: 0 for a in actions} for s in states}

    for s in states:
        my_prev, opp_prev = s
        for a in actions:
            opp_action = 1
            new_state = (a, opp_action)
            reward = payoff[(a, opp_action)]

            P[s][a] = [(1.0, new_state, reward)]
            R[s][a] = reward
    return states, actions, P, R
    
def m1_TfT():
    states = [(0, 0),(0, 1), (1, 0), (1, 1) ]
    actions = [0, 1]
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: 0 for a in actions} for s in states}
    for s in states:
        my_prev, opp_prev = s
        for a in actions:
            opp_action = my_prev
            new_state = (a, opp_prev)
            reward = payoff[(a, opp_action)]

            P[s][a] = [(1.0, new_state, reward)]
            R[s][a] = reward
    return states, actions, P, R
    
def m1_TfT_stochastic():
    states = [(0, 0),(0, 1), (1, 0), (1, 1) ]
    actions = [0, 1]
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: 0 for a in actions} for s in states}
    p = 0.9
    for s in states:   
        my_prev, _ = s
        for a in actions:
            P[s][a] = [
                (p, (a, my_prev), payoff[(a, my_prev)]), 
                (1-p, (a, 1 - my_prev), payoff[(a, 1-my_prev)])]
            R[s][a] = p*payoff.get((a, my_prev)) + (1-p)*payoff[(a, 1-my_prev)]
    return states, actions, P, R

states, actions, P, R = m1_TfT_stochastic()
gammas = [0.1, 0.5, 0.9, 0.99]
color = plt.cm.viridis(np.linspace(0,1, len(gammas)))
for gamma, c in zip(gammas, color):
    env = PrisonerDillema(states, actions, P, R, gamma = gamma)
    pi, V = env.policy_iteration()

    states_list = list(env.states)
    avg_values = [
        np.mean(list(V_t.values())) for V_t in env.V_history
    ]
    plt.plot(avg_values, label=f"γ={gamma}", color=c)

plt.xlabel("Policy Evaluation Iteration")
plt.ylabel("Value V(s)")
plt.title("Convergence of Value Function")
plt.grid()
plt.legend()
plt.show()