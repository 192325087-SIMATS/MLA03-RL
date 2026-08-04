import numpy as np
import random

# States: 0=low demand, 1=high demand
states = [0,1]

# Actions: 0=stay, 1=move up, 2=move down
actions = [0,1,2]

# Actor (policy parameters)
theta = np.random.rand(2,3)

# Critic (value function)
V = np.zeros(2)

alpha_actor = 0.1
alpha_critic = 0.1
gamma = 0.9

# Softmax policy
def policy(s):
    exp = np.exp(theta[s])
    return exp / np.sum(exp)

# Reward (minimize waiting time)
def reward(s,a):
    return 5 if (s==1 and a!=0) else -2

# Next state
def next_state():
    return random.choice(states)

# Training (A2C style)
for ep in range(100):
    s = random.choice(states)
    
    for _ in range(5):
        probs = policy(s)
        a = np.random.choice(actions, p=probs)
        
        r = reward(s,a)
        ns = next_state()
        
        # TD Error (Critic)
        td_error = r + gamma * V[ns] - V[s]
        
        # Critic update
        V[s] += alpha_critic * td_error
        
        # Actor update
        grad = -probs
        grad[a] += 1
        theta[s] += alpha_actor * td_error * grad
        
        s = ns

# Output
print("State Values:", V)
print("Policy:")
for s in states:
    print(f"State {s}:", policy(s))
