import numpy as np
import random

# States: 0=far, 1=near, 2=aligned
states = [0,1,2]

# Actions: 0=forward, 1=left, 2=right, 3=park
actions = [0,1,2,3]

# Policy parameters
theta = np.random.rand(3,4)

alpha = 0.1
gamma = 0.9

# Softmax policy
def policy(s):
    exp = np.exp(theta[s])
    return exp / np.sum(exp)

# Reward function
def reward(s,a):
    if s == 2 and a == 3:  # correct parking
        return 20
    elif a == 3:
        return -10          # wrong parking
    else:
        return -1           # movement cost

# Next state (simple simulation)
def next_state(s,a):
    if a == 0: return min(s+1,2)
    if a in [1,2]: return s
    return s

# Generate episode
def generate_episode():
    s = 0
    episode = []
    
    for _ in range(6):
        probs = policy(s)
        a = np.random.choice(actions, p=probs)
        r = reward(s,a)
        
        episode.append((s,a,r))
        s = next_state(s,a)
    
    return episode

# Training
for _ in range(100):
    ep = generate_episode()
    G = 0
    
    for s,a,r in reversed(ep):
        G = gamma * G + r
        
        probs = policy(s)
        grad = -probs
        grad[a] += 1
        
        theta[s] += alpha * G * grad

# Output learned policy
print("Policy (action probabilities):")
for s in states:
    print(f"State {s}:", policy(s))
