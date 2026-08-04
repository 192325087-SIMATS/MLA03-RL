import numpy as np
import random

# States: 0=empty, 1=object present
states = [0,1]

# Actions: 0=pick, 1=place
actions = [0,1]

# Policy parameters (weights)
theta = np.random.rand(2,2)

alpha = 0.1
gamma = 0.9

# Softmax policy
def policy(state):
    prefs = theta[state]
    exp = np.exp(prefs)
    probs = exp / np.sum(exp)
    return probs

# Reward
def reward(state, action):
    if state == 1 and action == 0:  # pick correctly
        return 10
    if state == 0 and action == 1:  # place correctly
        return 10
    return -1

# Generate episode
def generate_episode():
    episode = []
    state = random.choice(states)
    
    for _ in range(5):
        probs = policy(state)
        action = np.random.choice(actions, p=probs)
        r = reward(state, action)
        
        next_state = 0 if action == 0 else 1
        episode.append((state, action, r))
        state = next_state
    
    return episode

# Training (REINFORCE)
for _ in range(100):
    episode = generate_episode()
    G = 0
    
    for state, action, r in reversed(episode):
        G = gamma * G + r
        
        probs = policy(state)
        grad = -probs
        grad[action] += 1
        
        theta[state] += alpha * G * grad

# Output policy
print("Learned Policy (probabilities):")
for s in states:
    print(f"State {s}:", policy(s))
