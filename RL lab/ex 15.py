import numpy as np
import random

# States: 0=unstable, 1=balanced
states = [0,1]

# Actions: 0=left step, 1=right step
actions = [0,1]

# Policy parameters
theta = np.random.rand(2,2)

gamma = 0.9
alpha = 0.1
clip_eps = 0.2   # PPO clipping

# Softmax policy
def policy(s):
    exp = np.exp(theta[s])
    return exp / np.sum(exp)

# Reward (balance)
def reward(s,a):
    return 10 if s == 1 else -5

# Next state
def next_state():
    return random.choice(states)

# Training
for ep in range(100):
    s = random.choice(states)
    
    for _ in range(5):
        old_probs = policy(s)
        a = np.random.choice(actions, p=old_probs)
        
        r = reward(s,a)
        ns = next_state()
        
        # Advantage (simple)
        advantage = r
        
        new_probs = policy(s)
        
        # ----- PPO update -----
        ratio = new_probs[a] / old_probs[a]
        clipped = np.clip(ratio, 1-clip_eps, 1+clip_eps)
        theta[s][a] += alpha * min(ratio*advantage, clipped*advantage)
        
        # ----- TRPO idea (small safe update) -----
        theta[s][a] += alpha * 0.01 * advantage  # small constrained step
        
        s = ns

# Output
print("Policy:")
for s in states:
    print(f"State {s}:", policy(s))
