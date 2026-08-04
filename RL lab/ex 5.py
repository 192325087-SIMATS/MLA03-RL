import numpy as np
import random

# Number of ads (arms)
n_ads = 3

# True click probabilities (unknown to agent)
true_probs = [0.2, 0.5, 0.8]

# Initialize estimates
Q = [0] * n_ads   # estimated rewards
N = [0] * n_ads   # count of selections

epsilon = 0.2     # exploration rate

# Simulate 1000 users
for t in range(1000):
    
    # ε-greedy choice
    if random.random() < epsilon:
        ad = random.randint(0, n_ads-1)   # explore
    else:
        ad = np.argmax(Q)                 # exploit
    
    # Simulate user click (1 or 0)
    reward = 1 if random.random() < true_probs[ad] else 0
    
    # Update values
    N[ad] += 1
    Q[ad] += (reward - Q[ad]) / N[ad]

# Result
print("Estimated CTR:", Q)
print("Best Ad:", np.argmax(Q))
