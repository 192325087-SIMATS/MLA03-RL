import numpy as np
import random

# States: traffic levels (0=low, 1=high)
states = [0,1]

# Actions: 0=NS green, 1=EW green
actions = [0,1]

Q = np.zeros((2,2))        # Q-table
target_Q = np.zeros((2,2)) # for DDQN

alpha, gamma, epsilon = 0.1, 0.9, 0.2

# Reward (minimize waiting time)
def reward(state, action):
    return 5 if state == action else -5

# Simulate environment
def next_state():
    return random.choice(states)

# Experience memory (PER idea)
memory = []

# Training
for ep in range(100):
    s = random.choice(states)
    
    for _ in range(10):
        # ε-greedy
        if random.random() < epsilon:
            a = random.choice(actions)
        else:
            a = np.argmax(Q[s])
        
        ns = next_state()
        r = reward(s,a)
        
        # ----- DQN -----
        Q[s][a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s][a])
        
        # ----- DDQN -----
        best_next = np.argmax(Q[ns])
        target = r + gamma * target_Q[ns][best_next]
        Q[s][a] += alpha * (target - Q[s][a])
        
        # ----- Dueling (value + advantage idea) -----
        V = np.mean(Q[s])               # state value
        A = Q[s][a] - V                # advantage
        Q[s][a] = V + A               # combine
        
        # ----- PER (store important experiences) -----
        priority = abs(r)
        memory.append((priority, s, a))
        memory = sorted(memory, reverse=True)[:10]  # keep top
        
        s = ns
    
    target_Q = Q.copy()

# Output
print("Q-table:\n", Q)
print("Best Policy:", [np.argmax(Q[s]) for s in states])
