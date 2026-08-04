import numpy as np
import random

# Grid world (3x3 warehouse)
size = 3
goal = (2,2)

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

# Initialize
V = np.zeros((size,size))              # TD(0)
Q = np.zeros((size,size,4))            # SARSA & Q-Learning

alpha, gamma, epsilon = 0.1, 0.9, 0.2

def move(state, a):
    x,y = state
    dx,dy = actions[a]
    nx,ny = x+dx, y+dy
    if 0<=nx<size and 0<=ny<size:
        return (nx,ny)
    return state

def reward(s):
    return 10 if s==goal else -1

# Training
for ep in range(100):
    s = (0,0)
    
    # choose action (ε-greedy)
    a = random.randint(0,3)
    
    while s != goal:
        ns = move(s,a)
        r = reward(ns)
        
        # TD(0) update
        V[s] += alpha * (r + gamma * V[ns] - V[s])
        
        # SARSA (on-policy)
        if random.random() < epsilon:
            na = random.randint(0,3)
        else:
            na = np.argmax(Q[ns])
        
        Q[s][a] += alpha * (r + gamma * Q[ns][na] - Q[s][a])
        
        # Q-Learning (off-policy)
        Q[s][a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s][a])
        
        s, a = ns, na

# Test path
s = (0,0)
print("Path:")
while s != goal:
    print(s, end=" -> ")
    a = np.argmax(Q[s])
    s = move(s,a)
print("Goal!")
