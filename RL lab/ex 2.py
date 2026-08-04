import numpy as np
import random

# Grid (3x3 home)
grid_size = 3
goal = (2, 2)

# Actions: up, down, left, right
actions = [(-1,0),(1,0),(0,-1),(0,1)]

# Q-table
Q = {}

# Initialize Q-values
for i in range(grid_size):
    for j in range(grid_size):
        Q[(i,j)] = [0]*4

# Reward function
def reward(state):
    return 100 if state == goal else -1

# Next state
def move(state, action):
    x, y = state
    dx, dy = action
    nx, ny = x+dx, y+dy
    
    if 0 <= nx < grid_size and 0 <= ny < grid_size:
        return (nx, ny)
    return state

# Q-Learning
alpha, gamma, epsilon = 0.1, 0.9, 0.2

for episode in range(200):
    state = (0,0)  # start
    
    while state != goal:
        # choose action (epsilon-greedy)
        if random.random() < epsilon:
            a = random.randint(0,3)
        else:
            a = np.argmax(Q[state])
        
        next_state = move(state, actions[a])
        r = reward(next_state)
        
        # update Q-valuen
        Q[state][a] += alpha * (r + gamma * max(Q[next_state]) - Q[state][a])
        
        state = next_state

# Test learned path
state = (0,0)
print("Path:")
while state != goal:
    print(state, end=" -> ")
    a = np.argmax(Q[state])
    state = move(state, actions[a])
print("Goal!")
