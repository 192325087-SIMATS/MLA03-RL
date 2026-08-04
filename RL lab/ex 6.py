import numpy as np
import random

# Grid size (4x4)
size = 4
goal = (3, 3)

# Q-table
Q = np.zeros((size, size, 4))  # 4 actions

# Actions: up, down, left, right
actions = [(-1,0),(1,0),(0,-1),(0,1)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Reward function
def reward(state):
    return 100 if state == goal else -1

# Move function
def move(state, action):
    x, y = state
    dx, dy = actions[action]
    nx, ny = x+dx, y+dy
    
    if 0 <= nx < size and 0 <= ny < size:
        return (nx, ny)
    return state

# Training
for episode in range(200):
    state = (0, 0)
    
    while state != goal:
        if random.random() < epsilon:
            a = random.randint(0, 3)
        else:
            a = np.argmax(Q[state[0], state[1]])
        
        next_state = move(state, a)
        r = reward(next_state)
        
        Q[state[0], state[1], a] += alpha * (
            r + gamma * np.max(Q[next_state[0], next_state[1]]) 
            - Q[state[0], state[1], a]
        )
        
        state = next_state

# Test path
state = (0,0)
print("Path:")
while state != goal:
    print(state, end=" -> ")
    a = np.argmax(Q[state[0], state[1]])
    state = move(state, a)

print("Goal!")
