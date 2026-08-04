import numpy as np
import random

# Grid (3x3)
size = 3
goal = (2,2)

# Actions: up, down, left, right
actions = [(-1,0),(1,0),(0,-1),(0,1)]

# Q-table (acts like simple DQN)
Q = np.zeros((size, size, 6, 4))  # x, y, battery(0-5), actions

alpha, gamma, epsilon = 0.1, 0.9, 0.2

# Move
def move(state, a):
    x,y,b = state
    dx,dy = actions[a]
    nx,ny = x+dx, y+dy
    
    if 0 <= nx < size and 0 <= ny < size:
        return (nx,ny,b-1)
    return (x,y,b-1)

# Reward
def reward(state):
    x,y,b = state
    if (x,y) == goal:
        return 100
    if b <= 0:
        return -100
    return -1

# Training
for ep in range(200):
    state = (0,0,5)
    
    while True:
        x,y,b = state
        
        # ε-greedy
        if random.random() < epsilon:
            a = random.randint(0,3)
        else:
            a = np.argmax(Q[x,y,b])
        
        next_state = move(state, a)
        nx,ny,nb = next_state
        r = reward(next_state)
        
        # Q-learning update (DQN idea simplified)
        Q[x,y,b,a] += alpha * (
            r + gamma * np.max(Q[nx,ny,max(nb,0)]) - Q[x,y,b,a]
        )
        
        state = next_state
        
        if r == 100 or r == -100:
            break

# Test
state = (0,0,5)
print("Path:")
while True:
    x,y,b = state
    print(state, end=" -> ")
    
    a = np.argmax(Q[x,y,b])
    state = move(state, a)
    
    if (state[0],state[1]) == goal:
        print("Delivered!")
        break
    if state[2] <= 0:
        print("Battery Dead!")
        break
