import random

# States (rooms)
states = ["Dirty", "Clean"]

# Actions
actions = ["clean", "move"]

# Initialize values
returns = {s: [] for s in states}
V = {s: 0 for s in states}

Q = {(s,a): 0 for s in states for a in actions}
N = {(s,a): 0 for s in states for a in actions}

gamma = 0.9
epsilon = 0.2

# Episode simulation
def generate_episode():
    episode = []
    state = random.choice(states)
    
    for _ in range(5):
        # ε-greedy action
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda a: Q[(state,a)])
        
        # Reward
        reward = 10 if state == "Dirty" and action == "clean" else -1
        
        # Next state
        next_state = "Clean" if action == "clean" else random.choice(states)
        
        episode.append((state, action, reward))
        state = next_state
    
    return episode

# Monte Carlo Learning
for _ in range(100):
    episode = generate_episode()
    G = 0
    
    for state, action, reward in reversed(episode):
        G = gamma * G + reward
        
        N[(state,action)] += 1
        Q[(state,action)] += (G - Q[(state,action)]) / N[(state,action)]
        
        returns[state].append(G)
        V[state] = sum(returns[state]) / len(returns[state])

# Output
print("State Values:", V)
print("Q Values:", Q)
