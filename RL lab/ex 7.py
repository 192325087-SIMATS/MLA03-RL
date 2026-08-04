# States (locations)
states = ["A", "B", "C", "Goal"]

# Actions (routes)
actions = {
    "A": ["B", "C"],
    "B": ["Goal"],
    "C": ["Goal"],
    "Goal": []
}

# Rewards (negative = travel cost)
rewards = {
    ("A","B"): -1,
    ("A","C"): -4,
    ("B","Goal"): -2,
    ("C","Goal"): -1
}

gamma = 0.9

# Initialize value function
V = {s: 0 for s in states}

# Value Iteration (DP)
for _ in range(10):
    new_V = V.copy()
    for s in states:
        if s == "Goal":
            new_V[s] = 0
        else:
            values = []
            for a in actions[s]:
                values.append(rewards[(s,a)] + gamma * V[a])
            new_V[s] = max(values)
    V = new_V

# Optimal Policy
policy = {}
for s in states:
    if s != "Goal":
        best = max(actions[s],
                   key=lambda a: rewards[(s,a)] + gamma * V[a])
        policy[s] = best

print("Optimal Values:", V)
print("Optimal Policy:", policy)
