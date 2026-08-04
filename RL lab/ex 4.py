# States (locations)
states = ["A", "B", "C", "Goal"]

# Actions (possible moves)
actions = {
    "A": ["B", "C"],
    "B": ["Goal"],
    "C": ["Goal"],
    "Goal": []
}

# Cost (negative reward)
cost = {
    ("A","B"): -1,
    ("A","C"): -5,
    ("B","Goal"): -2,
    ("C","Goal"): -1
}

# Initialize value function
V = {s: 0 for s in states}
gamma = 0.9

# Bellman Update
for _ in range(10):
    for s in states:
        if s == "Goal":
            V[s] = 0
        else:
            values = []
            for a in actions[s]:
                values.append(cost[(s,a)] + gamma * V[a])
            V[s] = max(values)

# Optimal Policy
policy = {}
for s in states:
    if s != "Goal":
        best = max(actions[s], key=lambda a: cost[(s,a)] + gamma * V[a])
        policy[s] = best

print("Values:", V)
print("Optimal Path:", policy)
