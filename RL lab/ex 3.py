import random

# States (locations)
states = ["Start", "Shelf", "Packing", "Goal"]

# Actions
actions = ["move_forward", "pick", "drop"]

# Rewards
rewards = {
    "Goal": 100,
    "Packing": 10,
    "Shelf": -1,
    "Start": -1
}

# Transition function (probabilistic)
def transition(state, action):
    if state == "Start":
        return "Shelf"
    elif state == "Shelf":
        if action == "pick":
            return "Packing"
        else:
            return random.choice(["Shelf", "Start"])
    elif state == "Packing":
        if action == "drop":
            return "Goal"
        else:
            return "Packing"
    return state

# Value Iteration
V = {s: 0 for s in states}
gamma = 0.9

for _ in range(10):
    new_V = V.copy()
    for s in states:
        values = []
        for a in actions:
            next_state = transition(s, a)
            r = rewards[next_state]
            values.append(r + gamma * V[next_state])
        new_V[s] = max(values)
    V = new_V

# Optimal Policy
policy = {}
for s in states:
    best_action = None
    best_value = float("-inf")
    for a in actions:
        next_state = transition(s, a)
        val = rewards[next_state] + gamma * V[next_state]
        if val > best_value:
            best_value = val
            best_action = a
    policy[s] = best_action

# Output
print("Values:", V)
print("Policy:", policy)
