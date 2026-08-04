import random

# States (simplified positions)
states = ["start", "middle", "win", "lose"]

# Actions
actions = ["attack", "defend"]

# Rewards
rewards = {
    "win": 100,
    "lose": -100,
    "middle": -1,
    "start": -1
}

# Transition function
def transition(state, action):
    if state == "start":
        return "middle"
    elif state == "middle":
        if action == "attack":
            return random.choice(["win", "lose"])
        else:
            return "middle"
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

# Policy
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
