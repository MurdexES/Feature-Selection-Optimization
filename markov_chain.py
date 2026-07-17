import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

states = ['Working', 'Degraded', 'Failed']

transition_matrix = np.array([
    [0.9, 0.0, 0.1],
    [0.6, 0.2, 0.2],
    [0.3, 0.2, 0.5],
])

costs = {
    'Working': 10,
    'Degraded': 200,
    'Failed': 1200,
}

def simulate_equipment(transition_matrix, states, n_days, repair_state='Working', seed=42):
    np.random.seed(seed)
    state_index = {s: i for i, s in enumerate(states)}

    current = 'Working'
    history = [current]
    repairs = []

    for day in range(n_days):
        if current == 'Failed':
            # Repair — return to Working
            current = 'Working'
            repairs.append(day)
        else:
            idx = state_index[current]
            current = np.random.choice(states, p=transition_matrix[idx])
        history.append(current)

    return history, repairs

history, repairs = simulate_equipment(transition_matrix, states, n_days=365)

# ── Compute costs ─────────────────────────────────────────────────────────
repair_cost = 15000   # fixed cost per repair event
daily_costs = [costs[state] for state in history]
total_operating = sum(daily_costs)
total_repair = len(repairs) * repair_cost
total_cost = total_operating + total_repair

print("=" * 45)
print("EQUIPMENT SIMULATION REPORT — 1 YEAR")
print("=" * 45)
print(f"Total days simulated: {len(history)}")
print(f"\nDays in each state:")
c = Counter(history)
for state in states:
    pct = c[state] / len(history) * 100
    print(f"  {state:10s}: {c[state]:3d} days ({pct:.1f}%)")

print(f"\nNumber of failure events: {len(repairs)}")
print(f"\nCost breakdown:")
print(f"  Operating costs:  ${total_operating:>10,.0f}")
print(f"  Repair costs:     ${total_repair:>10,.0f}")
print(f"  Total cost:       ${total_cost:>10,.0f}")
print(f"  Cost per day:     ${total_cost/len(history):>10,.0f}")