import numpy as np


def simulate_binomial_experiment(p, n, num_trials):
    # Simulate num_trials experiments of n coin flips each
    results = np.random.binomial(n, p, num_trials)
    return np.mean(results)


# Parameters
probability_heads = 0.3
num_trials = 10000  # Number of experiments to average

# Values of n to test
n_values = [10, 100, 1000]

# Results storage
results = {}

for n in n_values:
    average_X = simulate_binomial_experiment(probability_heads, n, num_trials)
    np_value = n * probability_heads
    results[n] = (average_X, np_value)

# Display results
for n, (average_X, np_value) in results.items():
    print(f"n = {n}:")
    print(f"  Average of X = {average_X:.2f}")
    print(f"  np = {np_value:.2f}")
    print()
