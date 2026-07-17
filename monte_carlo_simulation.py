import numpy as np

def estimate_pi(number_of_samples):
    # Generating random coordinates
    x = np.random.uniform(-1, 1, number_of_samples)
    y = np.random.uniform(-1, 1, number_of_samples)

    # Calculating distance
    distance = np.sqrt(x**2 + y**2)

    inside_circle = np.sum(distance <= 1)
    pi_estimate = (inside_circle / number_of_samples) * 4

    return pi_estimate

samples = 1000000000
estimated_pi = estimate_pi(samples)
print(f"Estimated value of pi with {samples} samples: {estimated_pi}")