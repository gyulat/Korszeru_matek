import numpy as np
import pandas as pd
from scipy.stats import nbinom, beta

# Initialize parameters
n_socks = 18  # The total number of socks in the laundry
n_picked = 11  # The number of socks we are going to pick
n_pairs = 7  # for a total of 7*2=14 paired socks.
n_odd = 4

# Create socks array
socks = np.repeat(np.arange(1, n_pairs + n_odd + 1), [2] * n_pairs + [1] * n_odd)
np.random.shuffle(socks)  # Shuffle to simulate random arrangement

# Pick random socks
picked_socks = np.random.choice(socks, size=min(n_picked, n_socks), replace=False)

# Count occurrences of each sock
sock_counts = pd.Series(picked_socks).value_counts()

# Count unique socks and pairs
unique_count = np.sum(sock_counts == 1)
pairs_count = np.sum(sock_counts == 2)
print(f"Unique socks: {unique_count}, Pairs: {pairs_count}")

# Prior simulation
prior_mu = 30
prior_sd = 15
prior_size_param = -prior_mu**2 / (prior_mu - prior_sd**2)

n_socks = nbinom.rvs(n=prior_size_param, p=prior_mu / (prior_mu + prior_size_param), size=1)[0]
prop_pairs = beta.rvs(15, 2)
n_pairs = round(np.floor(n_socks / 2) * prop_pairs)
n_odd = n_socks - n_pairs * 2

# Function for simulating sock picking
def simulate_sock_picking(n_picked):
    prior_mu = 30
    prior_sd = 15
    prior_size_param = -prior_mu**2 / (prior_mu - prior_sd**2)
    n_socks = nbinom.rvs(n=prior_size_param, p=prior_mu / (prior_mu + prior_size_param), size=1)[0]
    prop_pairs = beta.rvs(15, 2)
    n_pairs = round(np.floor(n_socks / 2) * prop_pairs)
    n_odd = n_socks - n_pairs * 2

    # Create socks array
    socks = np.repeat(np.arange(1, n_pairs + n_odd + 1), [2] * n_pairs + [1] * n_odd)
    np.random.shuffle(socks)

    # Pick socks
    picked_socks = np.random.choice(socks, size=min(n_picked, n_socks), replace=False)
    sock_counts = pd.Series(picked_socks).value_counts()

    # Return counts and parameters
    return {
        'unique': np.sum(sock_counts == 1),
        'pairs': np.sum(sock_counts == 2),
        'n_socks': n_socks,
        'n_pairs': n_pairs,
        'n_odd': n_odd,
        'prop_pairs': prop_pairs
    }

# Simulating the process 100000 times
simulations = [simulate_sock_picking(n_picked) for _ in range(100000)]

# Convert to DataFrame for easier handling
sock_sim_df = pd.DataFrame(simulations)

# Filter for specific cases
post_samples_case1 = sock_sim_df[(sock_sim_df['unique'] == 11) & (sock_sim_df['pairs'] == 0)]
post_samples_case2 = sock_sim_df[(sock_sim_df['unique'] == 3) & (sock_sim_df['pairs'] == 4)]

# Print the first few rows of the simulations
print(sock_sim_df.head())

