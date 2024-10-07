import numpy as np

def load_data(filepath):
    """
    Loads data from the specified CSV file and extracts
    observations, actions, rewards, and terminals.

    Parameters:
    -----------
    filepath: str
        Path to the CSV file.

    Returns:
    --------
    Tuple (observations, actions, rewards, terminals)
    """
    data = np.loadtxt(open(filepath), delimiter=",")

    # Set indices for observation and action
    num_obs = 63
    num_actions = 7

    # Extract observations, actions, rewards, and timesteps
    observations = data[:, :num_obs]
    actions = data[:, num_obs:num_obs + num_actions]
    rewards = data[:, -2]
    timesteps = data[:, -1]

    return observations, actions, rewards, timesteps
