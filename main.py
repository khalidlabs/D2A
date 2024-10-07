import d3rlpy
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import (
    average_value_estimation_scorer,
    discounted_sum_of_advantage_scorer,
    initial_state_value_estimation_scorer
)
from config import DATA_PATH, MODEL_SAVE_PATH, SEED, FEATURE_SIZE, EPOCHS, TEST_SIZE
from data_processing import load_data
from custom_encoder import CustomEncoderFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Set random seed
    d3rlpy.seed(SEED)
    np.random.seed(SEED)

    # Load data
    data = load_data(DATA_PATH)

    # Extract data fields
    observations, actions, rewards, terminals = data

    # Create MDP Dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )

    # Split dataset into training and testing
    train_episodes, test_episodes = train_test_split(dataset, test_size=TEST_SIZE)

    # Initialize IQL with custom encoder
    iql = d3rlpy.algos.IQL(
        encoder_factory=CustomEncoderFactory(feature_size=FEATURE_SIZE),
        scaler='min_max',
        action_scaler='min_max',
        reward_scaler='min_max'
    )

    # Train the model
    logger.info("Starting training...")
    iql.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_epochs=EPOCHS,
        scorers={
            'value_scale': average_value_estimation_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'initial_value': initial_state_value_estimation_scorer
        }
    )

    # Save the trained policy
    iql.save_policy(MODEL_SAVE_PATH)
    logger.info(f"Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
