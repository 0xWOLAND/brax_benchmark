"""
Common configuration for fair comparison between RSL-RL and Brax training.
"""

# Training parameters
TRAINING_ITERATIONS = 100
LOG_INTERVAL = 10

# Environment parameters
HUMANOID_OBSERVATION_SIZE = 244
HUMANOID_ACTION_SIZE = 17

# Network architecture (shared between RSL-RL and Brax)
HIDDEN_DIMS = [512, 256, 128]  # Same architecture for both methods

# PPO parameters (for RSL-RL)
PPO_PARAMS = {
    'num_learning_epochs': 4,
    'num_mini_batches': 4,
    'clip_param': 0.2,
    'gamma': 0.99,
    'lam': 0.95,
    'value_loss_coef': 1.0,
    'learning_rate': 3e-4,
    'max_grad_norm': 1.0,
    'use_clipped_value_loss': True,
    'desired_kl': 0.01,
}

# Brax parameters
BRAX_MAX_STEPS = 1000

# Parallel environments (for RSL-RL)
NUM_ENVS = 2048

# Random seed for reproducibility
RANDOM_SEED = 42 