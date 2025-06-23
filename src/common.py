"""
Common base class for training scripts to ensure consistent behavior.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from config import *


class BaseTrainer(ABC):
    """Base class for training algorithms with common functionality."""

    def __init__(self, method_name):
        self.method_name = method_name
        self.start_time = None
        self.episode_rewards = []
        self.final_avg_reward = 0.0
        self.total_time = 0.0
        self.progress_callback = None

    def train(self, progress_callback=None):
        """Main training loop with common logging and timing."""
        print(f"=== Training with {self.method_name} ===")
        self.progress_callback = progress_callback

        # Setup
        self.setup_environment()
        self.setup_policy()

        # Training loop
        print("Starting training...")
        self.start_time = time.time()

        self.run_training_loop()

        # Finalize
        self.total_time = time.time() - self.start_time
        self.final_avg_reward = self.calculate_final_reward()

        print(f"Training completed in {self.total_time:.2f} seconds")
        print(f"Final average reward: {self.final_avg_reward:.4f}")

        return self.get_results()

    @abstractmethod
    def setup_environment(self):
        """Setup the training environment."""
        pass

    @abstractmethod
    def setup_policy(self):
        """Setup the policy/network."""
        pass

    @abstractmethod
    def run_training_loop(self):
        """Run the main training loop."""
        pass

    @abstractmethod
    def calculate_final_reward(self):
        """Calculate final average reward."""
        pass

    def log_progress(self, step, avg_reward, step_name="Iteration"):
        """Log training progress."""
        assert self.start_time is not None
        elapsed_time = time.time() - self.start_time
        print(f"{step_name} {step}: Avg Reward: {avg_reward:.4f}, Time: {elapsed_time:.2f}s")
        if self.progress_callback:
            self.progress_callback(
                step=step,
                avg_reward=avg_reward,
                elapsed_time=elapsed_time,
                step_name=step_name,
            )

    def get_results(self):
        """Get training results in standardized format."""
        return {
            "method": self.method_name,
            "total_time": self.total_time,
            "final_avg_reward": self.final_avg_reward,
            "episode_rewards": self.episode_rewards.copy(),
        }

    def print_environment_info(self, env_name, obs_size, act_size):
        """Print environment information."""
        print(f"Environment: {env_name}")
        print(f"Observation space: {obs_size}")
        print(f"Action space: {act_size}")

    def print_policy_info(self, policy_name):
        """Print policy information."""
        print(f"{policy_name} created")
