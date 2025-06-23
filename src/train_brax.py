from brax.envs.humanoid import Humanoid
from brax.training.agents.ppo.train import train as ppo_train
import numpy as np
from config import *
from common import BaseTrainer


class BraxTrainer(BaseTrainer):
    """Brax trainer implementation using built-in PPO."""

    def __init__(self):
        super().__init__("Brax")
        self.env = None
        self.make_policy = None
        self.params = None
        self.metrics = None

    def setup_environment(self):
        self.env = Humanoid()
        self.print_environment_info(
            self.env.__class__.__name__, HUMANOID_OBSERVATION_SIZE, HUMANOID_ACTION_SIZE
        )

    def setup_policy(self):
        self.print_policy_info("Built-in PPO networks")

    def run_training_loop(self):
        if self.env is None:
            raise ValueError("Environment not initialized")

        print("Starting PPO training with Brax...")
        total_timesteps = TRAINING_ITERATIONS * BRAX_MAX_STEPS

        compatible_batch_size = NUM_ENVS // PPO_PARAMS["num_mini_batches"]

        self.make_policy, self.params, self.metrics = ppo_train(
            environment=self.env,
            num_timesteps=total_timesteps,
            num_envs=NUM_ENVS,
            episode_length=BRAX_MAX_STEPS,
            learning_rate=PPO_PARAMS["learning_rate"],
            entropy_cost=0.01,
            discounting=PPO_PARAMS["gamma"],
            batch_size=compatible_batch_size,
            num_minibatches=PPO_PARAMS["num_mini_batches"],
            num_updates_per_batch=PPO_PARAMS["num_learning_epochs"],
            unroll_length=10,
            normalize_observations=True,
            reward_scaling=1.0,
            clipping_epsilon=PPO_PARAMS["clip_param"],
            gae_lambda=PPO_PARAMS["lam"],
            normalize_advantage=True,
            max_grad_norm=PPO_PARAMS["max_grad_norm"],
            seed=RANDOM_SEED,
            num_evals=10,
            run_evals=True,
            progress_fn=self._progress_callback,
        )

        if self.metrics and "eval/episode_reward" in self.metrics:
            self.episode_rewards = self.metrics["eval/episode_reward"].tolist()
        else:
            self.episode_rewards = [0.0] * TRAINING_ITERATIONS

    def _progress_callback(self, timestep, metrics):
        if "eval/episode_reward" in metrics:
            reward = float(metrics["eval/episode_reward"])
            self.episode_rewards.append(reward)

            if len(self.episode_rewards) % LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-LOG_INTERVAL:])
                self.log_progress(timestep, avg_reward, step_name="Timestep")

    def calculate_final_reward(self):
        if self.episode_rewards:
            return np.mean(self.episode_rewards[-20:])
        return 0.0


def train_brax_humanoid():
    trainer = BraxTrainer()
    return trainer.train()


if __name__ == "__main__":
    train_brax_humanoid()
