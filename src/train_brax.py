import os

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from ml_collections import config_dict

import mujoco_playground
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

from base_trainer import BaseTrainer
from constants import ENV_NAME, NUM_TIMESTEPS, NUM_EVALS, SEED

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


def get_config(env_name: str) -> config_dict.ConfigDict:
    """Get default PPO config for the environment."""
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name)
    elif env_name in mujoco_playground.locomotion._envs:
        from mujoco_playground.config import locomotion_params

        return locomotion_params.brax_ppo_config(env_name)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        from mujoco_playground.config import dm_control_suite_params

        return dm_control_suite_params.brax_ppo_config(env_name)

    raise ValueError(f"Environment {env_name} not found.")


class BraxTrainer(BaseTrainer):
    def _train_implementation(self):
        env = registry.load(self.env_name)
        network_factory = ppo_networks.make_ppo_networks

        def log_progress(step, metrics):
            # self.log_performance(step, metrics)
            print("step: ", step)

        _ = ppo.train(
            environment=env,
            progress_fn=log_progress,
            network_factory=network_factory,
            seed=SEED,
            # save_checkpoint_path=str(self.ckpt_path),
            wrap_env_fn=wrapper.wrap_for_brax_training,
            num_timesteps=NUM_TIMESTEPS,
            episode_length=1000,
        )


if __name__ == "__main__":
    trainer = BraxTrainer(ENV_NAME)
    trainer.start_training()
