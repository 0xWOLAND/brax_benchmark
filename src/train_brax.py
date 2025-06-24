import os

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
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
        env_cfg = registry.get_default_config(self.env_name)
        config = get_config(self.env_name)
        config.num_timesteps = NUM_TIMESTEPS
        config.num_evals = NUM_EVALS

        env = registry.load(self.env_name, config=env_cfg)

        def log_progress(step, metrics):
            print(f"step: {step}, metrics: {metrics}")
            for key, value in metrics.items():
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"training/{key}", value, step)

        _ = ppo.train(
            environment=env,
            progress_fn=log_progress,
            seed=SEED,
            num_timesteps=NUM_TIMESTEPS,
            training_metrics_steps=1,
            episode_length=config.episode_length,
            network_factory=ppo_networks.make_ppo_networks,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

        self.writer.add_graph(model, dummy_input)


if __name__ == "__main__":
    trainer = BraxTrainer(ENV_NAME)
    trainer.train()
