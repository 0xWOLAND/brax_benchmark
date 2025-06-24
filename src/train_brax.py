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
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
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
        
        self.save_config(env_cfg)

        env = registry.load(self.env_name, config=env_cfg)
        
        training_params = dict(config)
        if "network_factory" in training_params:
            del training_params["network_factory"]
        
        network_factory = ppo_networks.make_ppo_networks

        def log_progress(step, metrics):
            self.log_performance(step, metrics)

        _ = ppo.train(
            environment=env,
            progress_fn=log_progress,
            network_factory=network_factory,
            seed=SEED,
            save_checkpoint_path=str(self.ckpt_path),
            wrap_env_fn=wrapper.wrap_for_brax_training,
            **training_params,
        )


if __name__ == "__main__":
    trainer = BraxTrainer(ENV_NAME)
    trainer.start_training()