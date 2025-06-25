import os
import jax
from base_trainer import BaseTrainer
from constants import ENV_NAME, NUM_ENVS, NUM_TIMESTEPS, SEED
from ml_collections import config_dict
from mujoco_playground import registry, wrapper_torch
from mujoco_playground.config import manipulation_params, locomotion_params
from rsl_rl.runners import OnPolicyRunner


def get_config(env_name: str) -> config_dict.ConfigDict:
    """Get RL configuration for the environment."""
    if env_name in registry.manipulation._envs:
        return manipulation_params.rsl_rl_config(env_name)
    elif env_name in registry.locomotion._envs:
        return locomotion_params.rsl_rl_config(env_name)
    else:
        raise ValueError(f"No RL config for {env_name}")


class RSLRLTrainer(BaseTrainer):
    def _train(self):
        device, device_rank = (
            ("cuda:0", 0)
            if any(d.platform == "gpu" for d in jax.devices())
            else ("cpu", 0)
        )
        config = registry.get_default_config(self.env_name)
        randomizer = registry.get_domain_randomizer(self.env_name)

        raw_env = registry.load(self.env_name, config=config)
        brax_env = wrapper_torch.RSLRLBraxWrapper(
            raw_env,
            NUM_ENVS,
            SEED,
            config.episode_length,
            1,
            randomization_fn=randomizer,
            device_rank=device_rank,
        )

        train_cfg = get_config(self.env_name)
        train_cfg.seed = SEED
        train_cfg.run_name = self.experiment_name
        train_cfg.resume = False
        train_cfg.load_run = "-1"
        train_cfg.checkpoint = -1

        runner = OnPolicyRunner(
            brax_env, train_cfg.to_dict(), str(self.logdir), device=device
        )

        runner.learn(num_learning_iterations=NUM_TIMESTEPS)


if __name__ == "__main__":
    trainer = RSLRLTrainer(ENV_NAME)

    trainer.train()
