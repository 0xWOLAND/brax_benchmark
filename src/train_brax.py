import os

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper

from base_trainer import BaseTrainer
from constants import ENV_NAME, NUM_TIMESTEPS, NUM_EVALS, SEED


class BraxTrainer(BaseTrainer):
    def _train(self):
        config = registry.get_default_config(self.env_name)
        config.num_timesteps = NUM_TIMESTEPS
        config.num_evals = NUM_EVALS

        env = registry.load(self.env_name, config=config)

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
            log_training_metrics=True,
            training_metrics_steps=1,
            episode_length=config.episode_length,
            network_factory=ppo_networks.make_ppo_networks,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )


if __name__ == "__main__":
    trainer = BraxTrainer(ENV_NAME)
    trainer.train()
