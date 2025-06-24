# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Minimal training script using RSL-RL for the specified environment."""

import os

# Set environment variables for GPU support
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime
import json

from absl import logging
from ml_collections import config_dict
from rsl_rl.runners import OnPolicyRunner
import torch

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

from constants import ENV_NAME, NUM_TIMESTEPS, NUM_EVALS, SEED

# Suppress logs
logging.set_verbosity(logging.WARNING)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Get RL configuration for the environment."""
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config for {env_name}")


def main():
  # Detect available devices and set parameters
  import jax
  available_devices = jax.devices()
  print(f"Available devices: {available_devices}")
  
  if any(device.platform == 'gpu' for device in available_devices):
    device = "cuda:0"
    device_rank = 0
    print("Using GPU for training")
  else:
    device = "cpu"
    device_rank = 0
    print("Using CPU for training")
  
  num_envs = 4096

  # Load default config from registry
  env_cfg = registry.get_default_config(ENV_NAME)
  print(f"Environment: {ENV_NAME}")
  print(f"Environment config:\n{env_cfg}")

  # Generate experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{ENV_NAME}-{timestamp}"
  print(f"Experiment name: {exp_name}")

  # Setup logging directory
  logdir = os.path.abspath(os.path.join("logs", exp_name))
  os.makedirs(logdir, exist_ok=True)
  print(f"Logs stored in: {logdir}")

  # Setup checkpoint directory
  ckpt_path = os.path.join(logdir, "checkpoints")
  os.makedirs(ckpt_path, exist_ok=True)

  # Save environment config
  with open(os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # Get domain randomizer
  randomizer = registry.get_domain_randomizer(ENV_NAME)

  # Create environment
  raw_env = registry.load(ENV_NAME, config=env_cfg)
  brax_env = wrapper_torch.RSLRLBraxWrapper(
      raw_env,
      num_envs,
      SEED,
      env_cfg.episode_length,
      1,
      randomization_fn=randomizer,
      device_rank=device_rank,
  )

  # Build RSL-RL config
  train_cfg = get_rl_config(ENV_NAME)
  train_cfg.seed = SEED
  train_cfg.run_name = exp_name
  train_cfg.resume = False
  train_cfg.load_run = "-1"
  train_cfg.checkpoint = -1

  # Create runner
  train_cfg_dict = train_cfg.to_dict()
  runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

  # Train
  print("Starting training...")
  max_iterations = getattr(train_cfg, 'max_iterations', 1000)
  if hasattr(max_iterations, '__int__'):
    max_iterations = int(max_iterations)
  else:
    max_iterations = 1000  # Default fallback
    
  runner.learn(
      num_learning_iterations=max_iterations,
      init_at_random_ep_len=False,
  )
  print("Training completed.")


if __name__ == "__main__":
  main()
  