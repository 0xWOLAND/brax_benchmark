import json
import os
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params
from constants import ENV_NAME, NUM_TIMESTEPS, NUM_EVALS, SEED

# Environment settings
EXPERIMENT_NAME = "minimal_ppo_training"

# Environment setup
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
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


def main():
  """Run minimal PPO training."""
  
  # Load environment configuration
  env_cfg = registry.get_default_config(ENV_NAME)
  ppo_params = get_rl_config(ENV_NAME)
  
  # Override with our settings
  ppo_params.num_timesteps = NUM_TIMESTEPS
  ppo_params.num_evals = NUM_EVALS
  
  print(f"Environment: {ENV_NAME}")
  print(f"Training for {NUM_TIMESTEPS} timesteps")
  
  # Load environment
  env = registry.load(ENV_NAME, config=env_cfg)
  
  # Set up logging directory
  logdir = epath.Path("logs").resolve() / EXPERIMENT_NAME
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs stored in: {logdir}")
  
  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  
  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)
  
  # Prepare training parameters
  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]
  
  # Set up network factory
  network_factory = ppo_networks.make_ppo_networks
  
  # Progress function for logging
  def progress(num_steps, metrics):
    if "eval/episode_reward" in metrics:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
  
  # Train the model
  print("Starting training...")
  start_time = time.monotonic()
  
  make_inference_fn, params, _ = ppo.train(
      environment=env,
      progress_fn=progress,
      network_factory=network_factory,
      seed=SEED,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      **training_params,
  )
  
  training_time = time.monotonic() - start_time
  print(f"Training completed in {training_time:.2f} seconds")
  
  # Run evaluation
  print("Running evaluation...")
  
  # Create inference function
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  
  # Load evaluation environment
  eval_env = registry.load(ENV_NAME, config=env_cfg)
  eval_env = wrapper.wrap_for_brax_training(eval_env)
  
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  
  # Run evaluation rollout
  rng = jax.random.PRNGKey(123)
  state = jit_reset(rng)
  rollout = [state]
  
  for _ in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if state.done:
      break
  
  # Render and save the rollout
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"Rendering at {fps} FPS")
  
  traj = rollout[::render_every]
  
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  
  frames = eval_env.render(
      traj, height=480, width=640, scene_option=scene_option
  )
  media.write_video("rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'")


if __name__ == "__main__":
  main()