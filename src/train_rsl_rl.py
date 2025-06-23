import torch
import numpy as np
import time
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from config import *
from common import BaseTrainer

class HumanoidVecEnv(VecEnv):
    """Vectorized Humanoid environment for RSL-RL."""
    
    def __init__(self, num_envs=2048):
        self.num_envs = num_envs
        self.envs = [HumanoidEnv() for _ in range(num_envs)]
        self.obs = np.zeros((num_envs, 376))  # Humanoid observation space
        self.rews = np.zeros(num_envs)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.actions = np.zeros((num_envs, 17))  # Humanoid action space
        
    def step(self, actions):
        # Convert tensor to numpy if needed
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions = actions.astype(np.float32)
            
        self.actions = actions
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, _ = env.step(actions[i])
            self.obs[i] = obs
            self.rews[i] = rew
            self.dones[i] = terminated or truncated
            
        return self.obs.copy(), self.rews.copy(), self.dones.copy(), {}
    
    def reset(self):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self.obs[i] = obs
        return self.obs.copy()
    
    def get_observations(self):
        return self.obs.copy()
    
    def get_rewards(self):
        return self.rews.copy()
    
    def get_dones(self):
        return self.dones.copy()
    
    def get_actions(self):
        return self.actions.copy()

class RSLRLTrainer(BaseTrainer):
    """RSL-RL trainer implementation."""
    
    def __init__(self):
        super().__init__("RSL-RL")
        self.env = None
        self.actor_critic = None
        self.ppo = None
        self.obs = None
    
    def setup_environment(self):
        """Setup the vectorized environment."""
        self.env = HumanoidVecEnv(NUM_ENVS)
        self.print_environment_info(
            f"HumanoidVecEnv with {NUM_ENVS} parallel environments",
            HUMANOID_OBSERVATION_SIZE,
            HUMANOID_ACTION_SIZE
        )
    
    def setup_policy(self):
        """Setup the actor-critic network and PPO algorithm."""
        # Create actor-critic network
        self.actor_critic = ActorCritic(
            num_actor_obs=HUMANOID_OBSERVATION_SIZE,
            num_critic_obs=HUMANOID_OBSERVATION_SIZE,
            num_actions=HUMANOID_ACTION_SIZE,
            actor_hidden_dims=ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=CRITIC_HIDDEN_DIMS,
            device='cpu'
        )
        self.print_policy_info("Actor-critic network")
        
        # Create PPO algorithm
        self.ppo = PPO(
            policy=self.actor_critic,
            device='cpu',
            **PPO_PARAMS
        )
        self.print_policy_info("PPO algorithm")
    
    def run_training_loop(self):
        """Run the main training loop."""
        self.obs = torch.tensor(self.env.reset(), dtype=torch.float32)
        
        for iteration in range(TRAINING_ITERATIONS):
            # Get actions from policy
            with torch.no_grad():
                actions = self.actor_critic.act_inference(self.obs)
            
            # Step environment
            obs_np, rewards, dones, _ = self.env.step(actions)
            self.obs = torch.tensor(obs_np, dtype=torch.float32)
            
            # Store experience
            self.episode_rewards.extend(rewards)
            
            # Log progress
            if iteration % LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-NUM_ENVS*LOG_INTERVAL:]) if self.episode_rewards else 0
                self.log_progress(iteration, avg_reward)
    
    def calculate_final_reward(self):
        """Calculate final average reward."""
        return np.mean(self.episode_rewards[-NUM_ENVS*20:]) if self.episode_rewards else 0

def train_rsl_rl_humanoid():
    """Train humanoid using RSL-RL with PPO."""
    trainer = RSLRLTrainer()
    return trainer.train()

if __name__ == "__main__":
    train_rsl_rl_humanoid() 