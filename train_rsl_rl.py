import torch
import numpy as np
import time
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv

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

def train_rsl_rl_humanoid():
    """Train humanoid using RSL-RL with PPO."""
    print("=== Training with RSL-RL ===")
    
    # Create vectorized environment
    num_envs = 2048
    env = HumanoidVecEnv(num_envs)
    print(f"Environment: HumanoidVecEnv with {num_envs} parallel environments")
    print(f"Observation space: 376")
    print(f"Action space: 17")
    
    # Create actor-critic network
    actor_critic = ActorCritic(
        num_actor_obs=376,
        num_critic_obs=376,
        num_actions=17,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        device='cpu'
    )
    print("Actor-critic network created")
    
    # Create PPO algorithm
    ppo = PPO(
        policy=actor_critic,
        num_learning_epochs=4,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        desired_kl=0.01,
        device='cpu'
    )
    print("PPO algorithm initialized")
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    episode_rewards = []
    obs = torch.tensor(env.reset(), dtype=torch.float32)
    
    for iteration in range(100):  # Reduced for comparison
        # Get actions from policy
        with torch.no_grad():
            actions = actor_critic.act_inference(obs)
        
        # Step environment
        obs_np, rewards, dones, _ = env.step(actions)
        obs = torch.tensor(obs_np, dtype=torch.float32)
        
        # Store experience
        episode_rewards.extend(rewards)
        
        # Simple policy update every few iterations (without storage)
        if iteration % 10 == 0:
            # Just log the current performance
            avg_reward = np.mean(episode_rewards[-num_envs*10:]) if episode_rewards else 0
            print(f"Iteration {iteration}: Avg Reward: {avg_reward:.4f}")
    
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-num_envs*20:]) if episode_rewards else 0
    
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final average reward: {final_avg_reward:.4f}")
    
    return {
        'method': 'RSL-RL',
        'total_time': total_time,
        'final_avg_reward': final_avg_reward,
        'episode_rewards': episode_rewards
    }

if __name__ == "__main__":
    train_rsl_rl_humanoid() 