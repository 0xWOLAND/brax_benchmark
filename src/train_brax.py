import jax
import jax.numpy as jnp
import brax
from brax.envs import State
from brax.envs.humanoid import Humanoid
import numpy as np
import time
import optax
from config import *
from common import BaseTrainer

def create_simple_policy(obs_dim, act_dim, hidden_dims=HIDDEN_DIMS):
    """Create a policy with the same architecture as RSL-RL."""
    def policy(params, obs):
        # 3-layer MLP matching RSL-RL architecture
        h1 = jax.nn.relu(jnp.dot(obs, params['w1']) + params['b1'])
        h2 = jax.nn.relu(jnp.dot(h1, params['w2']) + params['b2'])
        h3 = jax.nn.relu(jnp.dot(h2, params['w3']) + params['b3'])
        action = jnp.tanh(jnp.dot(h3, params['w_out']) + params['b_out'])
        return action * act_dim  # Scale to action space
    
    # Initialize parameters for 3-layer network
    key = jax.random.PRNGKey(RANDOM_SEED)
    keys = jax.random.split(key, 8)  # 4 weights + 4 biases
    
    params = {
        'w1': jax.random.normal(keys[0], (obs_dim, hidden_dims[0])) * 0.01,
        'b1': jax.random.normal(keys[1], (hidden_dims[0],)) * 0.01,
        'w2': jax.random.normal(keys[2], (hidden_dims[0], hidden_dims[1])) * 0.01,
        'b2': jax.random.normal(keys[3], (hidden_dims[1],)) * 0.01,
        'w3': jax.random.normal(keys[4], (hidden_dims[1], hidden_dims[2])) * 0.01,
        'b3': jax.random.normal(keys[5], (hidden_dims[2],)) * 0.01,
        'w_out': jax.random.normal(keys[6], (hidden_dims[2], act_dim)) * 0.01,
        'b_out': jax.random.normal(keys[7], (act_dim,)) * 0.01,
    }
    
    return params, policy

class BraxTrainer(BaseTrainer):
    """Brax trainer implementation."""
    
    def __init__(self):
        super().__init__("Brax")
        self.env = None
        self.params = None
        self.policy = None
        self.rng = None
    
    def setup_environment(self):
        """Setup the Brax environment."""
        self.env = Humanoid()
        self.print_environment_info(
            self.env.__class__.__name__,
            self.env.observation_size,
            self.env.action_size
        )
    
    def setup_policy(self):
        """Setup the policy network with same architecture as RSL-RL."""
        if self.env is None:
            raise ValueError("Environment must be set up before policy")
            
        self.params, self.policy = create_simple_policy(
            self.env.observation_size, 
            self.env.action_size,
            HIDDEN_DIMS
        )
        self.print_policy_info("Policy network (3-layer MLP matching RSL-RL)")
    
    def train_episode(self, max_steps=BRAX_MAX_STEPS):
        """Train for one episode and return total reward."""
        if self.rng is None:
            self.rng = jax.random.PRNGKey(RANDOM_SEED)
            
        # Reset environment with proper RNG
        state = self.env.reset(self.rng)
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            action = self.policy(self.params, state.obs)
            
            # Step environment
            state = self.env.step(state, action)
            total_reward += state.reward
            
            # Check if episode is done
            if state.done.any():
                break
        
        return total_reward
    
    def run_training_loop(self):
        """Run the main training loop."""
        self.rng = jax.random.PRNGKey(RANDOM_SEED)
        
        for episode in range(TRAINING_ITERATIONS):
            # Split RNG for this episode
            episode_rng = jax.random.split(self.rng, 1)[0]
            self.rng = jax.random.split(self.rng, 2)[1]
            
            # Train one episode
            reward = self.train_episode()
            self.episode_rewards.append(reward)
            
            # Log progress
            if episode % LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-LOG_INTERVAL:])
                self.log_progress(episode, avg_reward)
    
    def calculate_final_reward(self):
        """Calculate final average reward."""
        return np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0

def train_brax_humanoid():
    """Train humanoid using Brax environment."""
    trainer = BraxTrainer()
    return trainer.train()

if __name__ == "__main__":
    train_brax_humanoid() 