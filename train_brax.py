import jax
import jax.numpy as jnp
import brax
from brax.envs import State
from brax.envs.humanoid import Humanoid
import numpy as np
import time
import optax

def create_simple_policy(obs_dim, act_dim, hidden_dim=64):
    """Create a simple MLP policy."""
    def policy(params, obs):
        # Simple 2-layer MLP
        h = jax.nn.relu(jnp.dot(obs, params['w1']) + params['b1'])
        action = jnp.tanh(jnp.dot(h, params['w2']) + params['b2'])
        return action * act_dim  # Scale to action space
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    params = {
        'w1': jax.random.normal(key1, (obs_dim, hidden_dim)) * 0.01,
        'b1': jax.random.normal(key2, (hidden_dim,)) * 0.01,
        'w2': jax.random.normal(key3, (hidden_dim, act_dim)) * 0.01,
        'b2': jax.random.normal(key4, (act_dim,)) * 0.01,
    }
    
    return params, policy

def train_episode(env, policy, params, rng, max_steps=1000):
    """Train for one episode and return total reward."""
    state = env.reset(rng)
    total_reward = 0
    
    for step in range(max_steps):
        action = policy(params, state.obs)
        state = env.step(state, action)
        total_reward += state.reward
        
        if state.done.any():
            break
    
    return total_reward

def train_brax_humanoid():
    """Train humanoid using Brax environment."""
    print("=== Training with Brax ===")
    
    # Create environment
    env = Humanoid()
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_size}")
    print(f"Action space: {env.action_size}")
    
    # Create policy
    params, policy = create_simple_policy(env.observation_size, env.action_size)
    print("Policy network created")
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    episode_rewards = []
    rng = jax.random.PRNGKey(0)
    
    for episode in range(100):
        episode_rng = jax.random.split(rng, 1)[0]
        rng = jax.random.split(rng, 2)[1]
        
        reward = train_episode(env, policy, params, episode_rng)
        episode_rewards.append(reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Avg Reward (last 10): {avg_reward:.4f}")
    
    total_time = time.time() - start_time
    final_avg_reward = np.mean(episode_rewards[-20:])
    
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final average reward (last 20 episodes): {final_avg_reward:.4f}")
    
    return {
        'method': 'Brax',
        'total_time': total_time,
        'final_avg_reward': final_avg_reward,
        'episode_rewards': episode_rewards
    }

if __name__ == "__main__":
    train_brax_humanoid() 