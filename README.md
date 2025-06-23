# PPO Framework Benchmark

A simple benchmark comparing [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl) for training a humanoid robot using PPO.

## Usage

```bash
uv sync
uv run src/train_brax.py
uv run src/train_rsl_rl.py
```

Both implementations train a humanoid robot for 100 iterations with identical hyperparameters for fair comparison.



