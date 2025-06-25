# PPO Framework Benchmark [![CI](https://github.com/0xWOLAND/brax_benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/0xWOLAND/brax_benchmark/actions/workflows/ci.yml)

A simple benchmark comparing [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl) for training a humanoid robot using PPO.

## Usage

```bash
uv sync
uv run src/train_brax.py
uv run src/train_rsl_rl.py
```

Both implementations train a humanoid robot for 1,000,000 iterations with identical hyperparameters for fair comparison.



