import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from tensorboardX import SummaryWriter


# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


class BaseTrainer(ABC):
    """Base class for training agents with common logging and timing functionality."""

    def __init__(self, env_name: str, experiment_name: Optional[str] = None):
        self.env_name = env_name
        self.experiment_name = (
            experiment_name or f"{env_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.start_time = None
        self.end_time = None

        self.logdir = Path.cwd() / "logs" / self.experiment_name
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.logdir / "checkpoints"
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.logdir))

        print(f"Environment: {env_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Logs stored in: {self.logdir}")

    def _start_tensorboard(self):
        try:
            log_dir = self.logdir.resolve()
            subprocess.Popen(
                ["tensorboard", "--logdir", str(log_dir), "--port", "0"],
            )

        except Exception as e:
            print(f"Warning: Could not start TensorBoard: {e}")
            print(f"Start manually with: tensorboard --logdir {self.logdir.resolve()}")

    def train(self):
        """Start training and record start time."""
        self.start_time = time.monotonic()
        print("Starting training...")

        self._start_tensorboard()

        self._train()
        self.end_time = time.monotonic()
        self._log_final_metrics()
        self.writer.close()

    def _log_final_metrics(self):
        """Log final training metrics including wall time."""
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            print(f"Training completed in {total_time:.2f} seconds")

            self.writer.add_scalar("training/total_wall_time_seconds", total_time, 0)

            final_metrics = {
                "total_wall_time_seconds": total_time,
                "experiment_name": self.experiment_name,
                "env_name": self.env_name,
                "timestamp": datetime.now().isoformat(),
            }

            metrics_file = self.logdir / "final_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as fp:
                json.dump(final_metrics, fp, indent=4)

            print(f"Final metrics saved to: {metrics_file}")

    @abstractmethod
    def _train(self):
        pass

    def __del__(self):
        self.writer.close()
