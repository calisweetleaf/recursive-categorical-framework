import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import yaml


@dataclass
class StageResult:
    name: str
    status: str
    duration_sec: float
    details: str
    metrics: Dict[str, Any]
    error: Optional[str] = None


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def get_env_info() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(vm.total / (1024 ** 3), 3),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def configure_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(f"bco_test.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers = [file_handler, stream_handler]
    return logger


class TerminalTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def capture_terminal(path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        tee = TerminalTee(sys.stdout, handle)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"count": 0}
    data = np.array(latencies)
    return {
        "count": int(len(latencies)),
        "mean_ms": float(np.mean(data) * 1000),
        "median_ms": float(np.median(data) * 1000),
        "p90_ms": float(np.percentile(data, 90) * 1000),
        "p95_ms": float(np.percentile(data, 95) * 1000),
        "p99_ms": float(np.percentile(data, 99) * 1000),
        "min_ms": float(np.min(data) * 1000),
        "max_ms": float(np.max(data) * 1000),
    }


def build_base_config() -> Dict[str, Any]:
    return {
        "model_params": {
            "d_model": {
                "value": 512,
                "bayesian_metadata": {
                    "distribution_type": "normal",
                    "distribution_params": {"loc": 512.0, "scale": 64.0},
                    "constraints": {"min_value": 256.0, "max_value": 2048.0}
                }
            },
            "dropout": {
                "value": 0.1,
                "bayesian_metadata": {
                    "distribution_type": "beta",
                    "distribution_params": {"alpha": 2.0, "beta": 5.0},
                    "constraints": {"min_value": 0.0, "max_value": 1.0}
                }
            },
            "n_layers": {
                "value": 12,
                "bayesian_metadata": {
                    "distribution_type": "uniform",
                    "distribution_params": {"low": 6.0, "high": 24.0},
                    "constraints": {"min_value": 6.0, "max_value": 24.0}
                }
            }
        },
        "generation_params": {
            "temperature": {
                "value": 0.8,
                "bayesian_metadata": {
                    "distribution_type": "beta",
                    "distribution_params": {"alpha": 3.0, "beta": 2.0},
                    "constraints": {"min_value": 0.1, "max_value": 2.0}
                }
            },
            "top_k": {
                "value": 40,
                "bayesian_metadata": {
                    "distribution_type": "uniform",
                    "distribution_params": {"low": 10.0, "high": 100.0},
                    "constraints": {"min_value": 10.0, "max_value": 100.0}
                }
            }
        },
        "training_params": {
            "learning_rate": {
                "value": 0.001,
                "bayesian_metadata": {
                    "distribution_type": "gamma",
                    "distribution_params": {"shape": 2.0, "scale": 0.0005},
                    "constraints": {"min_value": 1e-5, "max_value": 1e-1}
                }
            }
        }
    }


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def stage_banner(title: str, index: Optional[int] = None) -> None:
    label = f"[STAGE {index}] {title}" if index is not None else title
    print(label)
    print("-" * 60)


def render_stage_summary(stages: List[StageResult]) -> str:
    lines = ["## Stage Summary", "", "| Stage | Status | Duration | Details |", "| --- | --- | --- | --- |"]
    for stage in stages:
        lines.append(
            f"| {stage.name} | {stage.status} | {format_duration(stage.duration_sec)} | {stage.details} |"
        )
    return "\n".join(lines)
