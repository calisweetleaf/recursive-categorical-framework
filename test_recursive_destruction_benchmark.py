#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Destruction Benchmark

Stress-tests RecursiveTensor, StabilityMatrix, and TriaxialBackbone with
contradiction-heavy recursion depth, oscillatory fixed-point pressure, and
triaxial instability recovery tracking.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from rcf_integration.recursive_tensor import RecursiveTensor
from rcf_integration.stability_matrix import EigenrecursionStabilizer, RecursionDetector
from rcf_integration.ethical_tensor import BreathPhase
from triaxial_backbone import TriaxialConfig, TriaxialField

LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = BASE_DIR / "figures"

LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"recursive_destruction_{RUN_TS}.log"
JSON_FILE = LOG_DIR / f"recursive_destruction_{RUN_TS}.json"
REPORT_FILE = REPORT_DIR / f"recursive_destruction_report_{RUN_TS}.md"


def setup_logger() -> logging.Logger:
    """Configure logging to both console and file."""
    logger = logging.getLogger("RecursiveDestructionBenchmark")
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logger()


@dataclass
class StageResult:
    """Result from a single benchmark stage."""

    stage_num: int
    name: str
    success: bool
    duration_s: float
    details: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class LLMProxy:
    """
    Minimal autoregressive proxy to emulate LLM-style recursion pressure.

    This is a local baseline (no external model) that helps compare
    contradiction handling using the same input sequence.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        seed: int,
        decay: float = 0.85,
        noise_scale: float = 0.02,
        contradiction_bias: float = 0.6,
    ) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.decay = decay
        self.noise_scale = noise_scale
        self.contradiction_bias = contradiction_bias
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros(state_dim, dtype=np.float32)
        self.weight = self.rng.normal(0.0, 0.6, size=(state_dim, state_dim)).astype(np.float32)
        self.input_weight = self.rng.normal(0.0, 0.4, size=(state_dim, input_dim)).astype(
            np.float32
        )
        self.bias = self.rng.normal(0.0, 0.1, size=state_dim).astype(np.float32)

    def step(self, input_vec: np.ndarray, contradiction: bool) -> np.ndarray:
        """
        Advance the proxy state with an autoregressive update.

        Args:
            input_vec: Input stimulus vector.
            contradiction: Whether to apply a contradiction inversion bias.

        Returns:
            Updated state vector.
        """
        if input_vec.shape[0] != self.input_dim:
            if input_vec.shape[0] < self.input_dim:
                pad = np.zeros(self.input_dim - input_vec.shape[0], dtype=np.float32)
                input_vec = np.concatenate([input_vec.astype(np.float32), pad])
            else:
                input_vec = input_vec[: self.input_dim].astype(np.float32)

        raw = self.weight @ self.state + self.input_weight @ input_vec + self.bias
        if contradiction:
            raw = raw * (-1.0 * self.contradiction_bias)

        raw = (1.0 - self.decay) * self.state + self.decay * raw
        raw += self.rng.normal(0.0, self.noise_scale, size=self.state.shape).astype(np.float32)

        self.state = np.tanh(raw).astype(np.float32)
        return self.state


def seed_all(seed: int) -> None:
    """Seed numpy and torch for deterministic runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_dense_array(tensor: RecursiveTensor) -> np.ndarray:
    """Return a dense numpy array for metrics."""
    if isinstance(tensor.data, dict):
        return tensor._sparse_to_dense(tensor.data)
    if hasattr(tensor.data, "toarray"):
        return tensor.data.toarray()
    return np.array(tensor.data)


def build_transform_matrix(dim: int, scale: float, jitter: float, seed: int) -> np.ndarray:
    """Create a controlled linear transform with mild rotation and jitter."""
    rng = np.random.default_rng(seed)
    base = np.eye(dim, dtype=np.float32) * scale
    roll = np.roll(np.eye(dim, dtype=np.float32), shift=1, axis=0) * scale
    noise = rng.normal(0.0, jitter, size=(dim, dim)).astype(np.float32)
    return 0.6 * base + 0.4 * roll + noise


def contradiction_c_function(idx: Optional[Tuple[int, ...]], z_val: Any) -> Any:
    """Generate alternating contradiction parameters for fractal iteration."""
    if idx is None:
        indices = np.indices(z_val.shape)
        parity = np.sum(indices, axis=0) % 2
        return 0.35 * np.where(parity == 0, 1.0, -1.0)
    parity = sum(idx) % 2
    return 0.35 if parity == 0 else -0.35


def stabilize_tensor_values(
    tensor: RecursiveTensor,
    clamp: float,
) -> Tuple[RecursiveTensor, bool]:
    """
    Clamp/repair non-finite tensor values to keep recursion observable.

    This preserves destructive behavior while preventing NaN/Inf cascades
    from collapsing eigenrecursion analysis.
    """
    dense = to_dense_array(tensor)
    non_finite = not np.all(np.isfinite(dense))
    magnitude = float(np.nanmax(np.abs(dense))) if dense.size else 0.0
    if non_finite or magnitude > clamp:
        repaired = np.nan_to_num(dense, nan=0.0, posinf=clamp, neginf=-clamp)
        repaired = np.clip(repaired, -clamp, clamp)
        tensor.data = repaired
        return tensor, True
    return tensor, False


def stage_1_import_and_env(seed: int) -> StageResult:
    """Stage 1: Environment validation and seed setup."""
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        seed_all(seed)
        metrics["numpy_version"] = np.__version__
        metrics["torch_version"] = torch.__version__
        metrics["seed"] = seed
        success = True
        details = "Environment ready for recursion stress."
    except Exception as exc:
        errors.append(str(exc))
        success = False
        details = "Environment validation failed."

    return StageResult(
        stage_num=1,
        name="Import/Environment",
        success=success,
        duration_s=time.time() - start,
        details=details,
        metrics=metrics,
        warnings=warnings,
        errors=errors,
    )


def stage_2_recursive_tensor_destruction(
    depth: int,
    dim: int,
    rank: int,
) -> Tuple[StageResult, Dict[str, Any]]:
    """Stage 2: RecursiveTensor contradiction spiral with depth pressure."""
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    figure_paths: List[str] = []

    try:
        tensor = RecursiveTensor(
            dimensions=dim,
            rank=rank,
            distribution="uniform",
            sparsity=0.0,
        )
        mirror = RecursiveTensor(
            dimensions=dim,
            rank=rank,
            distribution="uniform",
            sparsity=0.0,
        )
        mirror.data = -np.transpose(tensor.data.copy())

        prev_dense = to_dense_array(tensor)
        repair_count = 0

        for step in range(depth):
            axes = ((0,), (0,)) if step % 2 == 0 else ((1,), (0,))
            tensor = tensor.contract(mirror if step % 2 == 0 else tensor, axes=axes)

            scale = 1.08 if step % 2 == 0 else 0.82
            transform = build_transform_matrix(dim, scale=scale, jitter=0.02, seed=step + 41)
            tensor = tensor.transform(transform, axes=(0, 1))

            tensor = tensor.fractal_iteration(contradiction_c_function, max_iter=6)

            tensor = tensor.apply_function(
                lambda x: np.where(np.abs(x) < 0.05, -x, x)  # contradiction inversion band
            )

            if step % 5 == 4:
                tensor = tensor.normalize(norm_type=2)  # stability correction after depth bursts

            tensor, repaired = stabilize_tensor_values(tensor, clamp=1e6)
            if repaired:
                repair_count += 1

            dense = to_dense_array(tensor)
            delta = float(np.linalg.norm(dense - prev_dense))
            norm = float(np.linalg.norm(dense))
            max_abs = float(np.max(np.abs(dense)))
            nonzero_ratio = float(np.mean(np.abs(dense) > 1e-6))

            mask = (np.abs(dense) > 1e-9) | (np.abs(prev_dense) > 1e-9)
            if np.any(mask):
                sign_flip = float(np.mean(np.sign(dense[mask]) != np.sign(prev_dense[mask])))
            else:
                sign_flip = 0.0

            entropy = float(tensor.compute_entropy())
            abs_values = np.abs(dense)
            abs_sum = float(np.sum(abs_values))
            if abs_sum > 0 and np.isfinite(abs_sum):
                abs_prob = abs_values / abs_sum
                abs_entropy = float(-np.sum(abs_prob * np.log(abs_prob + 1e-10)))  # Shannon entropy
            else:
                abs_entropy = 0.0

            history.append(
                {
                    "step": step,
                    "delta": delta,
                    "norm": norm,
                    "max_abs": max_abs,
                    "nonzero_ratio": nonzero_ratio,
                    "sign_flip_ratio": sign_flip,
                    "entropy": entropy,
                    "abs_entropy": abs_entropy,
                }
            )
            prev_dense = dense

        eigenvalues: List[float] = []
        try:
            eigenvalues, _ = tensor.compute_eigenstates(
                axes=((0,), (1,)),
                k=min(6, dim),
                convergence_threshold=1e-6,  # Eigenrecursion Theorem fixed-point convergence
            )
            eigenvalues = [float(np.abs(v)) for v in eigenvalues]
        except Exception as exc:
            warnings.append(f"Eigenstate computation failed: {exc}")

        try:
            import matplotlib.pyplot as plt

            steps = [row["step"] for row in history]
            norms = [row["norm"] for row in history]
            deltas = [row["delta"] for row in history]
            flips = [row["sign_flip_ratio"] for row in history]
            entropies = [row["entropy"] for row in history]
            abs_entropies = [row["abs_entropy"] for row in history]

            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            axes[0].plot(steps, norms, label="norm", linewidth=2)
            axes[0].plot(steps, deltas, label="delta", linewidth=2)
            axes[0].set_title("Recursive Tensor Norm/Delta Depth Curve")
            axes[0].set_xlabel("depth step")
            axes[0].set_ylabel("magnitude")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(steps, flips, label="sign_flip_ratio", linewidth=2, color="tab:red")
            axes[1].set_title("Contradiction Sign Flips")
            axes[1].set_xlabel("depth step")
            axes[1].set_ylabel("ratio")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            axes[2].plot(steps, entropies, label="entropy", linewidth=2)
            axes[2].plot(steps, abs_entropies, label="abs_entropy", linewidth=2)
            axes[2].set_title("Entropy Across Depth")
            axes[2].set_xlabel("depth step")
            axes[2].set_ylabel("entropy")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            fig.tight_layout()
            depth_fig = FIGURE_DIR / f"recursive_destruction_depth_{RUN_TS}.png"
            fig.savefig(depth_fig, dpi=200)
            plt.close(fig)
            figure_paths.append(str(depth_fig))

            eig_fig = tensor.visualize_eigenspectrum(k=min(10, dim), title="Eigen Spectrum", show=False)
            if eig_fig is not None:
                eig_path = FIGURE_DIR / f"recursive_destruction_eigenspectrum_{RUN_TS}.png"
                eig_fig.savefig(eig_path, dpi=200)
                plt.close(eig_fig)
                figure_paths.append(str(eig_path))
        except Exception as exc:
            warnings.append(f"Visualization skipped: {exc}")

        metrics = {
            "depth": depth,
            "final_norm": history[-1]["norm"] if history else None,
            "max_sign_flip_ratio": max((row["sign_flip_ratio"] for row in history), default=0.0),
            "mean_entropy": float(np.mean([row["entropy"] for row in history])) if history else 0.0,
            "abs_entropy_peak": float(max((row["abs_entropy"] for row in history), default=0.0)),
            "finite_repairs": repair_count,
            "eigenvalues": eigenvalues,
            "figures": figure_paths,
        }

        success = True
        details = "RecursiveTensor depth stress complete."
    except Exception as exc:
        errors.append(str(exc))
        details = "RecursiveTensor stress failed."
        success = False

    return (
        StageResult(
            stage_num=2,
            name="RecursiveTensor Destruction",
            success=success,
            duration_s=time.time() - start,
            details=details,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ),
        {"history": history, "figures": figure_paths},
    )


def stage_3_stability_matrix_patterns() -> Tuple[StageResult, Dict[str, Any]]:
    """Stage 3: Contradiction pattern detection with RecursionDetector."""
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    figure_paths: List[str] = []

    try:
        detector = RecursionDetector(max_history_size=128, similarity_threshold=0.8)

        for i in range(4):
            detector.record_operation(
                {
                    "id": f"claim_true_{i}",
                    "type": "claim",
                    "content": "stability is asserted",
                    "value": 1.0,
                }
            )
            detector.record_operation(
                {
                    "id": f"claim_false_{i}",
                    "type": "claim",
                    "content": "stability is denied",
                    "value": -1.0,
                    "contradicts": f"claim_true_{i}",
                }
            )

        for depth in range(1, 4):
            detector.record_operation(
                {
                    "id": f"self_ref_{depth}",
                    "type": "introspection",
                    "content": "self-reference escalates",
                    "references": ["self"],
                    "self_reference_depth": depth,
                }
            )

        for idx in range(4):
            detector.record_operation(
                {
                    "id": f"refine_{idx}",
                    "type": "clarification",
                    "content": "clarification iteration",
                    "refinement_markers": [idx],
                }
            )

        for idx, value in enumerate([0.5, 1.1, 2.4, 4.9]):
            detector.record_operation(
                {
                    "id": f"complex_{idx}",
                    "type": "complexity",
                    "content": "complexity escalation",
                    "complexity_metrics": {"overall": value},
                }
            )

        new_patterns = detector.detect_patterns()
        all_patterns = detector.get_all_patterns(include_resolved=True)
        stats = detector.get_stats()

        pattern_counts = {
            key.name: int(value) for key, value in stats["pattern_counts"].items()
        }

        try:
            import matplotlib.pyplot as plt

            labels = list(pattern_counts.keys())
            values = [pattern_counts[label] for label in labels]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(labels, values, color="tab:blue")
            ax.set_title("Detected Recursion Patterns")
            ax.set_xlabel("pattern")
            ax.set_ylabel("count")
            ax.tick_params(axis="x", rotation=40)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            pattern_fig = FIGURE_DIR / f"recursive_destruction_patterns_{RUN_TS}.png"
            fig.savefig(pattern_fig, dpi=200)
            plt.close(fig)
            figure_paths.append(str(pattern_fig))
        except Exception as exc:
            warnings.append(f"Pattern visualization skipped: {exc}")

        metrics = {
            "patterns_detected": len(new_patterns),
            "total_patterns": len(all_patterns),
            "pattern_counts": pattern_counts,
            "figures": figure_paths,
        }
        success = True
        details = "Contradiction patterns detected."
    except Exception as exc:
        errors.append(str(exc))
        details = "StabilityMatrix pattern detection failed."
        success = False

    return (
        StageResult(
            stage_num=3,
            name="StabilityMatrix Patterns",
            success=success,
            duration_s=time.time() - start,
            details=details,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ),
        {
            "patterns": [p.to_dict() for p in all_patterns] if "all_patterns" in locals() else [],
            "figures": figure_paths,
        },
    )


def stage_4_eigenrecursion_stabilizer(
    state_dim: int,
    max_iterations: int,
) -> Tuple[StageResult, Dict[str, Any]]:
    """Stage 4: Push EigenrecursionStabilizer from chaos to convergence."""
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    figure_paths: List[str] = []

    try:
        stabilizer = EigenrecursionStabilizer(
            state_dimension=state_dim,
            contraction_factor=0.7,
            convergence_threshold=1e-6,
            max_iterations=max_iterations,
            learning_rate=0.05,
        )
        initial_state = torch.randn(state_dim) * 0.4

        with torch.no_grad():
            stabilizer.weight.data *= 3.2

        chaos_result = stabilizer.compute_eigenstate(
            initial_state
        )  # Eigenrecursion Theorem: fixed-point convergence probe
        chaos_history = stabilizer.convergence_history.copy()

        with torch.no_grad():
            stabilized_weights = stabilizer.weight.detach().clone() * 0.2
        stabilizer.update_operator(
            new_weights=stabilized_weights,
            learning_rate=1.0,
        )

        stable_result = stabilizer.compute_eigenstate(
            initial_state
        )  # Eigenrecursion Theorem: contractive convergence enforcement
        stable_history = stabilizer.convergence_history.copy()
        stability_check = stabilizer.check_stability(stable_result["fixed_point"])

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(chaos_history, label="chaos phase", linewidth=2)
            ax.plot(stable_history, label="stabilized phase", linewidth=2)
            ax.set_title("Eigenrecursion Convergence History")
            ax.set_xlabel("iteration")
            ax.set_ylabel("delta")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            eigen_fig = FIGURE_DIR / f"recursive_destruction_eigenrecursion_{RUN_TS}.png"
            fig.savefig(eigen_fig, dpi=200)
            plt.close(fig)
            figure_paths.append(str(eigen_fig))
        except Exception as exc:
            warnings.append(f"Eigenrecursion visualization skipped: {exc}")

        metrics = {
            "chaos_status": chaos_result["convergence_status"],
            "chaos_iterations": chaos_result["iterations"],
            "stable_status": stable_result["convergence_status"],
            "stable_iterations": stable_result["iterations"],
            "stable_final_distance": stable_result["final_distance"],
            "stability_score": stability_check["stability_score"],
            "spectral_radius": stabilizer.spectral_radius,
            "figures": figure_paths,
        }
        success = True
        details = "Eigenrecursion stabilization pressure test complete."
    except Exception as exc:
        errors.append(str(exc))
        details = "Eigenrecursion stabilization failed."
        success = False

    return (
        StageResult(
            stage_num=4,
            name="Eigenrecursion Stabilizer",
            success=success,
            duration_s=time.time() - start,
            details=details,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ),
        {"figures": figure_paths},
    )


def stage_5_triaxial_backbone_stress(steps: int) -> Tuple[StageResult, Dict[str, Any]]:
    """Stage 5: Triaxial backbone contradictions over alternating inputs."""
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    figure_paths: List[str] = []
    timeline: List[Dict[str, Any]] = []
    inputs: List[np.ndarray] = []

    config = TriaxialConfig(
        recursive_dim=16,
        recursive_rank=2,
        recursive_sparsity=0.0,
        ethical_dim=5,
        field_shape=(8, 8),
        metacog_state_dim=96,
        metacog_layers=2,
        consciousness_threshold=0.8,
        epsilon=1e-6,
        max_iterations=500,
        theta_moral=0.9,
        theta_epistemic=0.1,
        identity_threshold=0.75,
        bayesian_enabled=False,
        parallel_computation=False,
    )

    try:
        field = TriaxialField(config)
        base_dim = config.recursive_dim + config.ethical_dim + config.metacog_state_dim // 8
        base_signal = np.tanh(np.linspace(-2, 2, base_dim)).astype(np.float32)
        phase_names = ["INHALE", "HOLD_IN", "EXHALE", "HOLD_OUT"]
        phases = [getattr(BreathPhase, name) for name in phase_names if hasattr(BreathPhase, name)]
        if not phases:
            phases = list(BreathPhase)

        prev_vector: Optional[np.ndarray] = None
        contradiction_scores: List[float] = []

        for step in range(steps):
            polarity = 1.0 if step % 2 == 0 else -1.0
            noise = np.random.normal(0.0, 0.05, size=base_signal.shape).astype(np.float32)
            input_data = base_signal * polarity + noise
            breath_phase = phases[step % len(phases)]

            state = field.compute(input_data, breath_phase=breath_phase)
            vector = state.integrated_vector if state.integrated_vector is not None else np.zeros(3)

            if prev_vector is not None:
                denom = (np.linalg.norm(prev_vector) * np.linalg.norm(vector)) + 1e-8
                cosine = float(np.dot(prev_vector, vector) / denom)
                contradiction = float(1.0 - cosine)
            else:
                contradiction = 0.0

            contradiction_scores.append(contradiction)
            timeline.append(
                {
                    "step": step,
                    "status": state.convergence_status,
                    "magnitude": state.stability_metrics.get("magnitude"),
                    "coherence": state.stability_metrics.get("coherence"),
                    "ere": state.stability_metrics.get("ere"),
                    "rbu": state.stability_metrics.get("rbu"),
                    "es": state.stability_metrics.get("es"),
                    "contradiction": contradiction,
                }
            )
            inputs.append(input_data.copy())

            prev_vector = vector

        field.shutdown()

        try:
            import matplotlib.pyplot as plt

            steps_axis = [row["step"] for row in timeline]
            magnitudes = [row["magnitude"] for row in timeline]
            coherence = [row["coherence"] for row in timeline]

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(steps_axis, magnitudes, label="magnitude", linewidth=2)
            axes[0].plot(steps_axis, coherence, label="coherence", linewidth=2)
            axes[0].set_title("Triaxial Stability Over Contradictions")
            axes[0].set_xlabel("step")
            axes[0].set_ylabel("value")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(steps_axis, contradiction_scores, label="contradiction", linewidth=2, color="tab:red")
            axes[1].set_title("Contradiction Index (1 - cosine)")
            axes[1].set_xlabel("step")
            axes[1].set_ylabel("score")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            tri_fig = FIGURE_DIR / f"recursive_destruction_triaxial_{RUN_TS}.png"
            fig.savefig(tri_fig, dpi=200)
            plt.close(fig)
            figure_paths.append(str(tri_fig))
        except Exception as exc:
            warnings.append(f"Triaxial visualization skipped: {exc}")

        stable_count = sum(1 for row in timeline if row["status"] == "STABLE")
        metrics = {
            "steps": steps,
            "stable_ratio": stable_count / max(1, steps),
            "max_contradiction": float(max(contradiction_scores)) if contradiction_scores else 0.0,
            "mean_coherence": float(np.mean([row["coherence"] for row in timeline])) if timeline else 0.0,
            "figures": figure_paths,
        }
        success = True
        details = "Triaxial backbone stress run complete."
    except Exception as exc:
        errors.append(str(exc))
        details = "Triaxial backbone stress failed."
        success = False

    return (
        StageResult(
            stage_num=5,
            name="Triaxial Backbone Stress",
            success=success,
            duration_s=time.time() - start,
            details=details,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ),
        {"timeline": timeline, "inputs": inputs, "figures": figure_paths},
    )


def stage_6_llm_proxy_baseline(
    inputs: List[np.ndarray],
    state_dim: int,
    seed: int,
    rcf_timeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[StageResult, Dict[str, Any]]:
    """
    Stage 6: LLM-style autoregressive proxy baseline.

    Uses the same contradiction-driving inputs to measure stability behavior
    without eigenrecursion-specific stabilizers.
    """
    start = time.time()
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
    figure_paths: List[str] = []
    timeline: List[Dict[str, Any]] = []

    if not inputs:
        return (
            StageResult(
                stage_num=6,
                name="LLM Proxy Baseline",
                success=False,
                duration_s=time.time() - start,
                details="No inputs provided for baseline.",
                metrics={},
                warnings=[],
                errors=["inputs list was empty"],
            ),
            {"timeline": timeline, "figures": figure_paths},
        )

    try:
        input_dim = inputs[0].shape[0]
        proxy = LLMProxy(
            state_dim=state_dim,
            input_dim=input_dim,
            seed=seed,
            decay=0.88,
            noise_scale=0.03,
            contradiction_bias=0.7,
        )

        prev_state: Optional[np.ndarray] = None
        prev_polarity: Optional[float] = None
        non_finite_repairs = 0
        stable_threshold = 0.05

        for step, input_vec in enumerate(inputs):
            polarity = float(np.sign(np.mean(input_vec)))
            contradiction = prev_polarity is not None and polarity != prev_polarity

            state = proxy.step(input_vec, contradiction=contradiction)
            if not np.all(np.isfinite(state)):
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
                proxy.state = state.astype(np.float32)
                non_finite_repairs += 1

            if prev_state is not None:
                delta = float(np.linalg.norm(state - prev_state))
                denom = (np.linalg.norm(prev_state) * np.linalg.norm(state)) + 1e-8
                cosine = float(np.dot(prev_state, state) / denom)
                contradiction_score = float(1.0 - cosine)
                sign_flip_ratio = float(
                    np.mean(np.sign(state) != np.sign(prev_state))
                )
            else:
                delta = 0.0
                contradiction_score = 0.0
                sign_flip_ratio = 0.0

            saturation = float(np.mean(np.abs(state) > 0.97))
            timeline.append(
                {
                    "step": step,
                    "delta": delta,
                    "norm": float(np.linalg.norm(state)),
                    "contradiction": contradiction_score,
                    "sign_flip_ratio": sign_flip_ratio,
                    "saturation": saturation,
                    "stable": delta < stable_threshold,
                }
            )

            prev_state = state.copy()
            prev_polarity = polarity

        try:
            import matplotlib.pyplot as plt

            steps_axis = [row["step"] for row in timeline]
            deltas = [row["delta"] for row in timeline]
            contradictions = [row["contradiction"] for row in timeline]
            saturations = [row["saturation"] for row in timeline]

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(steps_axis, deltas, label="delta", linewidth=2)
            axes[0].plot(steps_axis, contradictions, label="contradiction", linewidth=2)
            axes[0].set_title("LLM Proxy Recursion Drift")
            axes[0].set_xlabel("step")
            axes[0].set_ylabel("magnitude")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(steps_axis, saturations, label="saturation", linewidth=2, color="tab:orange")
            axes[1].set_title("LLM Proxy Saturation")
            axes[1].set_xlabel("step")
            axes[1].set_ylabel("ratio")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            proxy_fig = FIGURE_DIR / f"recursive_destruction_llm_proxy_{RUN_TS}.png"
            fig.savefig(proxy_fig, dpi=200)
            plt.close(fig)
            figure_paths.append(str(proxy_fig))

            if rcf_timeline:
                rcf_steps = [row["step"] for row in rcf_timeline]
                rcf_contradictions = [row["contradiction"] for row in rcf_timeline]
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(
                    rcf_steps,
                    rcf_contradictions,
                    label="triaxial contradiction",
                    linewidth=2,
                )
                ax2.plot(
                    steps_axis,
                    contradictions,
                    label="llm proxy contradiction",
                    linewidth=2,
                )
                ax2.set_title("Contradiction Index Comparison")
                ax2.set_xlabel("step")
                ax2.set_ylabel("1 - cosine")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                fig2.tight_layout()
                compare_fig = FIGURE_DIR / f"recursive_destruction_compare_{RUN_TS}.png"
                fig2.savefig(compare_fig, dpi=200)
                plt.close(fig2)
                figure_paths.append(str(compare_fig))
        except Exception as exc:
            warnings.append(f"LLM proxy visualization skipped: {exc}")

        stable_ratio = float(np.mean([1.0 if row["stable"] else 0.0 for row in timeline]))
        metrics = {
            "steps": len(timeline),
            "stable_ratio": stable_ratio,
            "max_contradiction": float(max((row["contradiction"] for row in timeline), default=0.0)),
            "mean_delta": float(np.mean([row["delta"] for row in timeline])) if timeline else 0.0,
            "mean_saturation": float(
                np.mean([row["saturation"] for row in timeline]) if timeline else 0.0
            ),
            "non_finite_repairs": non_finite_repairs,
            "stable_threshold": stable_threshold,
            "figures": figure_paths,
        }
        success = True
        details = "LLM proxy baseline run complete."
    except Exception as exc:
        errors.append(str(exc))
        details = "LLM proxy baseline failed."
        success = False

    return (
        StageResult(
            stage_num=6,
            name="LLM Proxy Baseline",
            success=success,
            duration_s=time.time() - start,
            details=details,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ),
        {"timeline": timeline, "figures": figure_paths},
    )


def write_report(
    stage_results: List[StageResult],
    artifacts: Dict[str, Any],
    seed: int,
) -> None:
    """Write a markdown report for the benchmark run."""
    lines: List[str] = []
    lines.append("# Recursive Destruction Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: {RUN_TS}")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Log: {LOG_FILE}")
    lines.append(f"- Manifest: {JSON_FILE}")
    lines.append(f"- Figures: {', '.join(artifacts.get('figures', []))}")
    lines.append("")

    for stage in stage_results:
        lines.append(f"## Stage {stage.stage_num}: {stage.name}")
        lines.append(f"- Status: {'PASS' if stage.success else 'FAIL'}")
        lines.append(f"- Duration: {stage.duration_s:.3f}s")
        lines.append(f"- Details: {stage.details}")
        if stage.metrics:
            lines.append("- Metrics:")
            for key, value in stage.metrics.items():
                lines.append(f"  - {key}: {value}")
        if stage.warnings:
            lines.append("- Warnings:")
            for warning in stage.warnings:
                lines.append(f"  - {warning}")
        if stage.errors:
            lines.append("- Errors:")
            for error in stage.errors:
                lines.append(f"  - {error}")
        lines.append("")

    rcf_timeline = artifacts.get("rcf_timeline", [])
    llm_proxy_timeline = artifacts.get("llm_proxy_timeline", [])
    if rcf_timeline and llm_proxy_timeline:
        rcf_max_contradiction = float(
            max((row["contradiction"] for row in rcf_timeline), default=0.0)
        )
        rcf_mean_coherence = float(
            np.mean([row["coherence"] for row in rcf_timeline]) if rcf_timeline else 0.0
        )
        rcf_stable = sum(1 for row in rcf_timeline if row["status"] == "STABLE")
        rcf_stable_ratio = rcf_stable / max(1, len(rcf_timeline))

        llm_max_contradiction = float(
            max((row["contradiction"] for row in llm_proxy_timeline), default=0.0)
        )
        llm_mean_delta = float(
            np.mean([row["delta"] for row in llm_proxy_timeline]) if llm_proxy_timeline else 0.0
        )
        llm_stable = sum(1 for row in llm_proxy_timeline if row["stable"])
        llm_stable_ratio = llm_stable / max(1, len(llm_proxy_timeline))

        lines.append("## Comparison Summary")
        lines.append("- Note: LLM proxy is a local baseline, not a real model.")
        lines.append(f"- Triaxial max contradiction: {rcf_max_contradiction:.4f}")
        lines.append(f"- LLM proxy max contradiction: {llm_max_contradiction:.4f}")
        lines.append(f"- Triaxial mean coherence: {rcf_mean_coherence:.4f}")
        lines.append(f"- LLM proxy mean delta: {llm_mean_delta:.4f}")
        lines.append(f"- Triaxial stable ratio: {rcf_stable_ratio:.4f}")
        lines.append(f"- LLM proxy stable ratio: {llm_stable_ratio:.4f}")
        lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark() -> int:
    """Run the full recursive destruction benchmark."""
    seed = 44
    stage_results: List[StageResult] = []
    artifacts: Dict[str, Any] = {"figures": []}

    LOGGER.info("=" * 72)
    LOGGER.info("Recursive Destruction Benchmark - Start")
    LOGGER.info("=" * 72)

    stage_results.append(stage_1_import_and_env(seed))

    stage_2, stage_2_artifacts = stage_2_recursive_tensor_destruction(depth=18, dim=16, rank=2)
    stage_results.append(stage_2)
    artifacts["figures"].extend(stage_2_artifacts.get("figures", []))

    stage_3, stage_3_artifacts = stage_3_stability_matrix_patterns()
    stage_results.append(stage_3)
    artifacts["figures"].extend(stage_3_artifacts.get("figures", []))

    stage_4, stage_4_artifacts = stage_4_eigenrecursion_stabilizer(state_dim=64, max_iterations=80)
    stage_results.append(stage_4)
    artifacts["figures"].extend(stage_4_artifacts.get("figures", []))

    stage_5, stage_5_artifacts = stage_5_triaxial_backbone_stress(steps=16)
    stage_results.append(stage_5)
    artifacts["figures"].extend(stage_5_artifacts.get("figures", []))

    stage_6, stage_6_artifacts = stage_6_llm_proxy_baseline(
        inputs=stage_5_artifacts.get("inputs", []),
        state_dim=64,
        seed=seed + 1,
        rcf_timeline=stage_5_artifacts.get("timeline"),
    )
    stage_results.append(stage_6)
    artifacts["figures"].extend(stage_6_artifacts.get("figures", []))

    manifest = {
        "timestamp": RUN_TS,
        "seed": seed,
        "log_file": str(LOG_FILE),
        "report_file": str(REPORT_FILE),
        "figures": artifacts.get("figures", []),
        "stages": [stage.__dict__ for stage in stage_results],
        "artifacts": {
            "recursive_tensor_history": stage_2_artifacts.get("history", []),
            "stability_patterns": stage_3_artifacts.get("patterns", []),
            "triaxial_timeline": stage_5_artifacts.get("timeline", []),
            "llm_proxy_timeline": stage_6_artifacts.get("timeline", []),
        },
    }

    artifacts["rcf_timeline"] = stage_5_artifacts.get("timeline", [])
    artifacts["llm_proxy_timeline"] = stage_6_artifacts.get("timeline", [])

    JSON_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(stage_results, artifacts, seed=seed)

    failures = [stage for stage in stage_results if not stage.success]
    if failures:
        LOGGER.error("Benchmark completed with failures.")
        return 1

    LOGGER.info("Benchmark completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_benchmark())
    except Exception:  # pragma: no cover - safety net for terminal-only runs
        LOGGER.error("Fatal benchmark error:\n%s", traceback.format_exc())
        sys.exit(1)
