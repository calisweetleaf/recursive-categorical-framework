"""
Base stability matrix primitives used by the public integration tests.

This module intentionally contains only a watered-down subset of the full
Rosemary/RCF stability node so the open-source tests can run without exposing
proprietary logic.  The production-grade implementation includes additional
spectral diagnostics, paradox buffers, and recursion-state governance layers.

For access to the full stability matrix (NDA + license required) contact:
    treyrowell1826@gmail.com
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

FULL_MATRIX_CONTACT = "treyrowell1826@gmail.com"
PUBLIC_NOTICE = (
    "BaseStabilityMatrix is intentionally limited. "
    "Contact treyrowell1826@gmail.com for the full Eigenrecursion stability node "
    "(NDA/licensing required)."
)

logger = logging.getLogger("BaseStabilityMatrix")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class StabilitySnapshot:
    """Minimal record of a single convergence measurement."""

    iteration: int
    delta: float
    spectral_radius: float
    contraction_factor: float
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class EigenrecursionStabilizer:
    """
    Lightweight stabilizer that approximates the behavior of the proprietary
    EigenrecursionStabilizer used in Rosemary deployments.

    It provides:
    - contraction-factor tracking
    - simple delta/spectral metrics
    - adaptive adjustments when instability is detected
    """

    _notice_emitted = False

    def __init__(
        self,
        state_dimension: int = 256,
        contraction_factor: float = 0.8,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 500,
        learning_rate: float = 0.01,
    ) -> None:
        self.state_dimension = state_dimension
        self.contraction_factor = float(np.clip(contraction_factor, 0.1, 0.99))
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

        self.snapshots: List[StabilitySnapshot] = []
        self.last_delta: float = 0.0
        self.last_spectral_radius: float = self.contraction_factor
        self.converged: bool = False
        self.iterations_to_converge: int = 0

        self._emit_notice()
        logger.debug(
            "Base EigenrecursionStabilizer initialized "
            "(dim=%s, contraction=%.3f, threshold=%s)",
            self.state_dimension,
            self.contraction_factor,
            self.convergence_threshold,
        )

    def _emit_notice(self) -> None:
        if not EigenrecursionStabilizer._notice_emitted:
            logger.warning(PUBLIC_NOTICE)
            print(f"[WARN] {PUBLIC_NOTICE}")
            EigenrecursionStabilizer._notice_emitted = True

    def reset(self) -> None:
        """Clear history between major runs."""
        self.snapshots.clear()
        self.last_delta = 0.0
        self.last_spectral_radius = self.contraction_factor
        self.converged = False
        self.iterations_to_converge = 0

    def evaluate_state(
        self,
        current_state: torch.Tensor,
        previous_state: Optional[torch.Tensor],
        iteration: int,
    ) -> Dict[str, float]:
        """
        Compute simple convergence metrics between successive states.

        Returns a dict with the delta norm, spectral estimate, and convergence flag.
        """
        current = current_state.detach().cpu().numpy().astype(float).ravel()
        if previous_state is None:
            prev = np.zeros_like(current)
        else:
            prev = previous_state.detach().cpu().numpy().astype(float).ravel()

        delta = float(np.linalg.norm(current - prev))
        spectral_radius = float(
            np.clip(self.contraction_factor + delta * 1e-3, 0.0, 1.5)
        )

        self.last_delta = delta
        self.last_spectral_radius = spectral_radius
        self.converged = delta < self.convergence_threshold
        self.iterations_to_converge = iteration if self.converged else 0

        snapshot = StabilitySnapshot(
            iteration=iteration,
            delta=delta,
            spectral_radius=spectral_radius,
            contraction_factor=self.contraction_factor,
        )
        self.snapshots.append(snapshot)

        return {
            "delta": delta,
            "spectral_radius": spectral_radius,
            "converged": self.converged,
        }

    def adaptive_adjustment(
        self,
        instability_detected: bool,
        delta: Optional[float] = None,
        spectral_radius: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Adjust contraction and learning heuristics when instability occurs.

        The real system tunes dozens of parameters; this version only nudges the
        contraction factor and reports the updated metrics.
        """
        effective_delta = delta if delta is not None else self.last_delta
        effective_radius = (
            spectral_radius if spectral_radius is not None else self.last_spectral_radius
        )

        if instability_detected:
            adjustment = -self.learning_rate * max(1.0, effective_radius)
        else:
            adjustment = self.learning_rate * 0.5

        self.contraction_factor = float(
            np.clip(self.contraction_factor + adjustment, 0.1, 0.99)
        )

        summary = {
            "contraction_factor": self.contraction_factor,
            "delta": effective_delta,
            "spectral_radius": effective_radius,
            "instability_detected": bool(instability_detected),
        }

        logger.debug(
            "Adaptive adjustment executed :: %s",
            summary,
        )

        return summary

    def summary(self) -> Dict[str, Any]:
        """Return a compact snapshot of recent stability metrics."""
        return {
            "state_dimension": self.state_dimension,
            "contraction_factor": self.contraction_factor,
            "converged": self.converged,
            "iterations_to_converge": self.iterations_to_converge,
            "last_delta": self.last_delta,
            "last_spectral_radius": self.last_spectral_radius,
            "history": [snap.to_dict() for snap in self.snapshots[-5:]],
            "contact": FULL_MATRIX_CONTACT,
        }


__all__ = ["EigenrecursionStabilizer", "StabilitySnapshot", "FULL_MATRIX_CONTACT"]
