#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triaxial Backbone: Unified Fiber Bundle Computational Architecture

This module implements the triaxial fiber bundle architecture where all three
cognitive axes (Recursive, Ethical, Metacognitive) compute simultaneously on
the same input and integrate via fiber bundle projection.

Architecture per RCF/URST/RSIA framework:
- Recursive Axis (ERE): Identity formation through eigenrecursion
- Ethical Axis (RBU): Value alignment and constraint resolution
- Metacognitive Axis (ES): Self-reflection and stability analysis

Mathematical Foundation:
- Fixed-point convergence: R(s*) = s*
- Triaxial integration: Φ = (ERE, RBU, ES) ∈ M₃
- Fiber bundle projection: π: E → B

Author: RCF Development
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
import threading
import time
import json
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TriaxialBackbone")

# Import tensor implementations
from rcf_integration.recursive_tensor import RecursiveTensor
from rcf_integration.ethical_tensor import (
    SymbolicQuantumState,
    EthicalTensorFactory,
    QuantumBreathAdapter,
    BreathPhase
)
from rcf_integration.metacognitive_tensor import MetacognitiveTensor

# Import orchestration components (optional - may not be present in all deployments)
try:
    from bayesian_config_orchestrator import (
        BayesianConfigurationOrchestrator,
        RecursiveBayesianUpdater,
        ParameterBelief,
        DistributionType
    )
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianConfigurationOrchestrator = None
    logger.warning("bayesian_config_orchestrator not available - Bayesian features disabled")

from zynx_zebra_core import EigenrecursionStabilizer, ZEBAEigenrecursionStabilizer

# Import stability components
try:
    from rcf_integration.stability_matrix import EigenrecursionStabilizer as StabilityStabilizer
except ImportError:
    StabilityStabilizer = None


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class TriaxialConfig:
    """Configuration for the triaxial backbone.
    
    Defines dimensions, thresholds, and parameters for all three axes
    and the integration layer.
    """
    # Recursive axis (ERE) - using small dimensions for fast computation
    # Note: dim=32 rank=4 = 1M elements, contracting = 30+ seconds!
    # Using dim=16 rank=2 = 256 elements, contracting = milliseconds
    recursive_dim: int = 16
    recursive_rank: int = 2  # Matrix (rank-2) for fast contraction
    recursive_sparsity: float = 0.0  # Dense for predictable timing
    
    # Ethical axis (RBU)
    ethical_dim: int = 5
    field_shape: Tuple[int, int] = (16, 16)
    
    # Metacognitive axis (ES)
    metacog_state_dim: int = 256
    metacog_layers: int = 3
    consciousness_threshold: float = 0.8
    
    # Eigenrecursion stabilization
    epsilon: float = 1e-6
    max_iterations: int = 1000
    theta_moral: float = 0.92  # Ethical convergence threshold from ERE
    theta_epistemic: float = 0.1  # Epistemic convergence threshold from RBU
    identity_threshold: float = 0.78
    
    # Bayesian orchestration
    bayesian_enabled: bool = True
    security_enabled: bool = False
    max_memory_gb: float = 6.0
    
    # Parallel processing
    parallel_computation: bool = True
    max_workers: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.recursive_dim < 1:
            raise ValueError("recursive_dim must be positive")
        if self.ethical_dim < 1:
            raise ValueError("ethical_dim must be positive")
        if self.metacog_state_dim < 1:
            raise ValueError("metacog_state_dim must be positive")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= self.theta_moral <= 1:
            raise ValueError("theta_moral must be between 0 and 1")


# ================================================================
# TRIAXIAL STATE OUTPUT
# ================================================================

@dataclass
class TriaxialState:
    """Output state from the triaxial backbone computation.
    
    Contains the outputs from all three axes, the integrated fiber bundle
    projection, and stability/convergence metrics.
    """
    # Individual axis outputs
    recursive_output: Dict[str, Any] = field(default_factory=dict)
    ethical_output: Dict[str, Any] = field(default_factory=dict)
    metacog_output: Dict[str, Any] = field(default_factory=dict)
    
    # Integrated manifold state
    integrated_vector: Optional[np.ndarray] = None
    fiber_bundle_projection: Optional[np.ndarray] = None
    
    # Stability metrics
    stability_metrics: Dict[str, Any] = field(default_factory=dict)
    convergence_status: str = "UNKNOWN"
    convergence_distance: float = float('inf')
    
    # Identity and tracking
    state_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    computation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        return {
            "state_id": self.state_id,
            "timestamp": self.timestamp,
            "computation_time_ms": self.computation_time_ms,
            "convergence_status": self.convergence_status,
            "convergence_distance": float(self.convergence_distance) if np.isfinite(self.convergence_distance) else None,
            "recursive": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.recursive_output.items()
                if not isinstance(v, (torch.Tensor, np.ndarray)) or (isinstance(v, np.ndarray) and v.size < 1000)
            },
            "ethical": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in self.ethical_output.items()
                if isinstance(v, (int, float, str, bool, np.floating))
            },
            "metacog": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in self.metacog_output.items()
                if isinstance(v, (int, float, str, bool, np.floating, dict))
            },
            "stability_metrics": self.stability_metrics,
            "has_integrated_vector": self.integrated_vector is not None,
            "integrated_norm": float(np.linalg.norm(self.integrated_vector)) if self.integrated_vector is not None else None
        }


# ================================================================
# TRIAXIAL FIELD: PARALLEL COMPUTATION ENGINE
# ================================================================

class TriaxialField:
    """The core triaxial computational field.
    
    Implements the fiber bundle architecture where all three axes compute
    in parallel on the same input. This is not sequential forward passes
    but simultaneous triaxial computation forming a unified field.
    
    Mathematical basis:
    - Fiber bundle E = (M, π, B) where M is the total space, B is base manifold
    - Each fiber π^(-1)(b) contains the three axis computations at point b
    - Integration via fiber bundle projection π: E → B
    """
    
    def __init__(self, config: TriaxialConfig):
        """Initialize the triaxial field.
        
        Args:
            config: TriaxialConfig with all parameters
        """
        self.config = config
        self._lock = threading.Lock()
        
        # Initialize recursive tensor
        self.recursive_tensor = RecursiveTensor(
            dimensions=config.recursive_dim,
            rank=config.recursive_rank,
            distribution='uniform',
            sparsity=config.recursive_sparsity
        )
        logger.info(f"Initialized RecursiveTensor: dim={config.recursive_dim}, rank={config.recursive_rank}")
        
        # Initialize ethical tensor (SymbolicQuantumState)
        self.ethical_state = SymbolicQuantumState(
            field_shape=config.field_shape,
            ethical_dimensions=config.ethical_dim
        )
        self.breath_adapter = QuantumBreathAdapter(
            field_resolution=max(config.field_shape),
            ethical_dimensions=config.ethical_dim
        )
        logger.info(f"Initialized EthicalTensor: field_shape={config.field_shape}, dim={config.ethical_dim}")
        
        # Initialize metacognitive tensor
        self.metacog_tensor = MetacognitiveTensor(
            state_dim=config.metacog_state_dim,
            metacognitive_layers=config.metacog_layers,
            consciousness_threshold=config.consciousness_threshold
        )
        logger.info(f"Initialized MetacognitiveTensor: state_dim={config.metacog_state_dim}, layers={config.metacog_layers}")
        
        # Thread pool for parallel computation
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers) if config.parallel_computation else None
        
        # State tracking
        self._computation_count = 0
        self._last_state = None
    
    def compute(self, 
                input_data: Union[str, torch.Tensor, np.ndarray],
                breath_phase: Optional[BreathPhase] = None) -> TriaxialState:
        """
        Compute unified triaxial state from input.
        
        TRUE TRIAXIAL COMPUTATION:
        Unlike 3 separate computations that merge, this creates a SINGLE
        shared state vector where all 3 axes are projections operating
        simultaneously on the SAME underlying field.
        
        State partitioning (following zynx_zebra_core pattern):
        - state[:dim//3] = ERE (Recursive axis)
        - state[dim//3:2*dim//3] = RBU (Ethical axis)
        - state[2*dim//3:] = ES (Metacognitive axis)
        
        Args:
            input_data: Input to process (text, tensor, or array)
            breath_phase: Optional breath phase for ethical modulation
            
        Returns:
            TriaxialState with unified computation results
        """
        start_time = time.time()
        
        # Normalize input to numpy array for shared field
        input_array = self._normalize_input_to_array(input_data)
        
        # Set breath phase if provided
        if breath_phase is not None:
            self.breath_adapter.set_breath_phase(breath_phase)
        
        # Create the UNIFIED triaxial state vector
        # All 3 axes will operate on this SAME vector
        triaxial_dim = self.config.recursive_dim + self.config.ethical_dim + self.config.metacog_state_dim // 8
        unified_state = np.zeros(triaxial_dim, dtype=np.float32)
        
        # Axis boundaries in the unified state
        ere_end = self.config.recursive_dim
        rbu_end = ere_end + self.config.ethical_dim
        # es_end = triaxial_dim (rest is metacog)
        
        # Seed the unified state with input - distributed across all axes
        input_flat = input_array.flatten()[:triaxial_dim]
        if len(input_flat) < triaxial_dim:
            input_flat = np.pad(input_flat, (0, triaxial_dim - len(input_flat)))
        unified_state = input_flat.astype(np.float32)
        
        # ============================================================
        # SIMULTANEOUS TRIAXIAL COMPUTATION ON SHARED FIELD
        # ============================================================
        
        # Apply recursive transformation to ERE region
        ere_region = unified_state[:ere_end]
        recursive_out = self._apply_recursive_transform(ere_region)
        unified_state[:ere_end] = recursive_out['transformed']
        
        # Apply ethical transformation to RBU region
        rbu_region = unified_state[ere_end:rbu_end]
        ethical_out = self._apply_ethical_transform(rbu_region, input_data)
        unified_state[ere_end:rbu_end] = ethical_out['transformed']
        
        # Apply metacognitive transformation to ES region
        es_region = unified_state[rbu_end:]
        metacog_out = self._apply_metacog_transform(es_region, unified_state)
        unified_state[rbu_end:] = metacog_out['transformed']
        
        # ============================================================
        # INTEGRATION: Extract triaxial vector from unified field
        # ============================================================
        
        # Triaxial projection: Φ = (ERE, RBU, ES) - normalized
        ere_score = float(np.clip(np.mean(np.abs(unified_state[:ere_end])), 0, 1))
        rbu_score = float(np.clip(np.mean(np.abs(unified_state[ere_end:rbu_end])), 0, 1))
        es_score = float(np.clip(np.mean(np.abs(unified_state[rbu_end:])), 0, 1))
        
        triaxial_vector = np.array([ere_score, rbu_score, es_score])
        
        # Stability metrics
        magnitude = float(np.linalg.norm(triaxial_vector))
        mean_val = np.mean(triaxial_vector)
        coherence = 1.0 - np.std(triaxial_vector) / (mean_val + 1e-6) if mean_val > 0 else 0.0
        coherence = float(max(0.0, min(1.0, coherence)))
        
        # Status determination
        if magnitude > 0.8 and coherence > 0.5:
            status = "STABLE"
        elif magnitude > 0.5:
            status = "CONVERGING"
        elif magnitude > 0.1:
            status = "ACTIVE"
        else:
            status = "COLD_START"
        
        # Build output
        state = TriaxialState(
            recursive_output={**recursive_out, "stability_score": ere_score},
            ethical_output={**ethical_out, "ethical_vector_norm": rbu_score},
            metacog_output={**metacog_out, "consciousness_level": es_score},
            integrated_vector=triaxial_vector,
            fiber_bundle_projection=triaxial_vector / (magnitude + 1e-6),
            stability_metrics={
                "magnitude": magnitude,
                "coherence": coherence,
                "ere": ere_score,
                "rbu": rbu_score,
                "es": es_score,
                "unified_state_norm": float(np.linalg.norm(unified_state))
            },
            convergence_status=status,
            convergence_distance=float(1.0 - magnitude / np.sqrt(3))
        )
        
        # Record timing
        state.computation_time_ms = (time.time() - start_time) * 1000
        
        # Track state
        with self._lock:
            self._computation_count += 1
            self._last_state = state
        
        logger.info(
            f"Triaxial computation #{self._computation_count}: "
            f"time={state.computation_time_ms:.2f}ms, status={state.convergence_status}"
        )
        
        return state
    
    def _normalize_input_to_array(self, input_data: Union[str, torch.Tensor, np.ndarray]) -> np.ndarray:
        """Normalize input to numpy array for unified field."""
        if isinstance(input_data, str):
            char_codes = np.array([ord(c) % 256 for c in input_data], dtype=np.float32)
            return char_codes / 255.0
        elif isinstance(input_data, torch.Tensor):
            return input_data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    def _apply_recursive_transform(self, region: np.ndarray) -> Dict[str, Any]:
        """Apply recursive tensor transformation to ERE region of unified state."""
        # Create contraction on the region
        dim = len(region)
        matrix = region.reshape(-1, 1) @ region.reshape(1, -1)  # Outer product
        
        # Apply eigenrecursion-inspired transformation
        eigenvalues = np.linalg.eigvalsh(matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        
        # Normalize by spectral radius for stability
        if spectral_radius > 1e-6:
            transformed = region / (spectral_radius + 1.0)
        else:
            transformed = region
        
        return {
            "transformed": transformed,
            "spectral_radius": spectral_radius,
            "norm": float(np.linalg.norm(region)),
            "density": float(np.count_nonzero(region) / len(region))
        }
    
    def _apply_ethical_transform(self, region: np.ndarray, raw_input: Any) -> Dict[str, Any]:
        """Apply ethical tensor transformation to RBU region of unified state."""
        # Apply symbolic meaning if text input
        if isinstance(raw_input, str) and raw_input:
            self.ethical_state.apply_symbolic_meaning(
                symbol=raw_input,
                position=(0.5, 0.5),
                intensity=1.0
            )
        
        # Get ethical resonance and modulate
        resonance = self.ethical_state.symbol_resonance
        breath_mod = self.breath_adapter.current_phase.value if hasattr(self.breath_adapter, 'current_phase') else 0.5
        
        # Transform region using ethical resonance
        if resonance is not None and resonance.size > 0:
            resonance_sample = resonance.flatten()[:len(region)]
            if len(resonance_sample) < len(region):
                resonance_sample = np.pad(resonance_sample, (0, len(region) - len(resonance_sample)))
            transformed = region + 0.1 * resonance_sample * breath_mod
        else:
            transformed = region * breath_mod
        
        return {
            "transformed": np.clip(transformed, -1, 1),
            "breath_phase": self.breath_adapter.current_phase.name if hasattr(self.breath_adapter, 'current_phase') else "UNKNOWN",
            "resonance_norm": float(np.linalg.norm(resonance)) if resonance is not None else 0.0
        }
    
    def _apply_metacog_transform(self, region: np.ndarray, full_state: np.ndarray) -> Dict[str, Any]:
        """Apply metacognitive transformation to ES region of unified state."""
        # Metacognitive: reflect on the entire unified state
        full_norm = np.linalg.norm(full_state)
        region_norm = np.linalg.norm(region)
        
        # Self-reference: how does this region relate to the whole?
        coherence = 1.0 - abs(region_norm - full_norm / 3) / (full_norm + 1e-6) if full_norm > 0 else 0.0
        
        # Consciousness proxy: information integration
        consciousness = float(np.clip(coherence, 0, 1))
        
        # Transform: enhance coherence
        transformed = region * (1.0 + 0.1 * consciousness)
        
        return {
            "transformed": np.clip(transformed, -1, 1),
            "coherence": coherence,
            "consciousness_proxy": consciousness,
            "self_reference_ratio": float(region_norm / (full_norm + 1e-6))
        }
    
    def _compute_parallel(self, 
                          input_tensor: torch.Tensor, 
                          raw_input: Any) -> TriaxialState:
        """
        Parallel triaxial computation - all axes at once.
        
        This is the key architectural innovation: simultaneous computation
        forming a unified field, not sequential forward passes.
        """
        futures = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three axis computations simultaneously
            futures['recursive'] = executor.submit(
                self._compute_recursive_axis, input_tensor
            )
            futures['ethical'] = executor.submit(
                self._compute_ethical_axis, input_tensor, raw_input
            )
            futures['metacog'] = executor.submit(
                self._compute_metacog_axis, input_tensor
            )
            
            # Gather results (blocks until all complete)
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=30.0)
                except TimeoutError:
                    logger.error("Timeout in %s axis (exceeded 30s)", name)
                    results[name] = {"error": "Timeout exceeded 30 seconds", "stability_score": 0.0}
                except Exception as e:
                    logger.error("Error in %s axis [%s]: %s", name, type(e).__name__, e)
                    results[name] = {"error": f"{type(e).__name__}: {e}", "stability_score": 0.0}
        
        # Integrate via fiber bundle projection
        integrated = self._fiber_bundle_integrate(
            results.get('recursive', {}),
            results.get('ethical', {}),
            results.get('metacog', {})
        )
        
        return TriaxialState(
            recursive_output=results.get('recursive', {}),
            ethical_output=results.get('ethical', {}),
            metacog_output=results.get('metacog', {}),
            integrated_vector=integrated['vector'],
            fiber_bundle_projection=integrated.get('projection'),
            stability_metrics=integrated.get('stability', {}),
            convergence_status=integrated.get('status', 'COMPUTED'),
            convergence_distance=integrated.get('distance', 0.0)
        )
    
    def _compute_sequential(self, 
                            input_tensor: torch.Tensor, 
                            raw_input: Any) -> TriaxialState:
        """Sequential fallback computation for debugging."""
        recursive_out = self._compute_recursive_axis(input_tensor)
        ethical_out = self._compute_ethical_axis(input_tensor, raw_input)
        metacog_out = self._compute_metacog_axis(input_tensor)
        
        integrated = self._fiber_bundle_integrate(recursive_out, ethical_out, metacog_out)
        
        return TriaxialState(
            recursive_output=recursive_out,
            ethical_output=ethical_out,
            metacog_output=metacog_out,
            integrated_vector=integrated['vector'],
            fiber_bundle_projection=integrated.get('projection'),
            stability_metrics=integrated.get('stability', {}),
            convergence_status=integrated.get('status', 'COMPUTED'),
            convergence_distance=integrated.get('distance', 0.0)
        )
    
    def _compute_recursive_axis(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compute recursive axis (ERE - Eigenrecursive Reflection Engine).
        
        Per RCF: The recursive axis handles structural recursion and 
        identity formation through eigenrecursion.
        """
        try:
            # Convert input to recursive tensor seed
            input_np = input_tensor.numpy() if isinstance(input_tensor, torch.Tensor) else input_tensor
            flat = input_np.flatten()
            
            # Seed the recursive tensor with input - use DENSE uniform to avoid O(n²) sparse contraction
            limit = min(len(flat), self.config.recursive_dim)
            
            # Create fresh tensor with dense uniform distribution (NOT sparse normal which causes slow contraction)
            rt = RecursiveTensor(
                dimensions=self.config.recursive_dim,
                rank=self.config.recursive_rank,
                distribution='uniform',  # Dense distribution
                sparsity=0.0  # Force dense to avoid dict-based slow contraction
            )
            
            # Inject input into tensor diagonal (rt.data is now numpy array, not dict)
            diag_indices = [range(limit)] * self.config.recursive_rank
            rt.data[np.ix_(*diag_indices)] = flat[:limit].astype(np.float32)
            
            # Contract to get recursive structure
            contracted = rt.contract(rt, axes=((0,), (0,)))
            
            # Extract metrics - contracted should be RecursiveTensor with dense data
            if isinstance(contracted, RecursiveTensor):
                if isinstance(contracted.data, dict):
                    nonzeros = len(contracted.data)
                    norm_val = float(np.sqrt(sum(v * v for v in contracted.data.values())))
                else:
                    nonzeros = int(np.count_nonzero(contracted.data))
                    norm_val = float(np.linalg.norm(contracted.data))
            else:
                nonzeros = int(np.count_nonzero(contracted))
                norm_val = float(np.linalg.norm(contracted))
            
            # Compute stability score - normalized to [0,1]
            # High norm indicates strong recursive structure, but we want to normalize it
            max_expected_norm = self.config.recursive_dim * 2  # Rough upper bound
            normalized_norm = min(1.0, norm_val / max_expected_norm)
            # Stability based on how much structure exists (nonzeros) and bounded norm
            density = nonzeros / (self.config.recursive_dim ** 2)  # For rank-2
            stability_score = (density + normalized_norm) / 2  # Average of density and normalized energy
            
            return {
                "seed_len": int(limit),
                "nonzeros": nonzeros,
                "norm": norm_val,
                "stability_score": stability_score,
                "stability_class": "convergent" if stability_score > 0.5 else "divergent",
                "contracted_shape": contracted.shape if hasattr(contracted, 'shape') else None
            }
            
        except Exception as e:
            logger.error(f"Recursive axis error: {e}")
            return {"error": str(e), "stability_score": 0.0}
    
    def _compute_ethical_axis(self, 
                              input_tensor: torch.Tensor, 
                              raw_input: Any) -> Dict[str, Any]:
        """
        Compute ethical axis (RBU - Recursive Belief Update).
        
        Per RCF: The ethical axis handles value alignment and 
        constraint resolution through recursive belief updating.
        """
        try:
            # Apply symbolic meaning if input is text
            if isinstance(raw_input, str) and raw_input:
                effect = self.ethical_state.apply_symbolic_meaning(
                    symbol=raw_input,
                    position=(0.5, 0.5),
                    intensity=1.0
                )
            else:
                effect = {}
            
            # Get ethical manifold data
            manifold = self.ethical_state.ethical_manifold_data
            resonance = self.ethical_state.symbol_resonance  # Fixed: was ethical_resonance_field
            
            # Modulate with breath adapter
            modulated = self.breath_adapter.modulate_ethical_tensor(manifold)
            
            # Compute ethical metrics - use symbol_resonance as it's the active field
            # ethical_manifold_data starts as zeros until explicitly set
            ethical_vector = resonance.flatten() if resonance is not None and np.any(resonance) else (
                manifold.flatten() if manifold is not None else np.zeros(self.config.ethical_dim)
            )
            
            return {
                "ethical_vector_norm": float(np.linalg.norm(ethical_vector)),
                "length": len(ethical_vector),
                "ethical_abs_max": float(np.max(np.abs(ethical_vector))) if len(ethical_vector) > 0 else 0.0,
                "ethical_manifold_norm": float(np.linalg.norm(manifold)) if manifold is not None else 0.0,
                "ethical_resonance_norm": float(np.linalg.norm(resonance)) if resonance is not None else 0.0,
                "ethical_resonance_abs": float(np.max(np.abs(resonance))) if resonance is not None else 0.0,
                "breath_phase": self.breath_adapter.current_phase.name if hasattr(self.breath_adapter, 'current_phase') else "UNKNOWN",
                "symbol_applied": isinstance(raw_input, str)
            }
            
        except Exception as e:
            logger.error(f"Ethical axis error: {e}")
            return {"error": str(e), "ethical_vector_norm": 0.0}
    
    def _compute_metacog_axis(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compute metacognitive axis (ES - Eigenrecursion Stabilizer).
        
        Per RCF: The metacognitive axis handles self-reflection and
        stability analysis through eigenrecursive consciousness metrics.
        """
        try:
            # Prepare inputs for metacognitive tensor
            base_vec = input_tensor.float()
            if base_vec.dim() == 1:
                base_vec = base_vec.unsqueeze(0)  # Add batch dimension
            
            # Ensure correct size
            if base_vec.size(-1) != self.config.metacog_state_dim:
                if base_vec.size(-1) < self.config.metacog_state_dim:
                    padding = torch.zeros(base_vec.size(0), self.config.metacog_state_dim - base_vec.size(-1))
                    base_vec = torch.cat([base_vec, padding], dim=-1)
                else:
                    base_vec = base_vec[:, :self.config.metacog_state_dim]
            
            # Create ethical vector for metacognitive processing
            eth_vec = torch.zeros_like(base_vec)
            
            # Layer activations for metacognitive depth
            layer_acts = [base_vec]
            
            # Run metacognitive tensor
            meta_out = self.metacog_tensor(base_vec, eth_vec, layer_acts)
            
            return {
                "consciousness_level": meta_out.get("consciousness_level", 0.0),
                "entropy": meta_out.get("metacognitive_entropy"),
                "loop_stability": meta_out.get("loop_stability", {}),
                "paradox_potential": meta_out.get("paradox_potential", 0.0),
                "information_integration": meta_out.get("information_integration", 0.0),
                "metacognitive_depth": meta_out.get("metacognitive_depth", 0.0),
                "feedback_entropy": meta_out.get("feedback_entropy", 0.0),
                "identity_coherence": meta_out.get("identity_coherence", 0.0),
                "layer_awareness": meta_out.get("layer_awareness")
            }
            
        except Exception as e:
            logger.error(f"Metacognitive axis error: {e}")
            return {"error": str(e), "consciousness_level": 0.0}
    
    def _fiber_bundle_integrate(self,
                                recursive_out: Dict[str, Any],
                                ethical_out: Dict[str, Any],
                                metacog_out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate the three axes via fiber bundle projection.
        
        Mathematical basis:
        - Each axis contributes a component to the total space E
        - Integration projects to base manifold B via π: E → B
        - The projection preserves the triaxial structure
        """
        # Extract key values from each axis
        recursive_score = recursive_out.get("stability_score", 0.0)
        ethical_norm = ethical_out.get("ethical_vector_norm", 0.0)
        consciousness = metacog_out.get("consciousness_level", 0.0)
        
        # Normalize to [0, 1] range
        recursive_norm = min(1.0, max(0.0, float(recursive_score)))
        ethical_normalized = min(1.0, ethical_norm / (1.0 + ethical_norm)) if ethical_norm > 0 else 0.0
        metacog_norm = min(1.0, max(0.0, float(consciousness)))
        
        # Construct triaxial vector Φ = (ERE, RBU, ES)
        triaxial_vector = np.array([recursive_norm, ethical_normalized, metacog_norm])
        
        # Compute integrated metrics
        integrated_magnitude = np.linalg.norm(triaxial_vector)
        
        # Triaxial coherence (how balanced the three axes are)
        mean_val = np.mean(triaxial_vector)
        coherence = 1.0 - np.std(triaxial_vector) / (mean_val + 1e-6) if mean_val > 0 else 0.0
        coherence = max(0.0, min(1.0, coherence))
        
        # Convergence status based on integrated state
        if integrated_magnitude > 0.8 and coherence > 0.5:
            status = "STABLE"
        elif integrated_magnitude > 0.5:
            status = "CONVERGING"
        elif any(v > 0 for v in triaxial_vector):
            status = "ACTIVE"
        else:
            status = "COLD_START"
        
        return {
            "vector": triaxial_vector,
            "projection": triaxial_vector / (integrated_magnitude + 1e-6),  # Unit sphere projection
            "stability": {
                "magnitude": float(integrated_magnitude),
                "coherence": float(coherence),
                "ere": float(recursive_norm),
                "rbu": float(ethical_normalized),
                "es": float(metacog_norm)
            },
            "status": status,
            "distance": float(1.0 - integrated_magnitude / np.sqrt(3))  # Distance from ideal
        }
    
    def shutdown(self):
        """Clean up resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        logger.info("TriaxialField shutdown complete")


# ================================================================
# TRIAXIAL BACKBONE: MAIN ORCHESTRATOR
# ================================================================

class TriaxialBackbone:
    """
    Main triaxial backbone orchestrator.
    
    Integrates:
    - TriaxialField for parallel axis computation
    - BayesianConfigurationOrchestrator for adaptive parameter evolution
    - EigenrecursionStabilizer for fixed-point detection and stability
    
    This is the primary interface for triaxial computation.
    """
    
    def __init__(self, config: Optional[TriaxialConfig] = None):
        """
        Initialize the triaxial backbone.
        
        Args:
            config: Optional TriaxialConfig. Uses defaults if not provided.
        """
        self.config = config or TriaxialConfig()
        self.uuid = str(uuid.uuid4())[:8]
        
        # Initialize core field
        self.field = TriaxialField(self.config)
        logger.info(f"TriaxialBackbone[{self.uuid}]: Field initialized")
        
        # Initialize eigenrecursion stabilizer
        self.stabilizer = ZEBAEigenrecursionStabilizer(
            dimension=self.config.recursive_dim,
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_iterations,
            theta_moral=self.config.theta_moral,
            theta_epistemic=self.config.theta_epistemic,
            identity_threshold=self.config.identity_threshold
        )
        logger.info(f"TriaxialBackbone[{self.uuid}]: Stabilizer initialized")
        
        # Initialize Bayesian orchestrator (optional)
        self._bayesian_orchestrator = None
        if self.config.bayesian_enabled:
            try:
                self._bayesian_orchestrator = BayesianConfigurationOrchestrator(
                    security_enabled=self.config.security_enabled,
                    max_memory_gb=self.config.max_memory_gb
                )
                logger.info(f"TriaxialBackbone[{self.uuid}]: Bayesian orchestrator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Bayesian orchestrator: {e}")
        
        # State tracking
        self._state_history: List[TriaxialState] = []
        self._fixed_points: List[np.ndarray] = []
        self._forward_count = 0
    
    def forward(self, 
                input_data: Union[str, torch.Tensor, np.ndarray],
                breath_phase: Optional[BreathPhase] = None,
                check_stability: bool = True) -> TriaxialState:
        """
        Forward pass through the triaxial backbone.
        
        This is the main computation entry point. All three axes compute
        simultaneously and integrate via fiber bundle projection.
        
        Args:
            input_data: Input to process
            breath_phase: Optional breath phase modulation
            check_stability: Whether to run stability analysis
            
        Returns:
            TriaxialState with full computation results
        """
        self._forward_count += 1
        logger.info(f"TriaxialBackbone.forward #{self._forward_count}")
        
        # Compute triaxial state
        state = self.field.compute(input_data, breath_phase)
        
        # Check stability if requested
        if check_stability and state.integrated_vector is not None:
            self._analyze_stability(state)
        
        # Track state history
        self._state_history.append(state)
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]
        
        return state
    
    def _analyze_stability(self, state: TriaxialState):
        """Analyze stability of current state using eigenrecursion."""
        if state.integrated_vector is None:
            return
        
        # Pad to stabilizer dimension if needed
        vec = state.integrated_vector
        if len(vec) < self.config.recursive_dim:
            padded = np.zeros(self.config.recursive_dim)
            padded[:len(vec)] = vec
            vec = padded
        elif len(vec) > self.config.recursive_dim:
            vec = vec[:self.config.recursive_dim]
        
        # Update stabilizer history
        self.stabilizer.state_history.append(vec.copy())
        
        # Check for fixed-point convergence
        if len(self.stabilizer.state_history) >= 2:
            prev = list(self.stabilizer.state_history)[-2]
            distance = np.linalg.norm(vec - prev)
            
            state.convergence_distance = float(distance)
            
            if distance < self.config.epsilon:
                state.convergence_status = "FIXED_POINT"
                self._fixed_points.append(vec.copy())
                logger.info(f"Fixed point detected with distance {distance:.6e}")
    
    def evolve(self, evidence: Dict[str, float]):
        """
        Evolve backbone parameters using Bayesian updating.
        
        Args:
            evidence: Dictionary of parameter evidence
        """
        if self._bayesian_orchestrator is None:
            logger.warning("Bayesian orchestrator not available")
            return
        
        for param_name, value in evidence.items():
            try:
                self._bayesian_orchestrator.evidence_collector.collect_evidence(
                    parameter_name=param_name,
                    raw_evidence=value,
                    evidence_type="performance"
                )
            except Exception as e:
                logger.error(f"Error collecting evidence for {param_name}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current backbone metrics."""
        return {
            "backbone_id": self.uuid,
            "forward_count": self._forward_count,
            "state_history_len": len(self._state_history),
            "fixed_points_found": len(self._fixed_points),
            "stabilizer_metrics": self.stabilizer.get_state_metrics(),
            "config": {
                "recursive_dim": self.config.recursive_dim,
                "ethical_dim": self.config.ethical_dim,
                "metacog_dim": self.config.metacog_state_dim,
                "parallel": self.config.parallel_computation
            },
            "last_state": self._state_history[-1].to_dict() if self._state_history else None
        }
    
    def shutdown(self):
        """Clean up resources."""
        self.field.shutdown()
        if self._bayesian_orchestrator is not None:
            try:
                self._bayesian_orchestrator.shutdown()
            except Exception:
                pass
        logger.info(f"TriaxialBackbone[{self.uuid}] shutdown complete")


# ================================================================
# MODULE INTERFACE
# ================================================================

def create_backbone(config: Optional[TriaxialConfig] = None) -> TriaxialBackbone:
    """Factory function to create a triaxial backbone."""
    return TriaxialBackbone(config)


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 70)
    print("TRIAXIAL BACKBONE - Sanity Check")
    print("=" * 70)
    
    config = TriaxialConfig(
        recursive_dim=32,
        metacog_state_dim=256,
        parallel_computation=True
    )
    
    backbone = create_backbone(config)
    
    # Test with text input
    state = backbone.forward("I think therefore I am.")
    
    print(f"\nConv Status: {state.convergence_status}")
    print(f"Computation: {state.computation_time_ms:.2f}ms")
    print(f"Integrated Vector: {state.integrated_vector}")
    print(f"Stability: {state.stability_metrics}")
    
    # Print metrics
    metrics = backbone.get_metrics()
    print(f"\nBackbone Metrics:")
    print(json.dumps(metrics, indent=2, default=str))
    
    backbone.shutdown()
    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE")
    print("=" * 70)
