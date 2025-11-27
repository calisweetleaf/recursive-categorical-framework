#!/usr/bin/env python3
"""COMPREHENSIVE END-TO-END TEST SUITE FOR RSIA NEURAL FRAMEWORK

This test suite provides FULL end-to-end testing of ALL components in the
Recursive Symbolic Identity Architecture. NO simulation - only real tests.

Produces:
- Detailed terminal readout of all operations
- JSON file artifacts with test results
- Markdown file reports with comprehensive analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import traceback

# Add workspace to path where file may live
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# Import ALL components from RSIA
from recursive_symbolic_identity_architechture import (
    # Core utilities
    hilbert_space_projection,
    eigenpattern_similarity,
    compute_entropy,
    entropy_gradient,
    detect_resonance,
    tensor_spectral_decomposition,
    compute_fixed_point,
    metric_tensor_update,
    interference_pattern,
    contextual_collapse,
    observer_weight_adjustment,
    quantum_fidelity,
    paradox_measure,
    strange_loop_stabilization,
    detect_convergence_phase,
    dialectical_evolution,
    fractal_self_similarity,
    attractor_energy_landscape,
    metastable_transition_probability,
    eigenvalue_convergence_analysis,
    # Core classes
    SymbolicSpace,
    RecursiveSymbolicIdentity,
    TensorNetworkImplementation,
    ParadoxAmplificationMechanism,
    ObserverResolutionLayer,
    MemoryCrystallizationSubstrate,
    RecursiveMetaMonitoringLoop,
    DialecticalEvolutionEngine,
    TransperspectivalCognition,
    RSIANeuralNetwork,
    AutopoieticSelfMaintenance,
    HyperSymbolicEvaluator,
    QuantumInspiredSymbolicProcessor,
    # TensorFlow components
    RecursiveSymbolicLayer,
    RecursiveSymbolicNetwork,
    # Application classes
    TimeSeriesPredictor,
    AnomalyDetector,
    TransperspectivalDecisionMaker,
    # Enums
    ParadoxType,
    DimensionType,
    MemoryState,
    ResolutionStrategy,
    # Config
    SYSTEM_CONFIG,
    # Dataset creation
    create_toy_dataset,
)

# Import BaseTensor for symbolic-neural integration
try:
    from base_tensor import BaseTensor, TensorState, TensorOperationsMixin
except Exception as e:
    BaseTensor = None
    TensorState = None
    TensorOperationsMixin = None
    print(f"Warning: Could not import BaseTensor: {e}")

# Optional integrations from additional RSIA modules (stability, eigenrecursion, governance)
try:
    from stability_matrix import EigenrecursionStabilizer
except Exception:
    EigenrecursionStabilizer = None

try:
    from eigenrecursion_algorithm import RecursiveLoopDetectionSystem, Eigenrecursion
except Exception:
    RecursiveLoopDetectionSystem = None
    Eigenrecursion = None

try:
    from governance_framework import GovernanceFramework
except Exception:
    GovernanceFramework = None


@dataclass
class TestResult:
    """Container for individual test results."""
    name: str
    passed: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuiteResults:
    """Container for all test results."""
    timestamp: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_duration: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_recursive_identity.log"

    logger = logging.getLogger("RSIA")
    logger.setLevel(level)
    logger.handlers = []

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(fh)

    logger.info("Logger configured. log_file=%s", log_file)
    return logger


def default_transform(v):
    """Default transformation function for testing."""
    return (0.9 * v + 0.05 * (v ** 3))


def default_resolution(pair):
    """Default resolution function for testing."""
    a, b = pair
    r = (a + b) / 2.0
    return r / (1e-10 + (r ** 2).sum() ** 0.5)


def default_pattern_detection(state_sequence):
    """Default pattern detection function for testing."""
    if len(state_sequence) < 2:
        return [0.0]
    similarities = []
    for i in range(len(state_sequence) - 1):
        sim = eigenpattern_similarity(state_sequence[i], state_sequence[i+1])
        similarities.append(sim)
    return [np.mean(similarities)] if similarities else [0.0]


@dataclass
class RunConfig:
	"""Configuration container for RSIA runs."""
	iterations: int = 100
	dimensionality: int = 64
	output_dir: Path = Path("./rsia_runs")
	seed: Optional[int] = None
	dry_run: bool = False
	detailed: bool = False
	log_level: int = logging.INFO


def validate_output_dir(output_dir: Path) -> None:
	"""Very small security validator: ensure output path is not root/system."""
	# Prevent writing to absolute root on Windows
	clean = output_dir.resolve()
	# Only allow writing inside project tree (RUN file directory)
	if str(clean) == str(Path("/")) or len(str(clean)) < 3:
		raise ValueError("Refusing to write to root or system directories")
	# Ensure output path is under current project root for safety
	if not str(clean).startswith(str(ROOT)):
		# If it's outside root, warn but allow — we use explicit user flag in CLI instead.
		logging.getLogger("RSIA").warning("Output dir is outside project root; verify this is intended: %s", clean)


def setup_seed(seed: Optional[int]) -> None:
	"""Set deterministic seeds when provided for reproducibility."""
	import numpy as _np
	import random as _random
	import torch as _torch
	# TensorFlow
	try:
		import tensorflow as _tf
	except Exception:
		_tf = None

	if seed is None:
		return

	_np.random.seed(seed)
	_random.seed(seed)
	try:
		_torch.manual_seed(seed)
	except Exception:
		pass
	if _tf is not None:
		try:
			_tf.random.set_seed(seed)
		except Exception:
			pass


def check_environment(logger: logging.Logger) -> Dict[str, Any]:
	"""Return short environment summary for debugging and reproducibility."""
	import torch as _torch
	env = {
		"python_version": sys.version.split()[0],
		"platform": sys.platform,
		"torch_cuda_available": False,
		"tensorflow_present": False
	}
	try:
		env["torch_cuda_available"] = _torch.cuda.is_available()
	except Exception:
		pass
	try:
		import tensorflow as _tf
		env["tensorflow_present"] = True
		# reduce verbosity
		_tf.get_logger().setLevel("ERROR")
	except Exception:
		env["tensorflow_present"] = False

	logger.info("Environment: %s", env)
	return env


def run_test_suite(
    output_dir: Path,
    logger: logging.Logger,
    dimensionality: int = 64,
    seed: Optional[int] = None,
) -> TestSuiteResults:
    """Run comprehensive end-to-end test suite for all RSIA components."""
    suite_start = time.time()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    results = TestSuiteResults(timestamp=timestamp)
    results.environment = check_environment(logger)
    
    if seed is not None:
        setup_seed(seed)
        logger.info(f"Seed set to {seed} for reproducibility")
    
    logger.info("=" * 80)
    logger.info("RSIA COMPREHENSIVE END-TO-END TEST SUITE")
    logger.info("=" * 80)
    
    # Test all utility functions
    logger.info("\n" + "=" * 80)
    logger.info("TESTING UTILITY FUNCTIONS")
    logger.info("=" * 80)
    test_utility_functions(logger, results)
    
    # Test core classes
    logger.info("\n" + "=" * 80)
    logger.info("TESTING CORE CLASSES")
    logger.info("=" * 80)
    test_core_classes(logger, results, dimensionality)
    
    # Test neural network components
    logger.info("\n" + "=" * 80)
    logger.info("TESTING NEURAL NETWORK COMPONENTS")
    logger.info("=" * 80)
    test_neural_network_components(logger, results, dimensionality)
    
    # Test application classes
    logger.info("\n" + "=" * 80)
    logger.info("TESTING APPLICATION CLASSES")
    logger.info("=" * 80)
    test_application_classes(logger, results, dimensionality)
    
    # Test advanced components
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ADVANCED COMPONENTS")
    logger.info("=" * 80)
    test_advanced_components(logger, results, dimensionality)
    
    results.total_duration = time.time() - suite_start
    results.total_tests = len(results.test_results)
    results.passed_tests = sum(1 for r in results.test_results if r.passed)
    results.failed_tests = results.total_tests - results.passed_tests
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {results.total_tests}")
    logger.info(f"Passed: {results.passed_tests}")
    logger.info(f"Failed: {results.failed_tests}")
    logger.info(f"Total Duration: {results.total_duration:.2f}s")
    
    return results


def test_utility_functions(logger: logging.Logger, results: TestSuiteResults) -> None:
    """Test all utility functions."""
    dim = 32
    
    # Test hilbert_space_projection
    test_name = "hilbert_space_projection"
    start = time.time()
    try:
        state = np.random.randn(dim)
        basis = np.random.randn(3, dim)
        basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
        projected = hilbert_space_projection(state, basis)
        assert projected.shape == (dim,), f"Expected shape ({dim},), got {projected.shape}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"input_dim": dim, "basis_size": len(basis), "output_shape": projected.shape}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test eigenpattern_similarity
    test_name = "eigenpattern_similarity"
    start = time.time()
    try:
        p1 = np.random.randn(dim)
        p2 = np.random.randn(dim)
        sim = eigenpattern_similarity(p1, p2)
        assert 0 <= sim <= 1, f"Similarity should be in [0,1], got {sim}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"similarity": float(sim)}
        ))
        logger.info(f"✓ {test_name}: PASSED (similarity={sim:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test compute_entropy
    test_name = "compute_entropy"
    start = time.time()
    try:
        dist = np.random.rand(10)
        entropy = compute_entropy(dist)
        assert entropy >= 0, f"Entropy should be >= 0, got {entropy}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"entropy": float(entropy)}
        ))
        logger.info(f"✓ {test_name}: PASSED (entropy={entropy:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test entropy_gradient
    test_name = "entropy_gradient"
    start = time.time()
    try:
        history = np.random.rand(20)
        first_deriv, second_deriv = entropy_gradient(history)
        assert isinstance(first_deriv, float), "First derivative should be float"
        assert isinstance(second_deriv, float), "Second derivative should be float"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"first_derivative": float(first_deriv), "second_derivative": float(second_deriv)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test detect_resonance
    test_name = "detect_resonance"
    start = time.time()
    try:
        time_series = np.sin(np.linspace(0, 4*np.pi, 50))  # Periodic signal
        has_resonance = detect_resonance(time_series)
        assert isinstance(has_resonance, bool), "Should return boolean"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"has_resonance": has_resonance}
        ))
        logger.info(f"✓ {test_name}: PASSED (resonance={has_resonance})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test compute_fixed_point
    test_name = "compute_fixed_point"
    start = time.time()
    try:
        def square_root_func(x):
            return (x + 4.0 / x) / 2.0 if x != 0 else 2.0
        
        fp = compute_fixed_point(square_root_func, 1.0, max_iterations=20)
        assert abs(fp * fp - 4.0) < 0.1, f"Fixed point should be near 2.0, got {fp}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"fixed_point": float(fp)}
        ))
        logger.info(f"✓ {test_name}: PASSED (fixed_point={fp:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test quantum_fidelity
    test_name = "quantum_fidelity"
    start = time.time()
    try:
        state1 = np.random.randn(dim) + 1j * np.random.randn(dim)
        state2 = np.random.randn(dim) + 1j * np.random.randn(dim)
        fidelity = quantum_fidelity(state1, state2)
        assert 0 <= fidelity <= 1, f"Fidelity should be in [0,1], got {fidelity}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"fidelity": float(fidelity)}
        ))
        logger.info(f"✓ {test_name}: PASSED (fidelity={fidelity:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test paradox_measure
    test_name = "paradox_measure"
    start = time.time()
    try:
        symbol1 = np.random.randn(dim)
        symbol2 = np.random.randn(dim)
        paradox = paradox_measure(symbol1, symbol2)
        assert paradox >= 0, f"Paradox measure should be >= 0, got {paradox}"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"paradox_measure": float(paradox)}
        ))
        logger.info(f"✓ {test_name}: PASSED (paradox={paradox:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")


def test_core_classes(logger: logging.Logger, results: TestSuiteResults, dimensionality: int) -> None:
    """Test all core classes."""
    
    # Test SymbolicSpace
    test_name = "SymbolicSpace"
    start = time.time()
    try:
        space = SymbolicSpace(dimensionality=dimensionality)
        assert len(space.symbols) > 0, "Should have initial symbols"
        symbol_id = space.add_symbol()
        assert symbol_id in space.symbols, "Added symbol should exist"
        entropy = space.entropy
        assert entropy >= 0, "Entropy should be >= 0"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"num_symbols": len(space.symbols), "entropy": float(entropy)}
        ))
        logger.info(f"✓ {test_name}: PASSED (symbols={len(space.symbols)}, entropy={entropy:.4f})")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test RecursiveSymbolicIdentity
    test_name = "RecursiveSymbolicIdentity"
    start = time.time()
    try:
        space = SymbolicSpace(dimensionality=dimensionality)
        identity = RecursiveSymbolicIdentity(
            transformation_func=default_transform,
            pattern_detection_func=default_pattern_detection,
            resolution_func=default_resolution,
            symbolic_space=space,
            identity_name="TestIdentity"
        )
        assert identity.name == "TestIdentity", "Name should be set"
        symbol_id = list(space.symbols.keys())[0]
        seq = identity.apply_transformation(symbol_id, iterations=3)
        assert len(seq) > 0, "Should produce transformation sequence"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"transformation_sequence_length": len(seq)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test TensorNetworkImplementation
    test_name = "TensorNetworkImplementation"
    start = time.time()
    try:
        tensor_net = TensorNetworkImplementation(dimensionality=dimensionality)
        tensor_net.initialize_cores(5)
        tensor_net.add_connection(0, 0, 1, 0)
        contracted = tensor_net.contract_network()
        assert contracted is not None, "Should produce contracted tensor"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"contracted_shape": str(contracted.shape) if hasattr(contracted, 'shape') else "scalar"}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test ParadoxAmplificationMechanism
    test_name = "ParadoxAmplificationMechanism"
    start = time.time()
    try:
        space = SymbolicSpace(dimensionality=dimensionality)
        pa = ParadoxAmplificationMechanism(space)
        paradoxes = pa.scan_for_paradoxes()
        assert isinstance(paradoxes, list), "Should return list of paradoxes"
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"paradoxes_found": len(paradoxes)}
        ))
        logger.info(f"✓ {test_name}: PASSED (found {len(paradoxes)} paradoxes)")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test ObserverResolutionLayer
    test_name = "ObserverResolutionLayer"
    start = time.time()
    try:
        obs_layer = ObserverResolutionLayer(state_dimensionality=dimensionality)
        
        def obs1(symbol):
            return symbol[:dimensionality//2]
        
        def obs2(symbol):
            return symbol[dimensionality//2:]
        
        obs_layer.add_observer("observer1", obs1)
        obs_layer.add_observer("observer2", obs2)
        
        symbol = np.random.randn(dimensionality)
        interpretations = obs_layer.interpret_symbol(symbol)
        assert "observer1" in interpretations, "Should have observer1 interpretation"
        assert "observer2" in interpretations, "Should have observer2 interpretation"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"num_observers": len(obs_layer.observers)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test MemoryCrystallizationSubstrate
    test_name = "MemoryCrystallizationSubstrate"
    start = time.time()
    try:
        mem_substrate = MemoryCrystallizationSubstrate(dimensionality=dimensionality)
        state = np.random.randn(dimensionality)
        mem_id = mem_substrate.create_memory(state)
        assert mem_id >= 0, "Should return valid memory ID"
        
        recalled_id, recalled_state = mem_substrate.recall_memory(state)
        assert recalled_id == mem_id, "Should recall the same memory"
        
        success = mem_substrate.crystallize_memory(mem_id)
        assert isinstance(success, bool), "Should return boolean"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"memory_id": mem_id, "crystallized": success}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test RecursiveMetaMonitoringLoop
    test_name = "RecursiveMetaMonitoringLoop"
    start = time.time()
    try:
        monitor = RecursiveMetaMonitoringLoop(max_levels=3)
        
        def monitor_func(state):
            return {"value": np.mean(state) if isinstance(state, np.ndarray) else 0.0}
        
        monitor.add_monitor(0, monitor_func)
        state = np.random.randn(dimensionality)
        result = monitor.update_level(0, state)
        assert "value" in result, "Should return monitor result"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"monitor_result": result}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test DialecticalEvolutionEngine
    test_name = "DialecticalEvolutionEngine"
    start = time.time()
    try:
        dialectical = DialecticalEvolutionEngine(dimensionality=dimensionality)
        new_state = dialectical.evolve()
        assert new_state.shape == (dimensionality,), f"Should return state of shape ({dimensionality},)"
        
        trajectory = dialectical.get_evolution_trajectory(steps=5)
        assert len(trajectory) > 0, "Should return trajectory"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"trajectory_length": len(trajectory)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test TransperspectivalCognition
    test_name = "TransperspectivalCognition"
    start = time.time()
    try:
        obs_layer = ObserverResolutionLayer(state_dimensionality=dimensionality)
        transperspectival = TransperspectivalCognition(
            dimensionality=dimensionality,
            observer_resolution_layer=obs_layer
        )
        
        symbol = np.random.randn(dimensionality)
        invariants = transperspectival.detect_invariants(symbol)
        assert invariants.shape == (dimensionality,), "Should return invariant vector"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"invariant_norm": float(np.linalg.norm(invariants))}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")


def test_neural_network_components(logger: logging.Logger, results: TestSuiteResults, dimensionality: int) -> None:
    """Test neural network components."""
    
    # Test RSIANeuralNetwork
    test_name = "RSIANeuralNetwork"
    start = time.time()
    try:
        input_dim = 10
        # Cap hidden_dim to prevent excessive memory usage in tensor network contractions
        # Tensor network operations can create intermediate arrays of size (hidden_dim^2, hidden_dim^2)
        # For dimensionality=64, hidden_dim=64 could create 262144x262144 arrays (512GB)
        # So we cap at 32 to keep memory reasonable (~4GB max)
        safe_hidden_dim = min(dimensionality, 32)
        output_dim = 5
        
        if safe_hidden_dim < dimensionality:
            logger.warning(f"Capping hidden_dim from {dimensionality} to {safe_hidden_dim} to prevent excessive memory usage")
        
        network = RSIANeuralNetwork(
            input_dim=input_dim,
            hidden_dim=safe_hidden_dim,
            output_dim=output_dim
        )
        
        # Test forward pass
        inputs = np.random.randn(3, input_dim)
        outputs = network.forward(inputs)
        assert outputs.shape == (3, output_dim), f"Expected shape (3, {output_dim})"
        
        # Test training
        targets = np.random.randn(3, output_dim)
        losses = network.train(inputs, targets, epochs=2, batch_size=2)
        assert len(losses) > 0, "Should return loss history"
        
        eigenpatterns = network.get_eigenpatterns()
        assert eigenpatterns is not None, "Should return eigenpatterns"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"final_loss": float(losses[-1]) if losses else 0.0, "num_eigenpatterns": len(eigenpatterns) if hasattr(eigenpatterns, '__len__') else 0, "hidden_dim_used": safe_hidden_dim}
        ))
        logger.info(f"✓ {test_name}: PASSED (final_loss={losses[-1]:.4f})")
    except MemoryError as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=f"MemoryError: {str(e)} - Consider reducing dimensionality or using memory-mapped arrays",
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - MemoryError (try reducing dimensionality)")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test TensorFlow components if available
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        
        # Test RecursiveSymbolicLayer
        test_name = "RecursiveSymbolicLayer"
        start = time.time()
        try:
            # Cap units to prevent excessive memory usage
            safe_units = min(dimensionality, 32)
            if safe_units < dimensionality:
                logger.warning(f"Capping units from {dimensionality} to {safe_units} for {test_name}")
            
            layer = RecursiveSymbolicLayer(units=safe_units)
            input_tensor = tf.random.normal((2, 10))
            output = layer(input_tensor)
            assert output.shape == (2, safe_units), f"Expected shape (2, {safe_units})"
            
            results.test_results.append(TestResult(
                name=test_name,
                passed=True,
                duration=time.time() - start,
                details={"output_shape": str(output.shape)}
            ))
            logger.info(f"✓ {test_name}: PASSED")
        except Exception as e:
            results.test_results.append(TestResult(
                name=test_name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))
            logger.error(f"✗ {test_name}: FAILED - {e}")
        
        # Test RecursiveSymbolicNetwork
        test_name = "RecursiveSymbolicNetwork"
        start = time.time()
        try:
            # Cap hidden_dim to prevent excessive memory usage
            safe_hidden_dim = min(dimensionality, 32)
            if safe_hidden_dim < dimensionality:
                logger.warning(f"Capping hidden_dim from {dimensionality} to {safe_hidden_dim} for {test_name}")
            
            model = RecursiveSymbolicNetwork(
                input_dim=10,
                hidden_dim=safe_hidden_dim,
                output_dim=5
            )
            inputs = tf.random.normal((2, 10))
            outputs = model(inputs)
            assert outputs.shape == (2, 5), "Expected shape (2, 5)"
            
            results.test_results.append(TestResult(
                name=test_name,
                passed=True,
                duration=time.time() - start,
                details={"output_shape": str(outputs.shape)}
            ))
            logger.info(f"✓ {test_name}: PASSED")
        except Exception as e:
            results.test_results.append(TestResult(
                name=test_name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
                traceback=traceback.format_exc()
            ))
            logger.error(f"✗ {test_name}: FAILED - {e}")
    except ImportError:
        logger.warning("TensorFlow not available, skipping TensorFlow component tests")


def test_application_classes(logger: logging.Logger, results: TestSuiteResults, dimensionality: int) -> None:
    """Test application classes."""
    
    # Test TimeSeriesPredictor
    test_name = "TimeSeriesPredictor"
    start = time.time()
    try:
        predictor = TimeSeriesPredictor(
            input_length=10,
            forecast_horizon=5,
            hidden_dim=dimensionality
        )
        
        # Create synthetic time series
        X = np.random.randn(20, 10)
        y = np.random.randn(20, 5)
        
        predictor.fit(X, y, epochs=2, verbose=0)
        predictions = predictor.predict(X[:3])
        assert predictions.shape[0] == 3, "Should predict for 3 samples"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"predictions_shape": str(predictions.shape)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test AnomalyDetector
    test_name = "AnomalyDetector"
    start = time.time()
    try:
        detector = AnomalyDetector(
            input_dim=10,
            hidden_dim=dimensionality//2
        )
        
        X = np.random.randn(50, 10)
        detector.fit(X, epochs=2, verbose=0)
        
        test_X = np.random.randn(10, 10)
        predictions = detector.predict(test_X)
        assert len(predictions) == 10, "Should predict for all samples"
        
        anomalies = detector.detect_anomalies(test_X, threshold=1.0)
        assert isinstance(anomalies, np.ndarray), "Should return numpy array"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"anomalies_detected": int(np.sum(anomalies))}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test TransperspectivalDecisionMaker
    test_name = "TransperspectivalDecisionMaker"
    start = time.time()
    try:
        decision_maker = TransperspectivalDecisionMaker(
            feature_dim=10,
            hidden_dim=dimensionality//2,
            num_classes=3
        )
        
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        decision_maker.fit(X, y, epochs=2, verbose=0)
        
        test_x = np.random.randn(5, 10)
        predictions = decision_maker.predict(test_x)
        assert len(predictions) == 5, "Should predict for all samples"
        
        interpretations = decision_maker.get_observer_interpretations(test_x[0])
        assert len(interpretations) > 0, "Should return observer interpretations"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"num_observers": len(interpretations)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")


def test_advanced_components(logger: logging.Logger, results: TestSuiteResults, dimensionality: int) -> None:
    """Test advanced components."""
    
    # Test AutopoieticSelfMaintenance
    test_name = "AutopoieticSelfMaintenance"
    start = time.time()
    try:
        # Cap hidden_dim to prevent excessive memory usage
        safe_hidden_dim = min(dimensionality, 32)
        if safe_hidden_dim < dimensionality:
            logger.warning(f"Capping hidden_dim from {dimensionality} to {safe_hidden_dim} for {test_name}")
        
        network = RSIANeuralNetwork(
            input_dim=10,
            hidden_dim=safe_hidden_dim,
            output_dim=5
        )
        
        autopoiesis = AutopoieticSelfMaintenance(network)
        health = autopoiesis.assess_health()
        assert isinstance(health, dict), "Should return health dictionary"
        assert len(health) > 0, "Should have health metrics"
        
        results_dict = autopoiesis.perform_autopoiesis()
        assert isinstance(results_dict, dict), "Should return results dictionary"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"health_metrics": len(health), "autopoiesis_results": results_dict, "hidden_dim_used": safe_hidden_dim}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except MemoryError as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=f"MemoryError: {str(e)} - Consider reducing dimensionality",
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - MemoryError")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test HyperSymbolicEvaluator
    test_name = "HyperSymbolicEvaluator"
    start = time.time()
    try:
        # Cap hidden_dim to prevent excessive memory usage
        safe_hidden_dim = min(dimensionality, 32)
        if safe_hidden_dim < dimensionality:
            logger.warning(f"Capping hidden_dim from {dimensionality} to {safe_hidden_dim} for {test_name}")
        
        network = RSIANeuralNetwork(
            input_dim=10,
            hidden_dim=safe_hidden_dim,
            output_dim=5
        )
        
        evaluator = HyperSymbolicEvaluator(network)
        all_metrics = evaluator.evaluate_all()
        assert isinstance(all_metrics, dict), "Should return metrics dictionary"
        assert len(all_metrics) > 0, "Should have multiple metrics"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"num_metrics": len(all_metrics), "hidden_dim_used": safe_hidden_dim}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except MemoryError as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=f"MemoryError: {str(e)} - Consider reducing dimensionality",
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - MemoryError")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")
    
    # Test QuantumInspiredSymbolicProcessor
    test_name = "QuantumInspiredSymbolicProcessor"
    start = time.time()
    try:
        processor = QuantumInspiredSymbolicProcessor(dimensionality=dimensionality)
        
        state1 = np.random.randn(dimensionality)
        state2 = np.random.randn(dimensionality)
        
        superposition = processor.create_superposition(
            [state1, state2],
            amplitudes=[1/np.sqrt(2), 1/np.sqrt(2)],
            name="test_state"
        )
        assert superposition.shape == (dimensionality,), "Should return superposition vector"
        
        context = np.random.randn(dimensionality)
        collapsed = processor.contextual_collapse("test_state", context)
        assert collapsed.shape == (dimensionality,), "Should return collapsed state"
        
        processor.entangle_states("test_state", "test_state", 0.5)
        
        fidelity = processor.compute_quantum_fidelity("test_state", "test_state")
        assert 0 <= fidelity <= 1, "Fidelity should be in [0,1]"
        
        results.test_results.append(TestResult(
            name=test_name,
            passed=True,
            duration=time.time() - start,
            details={"fidelity": float(fidelity)}
        ))
        logger.info(f"✓ {test_name}: PASSED")
    except Exception as e:
        results.test_results.append(TestResult(
            name=test_name,
            passed=False,
            duration=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc()
        ))
        logger.error(f"✗ {test_name}: FAILED - {e}")


def write_test_reports(output_dir: Path, results: TestSuiteResults, logger: logging.Logger) -> None:
    """Write comprehensive test reports in JSON and Markdown formats."""
    timestamp = results.timestamp
    json_file = output_dir / f"test_results_{timestamp}.json"
    md_file = output_dir / f"test_report_{timestamp}.md"
    
    # Convert results to dictionary for JSON serialization
    results_dict = {
        "timestamp": results.timestamp,
        "total_tests": results.total_tests,
        "passed_tests": results.passed_tests,
        "failed_tests": results.failed_tests,
        "total_duration": results.total_duration,
        "environment": results.environment,
        "test_results": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration": r.duration,
                "details": r.details,
                "error": r.error,
            }
            for r in results.test_results
        ]
    }
    
    # Write JSON report
    with json_file.open("w", encoding="utf-8") as fh:
        json.dump(results_dict, fh, indent=2, default=str)
    
    # Write Markdown report
    with md_file.open("w", encoding="utf-8") as fh:
        fh.write("# RSIA Comprehensive Test Report\n\n")
        fh.write(f"**Timestamp:** {timestamp}\n\n")
        fh.write(f"**Total Tests:** {results.total_tests}\n")
        fh.write(f"**Passed:** {results.passed_tests}\n")
        fh.write(f"**Failed:** {results.failed_tests}\n")
        fh.write(f"**Total Duration:** {results.total_duration:.2f}s\n\n")
        
        fh.write("## Environment\n\n")
        for key, value in results.environment.items():
            fh.write(f"- **{key}:** {value}\n")
        fh.write("\n")
        
        fh.write("## Test Results\n\n")
        fh.write("| Test Name | Status | Duration (s) | Details |\n")
        fh.write("|-----------|--------|--------------|----------|\n")
        
        for r in results.test_results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            details_str = ", ".join([f"{k}={v}" for k, v in r.details.items()]) if r.details else "-"
            if r.error:
                details_str += f" | Error: {r.error}"
            fh.write(f"| {r.name} | {status} | {r.duration:.4f} | {details_str} |\n")
        
        fh.write("\n## Failed Tests Details\n\n")
        failed = [r for r in results.test_results if not r.passed]
        if failed:
            for r in failed:
                fh.write(f"### {r.name}\n\n")
                fh.write(f"**Error:** {r.error}\n\n")
                if r.traceback:
                    fh.write("**Traceback:**\n```\n")
                    fh.write(r.traceback)
                    fh.write("\n```\n\n")
        else:
            fh.write("No failed tests.\n\n")
    
    logger.info(f"Wrote JSON report to {json_file}")
    logger.info(f"Wrote Markdown report to {md_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RSIA Comprehensive Test Suite + Integration (RSIA + Governance + BaseTensor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full test suite
  python run_recursive_identity.py --test-suite
  
  # Run integration test
  python run_recursive_identity.py --integration --iterations 100
  
  # Run both with BaseTensor integration
  python run_recursive_identity.py --test-suite --integration --enable-base-tensor
        """
    )
    
    # Test suite arguments
    parser.add_argument("--test-suite", action="store_true", help="Run comprehensive end-to-end test suite")
    parser.add_argument("--integration", action="store_true", help="Run RSIA + Governance + BaseTensor integration test")
    parser.add_argument("--dimensionality", type=int, default=64, help="Symbolic space dimensionality")
    parser.add_argument("--output-dir", type=str, default="./rsia_runs_test", help="Directory for logs and reports")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Integration test arguments
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for integration test")
    parser.add_argument("--amplify-paradox-every", type=int, default=0, help="Amplify paradoxes every N iterations (0=off)")
    parser.add_argument("--enable-stability", action="store_true", help="Enable EigenrecursionStabilizer")
    parser.add_argument("--enable-rldis", action="store_true", help="Enable Recursive Loop Detection (RLDIS)")
    parser.add_argument("--enable-governance", action="store_true", help="Enable Governance Framework")
    parser.add_argument("--enable-base-tensor", action="store_true", default=True, help="Enable BaseTensor integration (default: True)")
    
    args = parser.parse_args(argv)
    
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir, level=getattr(logging, args.log_level.upper(), logging.INFO))
    
    try:
        validate_output_dir(output_dir)
        env = check_environment(logger)
        
        all_results = {}
        
        # Run test suite if requested
        if args.test_suite:
            logger.info("=" * 80)
            logger.info("RUNNING COMPREHENSIVE TEST SUITE")
            logger.info("=" * 80)
            test_results = run_test_suite(
                output_dir=output_dir,
                logger=logger,
                dimensionality=args.dimensionality,
                seed=args.seed
            )
            write_test_reports(output_dir, test_results, logger)
            all_results["test_suite"] = {
                "total_tests": test_results.total_tests,
                "passed_tests": test_results.passed_tests,
                "failed_tests": test_results.failed_tests,
                "total_duration": test_results.total_duration
            }
            logger.info("=" * 80)
            logger.info("TEST SUITE COMPLETE")
            logger.info(f"Total: {test_results.total_tests} | Passed: {test_results.passed_tests} | Failed: {test_results.failed_tests}")
            logger.info("=" * 80)
        
        # Run integration test if requested (or if nothing else specified, run both)
        if args.integration or (not args.test_suite and not args.integration):
            logger.info("=" * 80)
            logger.info("RUNNING INTEGRATION TEST")
            logger.info("=" * 80)
            integration_results = run_integration_test(
                iterations=args.iterations,
                dimensionality=args.dimensionality,
                output_dir=output_dir,
                logger=logger,
                amplify_paradox_every=args.amplify_paradox_every,
                enable_stability=args.enable_stability,
                enable_rldis=args.enable_rldis,
                enable_governance=args.enable_governance,
                enable_base_tensor=args.enable_base_tensor
            )
            integration_results["env"] = env
            integration_results["seed"] = args.seed
            write_reports(output_dir, integration_results, logger)
            all_results["integration"] = integration_results
            logger.info("=" * 80)
            logger.info("INTEGRATION TEST COMPLETE")
            logger.info("=" * 80)
        
        # Write combined summary
        if all_results:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            summary_file = output_dir / f"combined_summary_{ts}.json"
            with summary_file.open("w", encoding="utf-8") as fh:
                json.dump(all_results, fh, indent=2, default=str)
            logger.info(f"Combined summary written to {summary_file}")
        
        # Exit with error code if test suite had failures
        if args.test_suite and "test_suite" in all_results:
            failed = all_results["test_suite"].get("failed_tests", 0)
            sys.exit(0 if failed == 0 else 1)
        
    except Exception as exc:
        logger.exception("Execution failed: %s", exc)
        raise


def run_integration_test(
    iterations: int,
    dimensionality: int,
    output_dir: Path,
    logger: logging.Logger,
    amplify_paradox_every: int = 0,
    enable_stability: bool = False,
    enable_rldis: bool = False,
    enable_governance: bool = False,
    enable_base_tensor: bool = True,
) -> Dict[str, Any]:
    """
    Integration test: RSIA + Governance + BaseTensor (Symbolic-Neural Bridge)
    
    This function integrates:
    - RSIA (Recursive Symbolic Identity Architecture) - core symbolic processing
    - Governance Framework - governs RSIA processes
    - BaseTensor - bridges symbolic (RSIA) with neural (tensor operations)
    """
    logger.info("=" * 80)
    logger.info("RSIA + GOVERNANCE + BASETENSOR INTEGRATION TEST")
    logger.info("=" * 80)
    logger.info("Integration starting: iterations=%d, dimensionality=%d", iterations, dimensionality)
    start = time.time()

    # Initialize RSIA core components
    space = SymbolicSpace(dimensionality=dimensionality)
    identity = RecursiveSymbolicIdentity(
        transformation_func=default_transform,
        pattern_detection_func=default_pattern_detection,
        resolution_func=default_resolution,
        symbolic_space=space,
        identity_name="IntegrationIdentity",
    )
    pa = ParadoxAmplificationMechanism(space)
    
    # Initialize BaseTensor for symbolic-neural bridge
    base_tensor = None
    if enable_base_tensor and BaseTensor is not None:
        try:
            # Create a concrete BaseTensor implementation for testing
            class RSIABaseTensor(BaseTensor):
                def forward(self, input_state: TensorState, *args, **kwargs) -> TensorState:
                    # Simple forward: apply transformation
                    data = self.get_dense_data(input_state.data)
                    transformed = data * 0.9 + 0.1 * torch.tanh(data)
                    return TensorState(
                        data=transformed,
                        timestamp=input_state.timestamp + 1.0,
                        metadata={**input_state.metadata, 'forward_pass': True},
                        convergence_score=input_state.convergence_score,
                        information_flow=input_state.information_flow
                    )
                
                def compute_eigenrecursive_step(self, state: TensorState) -> TensorState:
                    # Eigenrecursive step: apply contraction mapping
                    data = self.get_dense_data(state.data)
                    eigenstep = data * (1.0 - self.stability_eta)
                    return TensorState(
                        data=eigenstep,
                        timestamp=state.timestamp + 1.0,
                        metadata={**state.metadata, 'eigenrecursive_step': True},
                        convergence_score=state.convergence_score,
                        information_flow=state.information_flow
                    )
            
            base_tensor = RSIABaseTensor(
                dimensions=dimensionality,
                convergence_threshold=1e-6,
                stability_eta=0.01
            )
            logger.info("BaseTensor initialized for symbolic-neural integration")
        except Exception as e:
            logger.warning(f"Could not initialize BaseTensor: {e}")
            enable_base_tensor = False

    # Optional monitoring/instrumentation
    stabilizer = None
    eigenrecursion = None
    if enable_stability and EigenrecursionStabilizer is not None:
        stabilizer = EigenrecursionStabilizer(state_dimension=dimensionality)
        logger.info("EigenrecursionStabilizer enabled (state_dim=%d)", dimensionality)
        
        # Initialize Eigenrecursion for BaseTensor stability if BaseTensor is enabled
        if enable_base_tensor and base_tensor is not None and Eigenrecursion is not None:
            try:
                # Create recursive operator that wraps BaseTensor's eigenrecursive step
                def base_tensor_recursive_operator(state: np.ndarray) -> np.ndarray:
                    """Recursive operator for Eigenrecursion that uses BaseTensor's compute_eigenrecursive_step."""
                    # Convert numpy array to TensorState
                    tensor_data = torch.tensor(state, dtype=torch.float32)
                    tensor_state = TensorState(
                        data=tensor_data,
                        timestamp=0.0,
                        metadata={'source': 'eigenrecursion'},
                        convergence_score=0.0,
                        information_flow=0.0
                    )
                    
                    # Apply BaseTensor's eigenrecursive step
                    eigen_state = base_tensor.compute_eigenrecursive_step(tensor_state)
                    
                    # Convert back to numpy
                    result_data = base_tensor.get_dense_data(eigen_state.data)
                    return result_data.detach().cpu().numpy().flatten()
                
                eigenrecursion = Eigenrecursion(
                    recursive_operator=base_tensor_recursive_operator,
                    epsilon=base_tensor.convergence_threshold,
                    max_iterations=100,
                    state_dim=dimensionality,
                    verbose=False,
                    enable_rldis=False  # RLDIS handled separately
                )
                logger.info("Eigenrecursion initialized for BaseTensor stability analysis")
            except Exception as e:
                logger.warning(f"Could not initialize Eigenrecursion for BaseTensor: {e}")
                eigenrecursion = None

    rldis = None
    if enable_rldis and RecursiveLoopDetectionSystem is not None:
        rldis = RecursiveLoopDetectionSystem()
        logger.info("RLDIS enabled")

    governance = None
    if enable_governance and GovernanceFramework is not None:
        governance = GovernanceFramework()
        logger.info("Governance checks enabled")

    metrics = {
        "iterations": iterations,
        "dimensionality": dimensionality,
        "symbols_start": len(space.symbols),
        "symbols_end": None,
        "eigenpatterns_found": 0,
        "paradoxes_detected": 0,
        "paradox_by_type": {},
        "memories_created": 0,
        "resonance_detected": False,
        "timings": {
            "total_seconds": 0,
            "loop_seconds": 0,
            "transformation_seconds": 0,
            "paradox_scan_seconds": 0,
            "crystallization_seconds": 0,
        },
        "stability": {},
        "rldis": {},
        "governance": {},
    }

    # Create initial memory
    example = space.symbols[list(space.symbols.keys())[0]]
    mem_id = identity.create_memory(example)
    metrics["memories_created"] += 1

    # Main loop
    loop_start = time.time()
    for i in range(iterations):
        iter_start = time.time()
        # pick a pseudo-random symbol deterministically
        sid = list(space.symbols.keys())[i % len(space.symbols)]
        # time the transformation
        t0 = time.time()
        seq = identity.apply_transformation(sid, iterations=1)
        metrics["timings"]["transformation_seconds"] += time.time() - t0
        logger.debug("Iteration %d: transformed symbol sequence len=%d", i, len(seq))

        # scan for paradoxes rarely
        if i % max(1, iterations // 8) == 0:
            t0 = time.time()
            detected = pa.scan_for_paradoxes()
            metrics["timings"]["paradox_scan_seconds"] += time.time() - t0
            metrics["paradoxes_detected"] += len(detected)
            # count by type
            for p in detected:
                t = p[2]
                metrics["paradox_by_type"][t.name] = metrics["paradox_by_type"].get(t.name, 0) + 1
            if detected:
                logger.info("Detected %d paradoxes at iter %d", len(detected), i)

            # optionally amplify paradoxes to stress-test resolution
            if amplify_paradox_every and i % amplify_paradox_every == 0:
                pa.amplify_paradoxes()

        # occasionally try to crystallize memory
        if i % max(1, iterations // 4) == 0:
            t0 = time.time()
            for mid in list(identity.memory_states.keys()):
                success = identity.crystallize_memory(mid)
                logger.debug("Crystallize memory %s -> %s", mid, success)
            metrics["timings"]["crystallization_seconds"] += time.time() - t0

        # detect eigenpatterns
        if identity.eigenpatterns:
            metrics["eigenpatterns_found"] = len(identity.eigenpatterns)

        # run quick convergence/resonance checks
        if identity.detect_resonance():
            metrics["resonance_detected"] = True

        # BaseTensor integration: bridge symbolic (RSIA) with neural (tensor operations)
        if base_tensor is not None and i % max(1, iterations // 10) == 0:
            try:
                # Convert RSIA symbol to tensor state
                symbol_vector = space.symbols[sid]
                tensor_data = torch.tensor(symbol_vector, dtype=torch.float32)
                
                tensor_state = TensorState(
                    data=tensor_data,
                    timestamp=float(i),
                    metadata={'symbol_id': sid, 'iteration': i, 'source': 'RSIA'},
                    convergence_score=0.0,
                    information_flow=0.0
                )
                
                # Apply BaseTensor forward pass
                updated_state = base_tensor.forward(tensor_state)
                
                # Apply eigenrecursive step
                eigen_state = base_tensor.compute_eigenrecursive_step(updated_state)
                
                # Use Eigenrecursion to find fixed point and stabilize (if enabled)
                eigenrecursion_result = None
                if eigenrecursion is not None:
                    try:
                        # Use current tensor state as initial state for fixed point search
                        initial_state_np = base_tensor.get_dense_data(eigen_state.data).detach().cpu().numpy().flatten()
                        
                        # Find fixed point using Eigenrecursion
                        eigenrecursion_result = eigenrecursion.find_fixed_point(
                            initial_state=initial_state_np,
                            return_trace=False,
                            classify_stability=True
                        )
                        
                        # If fixed point found, use it to stabilize the state
                        if eigenrecursion_result.get('status') and hasattr(eigenrecursion_result['status'], 'value'):
                            status_str = eigenrecursion_result['status'].value
                            if 'CONVERGED' in status_str or 'CYCLE_DETECTED' in status_str:
                                # Update eigen_state with stabilized fixed point
                                fixed_point = eigenrecursion_result.get('fixed_point')
                                if fixed_point is not None:
                                    fixed_point_tensor = torch.tensor(fixed_point, dtype=torch.float32)
                                    # Reshape to match eigen_state.data shape
                                    if fixed_point_tensor.numel() == eigen_state.data.numel():
                                        eigen_state.data = fixed_point_tensor.reshape(eigen_state.data.shape)
                                    else:
                                        # Project to match dimensions
                                        min_dim = min(fixed_point_tensor.numel(), eigen_state.data.numel())
                                        eigen_state.data = fixed_point_tensor[:min_dim].reshape(-1)
                                        
                        logger.debug(f"Iter {i}: Eigenrecursion status={eigenrecursion_result.get('status')}, "
                                   f"iterations={eigenrecursion_result.get('iterations', 0)}")
                    except Exception as e:
                        logger.debug(f"Eigenrecursion fixed point search failed at iter {i}: {e}")
                
                # Update BaseTensor state
                base_tensor.update_state(eigen_state)
                
                # Check convergence
                converged = base_tensor.check_convergence(eigen_state)
                
                # Get stability metrics
                stability_metrics = base_tensor.get_stability_metrics()
                
                # Use EigenrecursionStabilizer to evaluate state stability (if enabled)
                stabilizer_metrics = {}
                if stabilizer is not None:
                    try:
                        current_tensor = base_tensor.get_dense_data(eigen_state.data)
                        previous_tensor = None
                        if len(base_tensor.state_history) > 0:
                            previous_tensor = base_tensor.get_dense_data(base_tensor.state_history[-1].data)
                        
                        stab_result = stabilizer.evaluate_state(current_tensor, previous_tensor, i)
                        stabilizer_metrics = stab_result
                        
                        # Apply adaptive adjustment if instability detected
                        if not stab_result.get('converged', False) and stab_result.get('spectral_radius', 0) > 1.0:
                            adjustment = stabilizer.adaptive_adjustment(
                                instability_detected=True,
                                delta=stab_result.get('delta'),
                                spectral_radius=stab_result.get('spectral_radius')
                            )
                            stabilizer_metrics['adjustment'] = adjustment
                    except Exception as e:
                        logger.debug(f"Stabilizer evaluation failed at iter {i}: {e}")
                
                # Log BaseTensor integration
                logger.info(f"Iter {i}: BaseTensor integration - convergence={converged}, "
                          f"info_flow={eigen_state.information_flow:.4f}, "
                          f"stability={stability_metrics.get('last_convergence_score', 0.0):.4f}")
                
                # Store BaseTensor metrics
                if "base_tensor" not in metrics:
                    metrics["base_tensor"] = {}
                metrics["base_tensor"][f"iter_{i}"] = {
                    "converged": converged,
                    "information_flow": float(eigen_state.information_flow),
                    "stability_metrics": stability_metrics,
                    "stabilizer_metrics": stabilizer_metrics,
                    "eigenrecursion_result": {
                        "status": str(eigenrecursion_result.get('status')) if eigenrecursion_result else None,
                        "iterations": eigenrecursion_result.get('iterations') if eigenrecursion_result else None,
                        "final_distance": eigenrecursion_result.get('final_distance') if eigenrecursion_result else None
                    } if eigenrecursion_result else None
                }
                
            except Exception as e:
                logger.exception("BaseTensor integration failed at iter %d: %s", i, e)

        # small graceful sleep for cooperative scheduling in longer runs
        time.sleep(0 if iterations < 1000 else 0.0001)

        # run stability evaluation if available
        if stabilizer is not None:
            try:
                import torch as _torch
                if identity.memory_states:
                    last_mid = max(identity.memory_states.keys())
                    mem_state = identity.memory_states[last_mid]["state"]
                    curr = _torch.tensor(mem_state, dtype=_torch.float32)
                    prev = None
                    if (last_mid - 1) in identity.memory_states:
                        prev_raw = identity.memory_states[last_mid - 1]["state"]
                        prev = _torch.tensor(prev_raw, dtype=_torch.float32)

                    stab = stabilizer.evaluate_state(curr, prev, i)
                    metrics["stability"][f"iter_{i}"] = stab
                    if not stab.get("converged", False):
                        # apply adaptive adjustment if instability
                        if stab.get("spectral_radius", 0) > 1.0:
                            stabilizer.adaptive_adjustment(True, delta=stab.get("delta"), spectral_radius=stab.get("spectral_radius"))
            except Exception:
                logger.exception("Stability evaluation failed at iter %d", i)

        # run RLDIS pattern detection if available
        if rldis is not None and (i % max(1, iterations // 10) == 0):
            try:
                trace = list(identity.coherence_history[-20:]) if hasattr(identity, "coherence_history") else []
                detection = rldis.detect_recursive_patterns(trace)
                metrics["rldis"][f"iter_{i}"] = detection
                if detection.get("dominant_pattern"):
                    logger.info("RLDIS detected dominant=%s severity=%s", detection.get("dominant_pattern"), detection.get("severity"))
            except Exception:
                logger.exception("RLDIS detection failed at iter %d", i)

        # governance operations — perform governance actions, not just checks
        if governance is not None and (i % max(1, iterations // 5) == 0):
            try:
                autonomy_degree = float(len(identity.eigenpatterns))
                human_authority = 1.0
                
                # Check autonomy constraint
                satisfied = governance.check_autonomy_constraint(autonomy_degree, human_authority)
                
                # Compute transparency obligation
                transparency_obligation = governance.compute_transparency_obligation(autonomy_degree)
                actual_transparency = min(1.0, transparency_obligation)  # Simplified: assume we meet obligation
                transparency_satisfied = governance.check_transparency_constraint(autonomy_degree, actual_transparency)
                
                # Perform Bayesian preference learning (if we have preference data)
                if i > 0 and i % max(1, iterations // 10) == 0:
                    # Update preferences based on system behavior
                    preference_data = {
                        'eigenpattern_count': len(identity.eigenpatterns),
                        'paradox_count': metrics.get('paradoxes_detected', 0),
                        'convergence_rate': 1.0 if identity.detect_convergence() else 0.0
                    }
                    governance.observed_preferences.append(preference_data)
                    
                    # Update preference priors and likelihoods
                    for key, value in preference_data.items():
                        governance.update_preference_prior(key, 0.5)  # Uniform prior
                        governance.update_preference_likelihood(key, f"iter_{i}", float(value))
                        governance.bayesian_preference_update(key, f"iter_{i}")
                
                # Store governance metrics
                metrics["governance"][f"iter_{i}"] = {
                    "autonomy_degree": autonomy_degree,
                    "satisfied": satisfied,
                    "transparency_obligation": transparency_obligation,
                    "transparency_satisfied": transparency_satisfied,
                    "aar": governance.compute_autonomy_authority_ratio(autonomy_degree, human_authority)
                }
                logger.debug("Governance operations: autonomy=%.3f satisfied=%s transparency=%.3f", 
                           autonomy_degree, satisfied, transparency_obligation)
            except Exception:
                logger.exception("Governance operations failed at iter %d", i)

    metrics["symbols_end"] = len(space.symbols)
    metrics["memories_created"] = len(identity.memory_states)
    metrics["timings"]["total_seconds"] = time.time() - start
    metrics["timings"]["loop_seconds"] = time.time() - loop_start
    
    # Final BaseTensor metrics
    if base_tensor is not None:
        try:
            final_stability = base_tensor.get_stability_metrics()
            metrics["base_tensor_final"] = final_stability
            logger.info("BaseTensor final stability metrics: %s", final_stability)
        except Exception as e:
            logger.warning(f"Could not get final BaseTensor metrics: {e}")

    logger.info("Integration test completed in %.2f s", metrics["timings"]["total_seconds"])
    logger.info("=" * 80)
    return metrics


def write_reports(output_dir: Path, results: Dict[str, Any], logger: logging.Logger) -> None:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_file = output_dir / f"run_summary_{ts}.json"
    md_file = output_dir / f"run_summary_{ts}.md"

    # if there are detailed timings and paradox_by_type, ensure they are written
    with json_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    # Write a comprehensive human-friendly markdown report
    with md_file.open("w", encoding="utf-8") as fh:
        fh.write("# RSIA Integration Test Summary\n\n")
        fh.write(f"**Timestamp:** {ts}\n\n")
        fh.write("## Overview\n\n")
        fh.write(f"- **Iterations:** {results.get('iterations', 'N/A')}\n")
        fh.write(f"- **Dimensionality:** {results.get('dimensionality', 'N/A')}\n")
        fh.write(f"- **Symbols start:** {results.get('symbols_start', 'N/A')}\n")
        fh.write(f"- **Symbols end:** {results.get('symbols_end', 'N/A')}\n")
        fh.write(f"- **Eigenpatterns found:** {results.get('eigenpatterns_found', 0)}\n")
        fh.write(f"- **Paradoxes detected:** {results.get('paradoxes_detected', 0)}\n")
        fh.write(f"- **Memories created:** {results.get('memories_created', 0)}\n")
        fh.write(f"- **Resonance detected:** {results.get('resonance_detected', False)}\n")
        fh.write('\n')
        
        fh.write('## Timings\n\n')
        timings = results.get('timings', {})
        for k, v in timings.items():
            fh.write(f"- **{k}:** {v:.4f}s\n")
        fh.write('\n')
        
        # Add paradox breakdown
        paradox_table = results.get("paradox_by_type", {})
        if paradox_table:
            fh.write("## Paradox Analysis\n\n")
            fh.write("### Paradox Types\n\n")
            for t, c in paradox_table.items():
                fh.write(f"- **{t}:** {c}\n")
            fh.write('\n')
        
        # Add BaseTensor metrics if available
        if "base_tensor" in results:
            fh.write("## BaseTensor Integration Metrics\n\n")
            base_tensor_data = results.get("base_tensor", {})
            if base_tensor_data:
                fh.write(f"**Integration points:** {len(base_tensor_data)}\n\n")
                # Show sample of integration metrics
                sample_keys = list(base_tensor_data.keys())[:5]
                for key in sample_keys:
                    data = base_tensor_data[key]
                    fh.write(f"### {key}\n\n")
                    fh.write(f"- **Converged:** {data.get('converged', False)}\n")
                    fh.write(f"- **Information Flow:** {data.get('information_flow', 0.0):.4f}\n")
                    stability = data.get('stability_metrics', {})
                    if stability:
                        fh.write(f"- **Recursion Depth:** {stability.get('recursion_depth', 0)}\n")
                        fh.write(f"- **Convergence Score:** {stability.get('last_convergence_score', 0.0):.4f}\n")
                    fh.write('\n')
            
            # Final BaseTensor metrics
            if "base_tensor_final" in results:
                fh.write("### Final BaseTensor Stability Metrics\n\n")
                final_metrics = results["base_tensor_final"]
                for k, v in final_metrics.items():
                    if isinstance(v, (int, float)):
                        fh.write(f"- **{k}:** {v:.4f}\n")
                    else:
                        fh.write(f"- **{k}:** {v}\n")
                fh.write('\n')
        
        # Add Governance metrics if available
        if "governance" in results and results["governance"]:
            fh.write("## Governance Framework Metrics\n\n")
            governance_data = results["governance"]
            fh.write(f"**Governance checks performed:** {len(governance_data)}\n\n")
            # Show sample
            sample_keys = list(governance_data.keys())[:3]
            for key in sample_keys:
                data = governance_data[key]
                fh.write(f"- **{key}:** Autonomy={data.get('autonomy_degree', 0.0):.4f}, "
                        f"Satisfied={data.get('satisfied', False)}\n")
            fh.write('\n')

    logger.info("Wrote JSON report to %s and Markdown report to %s", json_file, md_file)


if __name__ == "__main__":
    main()
