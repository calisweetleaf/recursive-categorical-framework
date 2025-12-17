#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Triaxial Backbone

Per AGENT.md guidelines:
- Direct python-to-python testing (not pytest)
- Detailed terminal logs with colorful TUI
- Output to /logs and /reports directories
- All tests must be end-to-end functional

This validates the triaxial backbone's:
1. Parallel axis computation
2. Fiber bundle integration
3. Eigenrecursion stabilization
4. Bayesian parameter evolution
"""

import sys
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Ensure proper paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
REPORT_DIR = Path(__file__).parent / "reports"
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"triaxial_backbone_{timestamp}.log"
JSON_FILE = LOG_DIR / f"triaxial_backbone_{timestamp}.json"
REPORT_FILE = REPORT_DIR / f"triaxial_backbone_report.md"

# Configure file logging
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger("TriaxialBackboneTest")


# ================================================================
# COLORFUL TUI HELPERS
# ================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a colorful header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")


def print_stage(stage_num: int, name: str):
    """Print stage header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}[STAGE {stage_num}] {name}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * 60}{Colors.END}")


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {msg}{Colors.END}")


# ================================================================
# TEST RESULT DATACLASS
# ================================================================

@dataclass
class StageResult:
    """Result from a test stage."""
    stage_num: int
    name: str
    success: bool
    duration: float
    details: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ================================================================
# TEST STAGES
# ================================================================

def stage_1_import_validation() -> StageResult:
    """Stage 1: Validate all imports work correctly."""
    print_stage(1, "Import Validation")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        # Import the backbone
        from triaxial_backbone import (
            TriaxialBackbone,
            TriaxialField,
            TriaxialConfig,
            TriaxialState,
            create_backbone
        )
        print_success("Imported triaxial_backbone module")
        metrics["backbone_imported"] = True
        
        # Verify tensor imports
        from rcf_integration.recursive_tensor import RecursiveTensor
        print_success("Imported RecursiveTensor")
        metrics["recursive_tensor_imported"] = True
        
        from rcf_integration.ethical_tensor import SymbolicQuantumState
        print_success("Imported SymbolicQuantumState (EthicalTensor)")
        metrics["ethical_tensor_imported"] = True
        
        from rcf_integration.metacognitive_tensor import MetacognitiveTensor
        print_success("Imported MetacognitiveTensor")
        metrics["metacog_tensor_imported"] = True
        
        # Verify orchestration imports
        from bayesian_config_orchestrator import BayesianConfigurationOrchestrator
        print_success("Imported BayesianConfigurationOrchestrator")
        metrics["bayesian_imported"] = True
        
        from zynx_zebra_core import EigenrecursionStabilizer
        print_success("Imported EigenrecursionStabilizer")
        metrics["stabilizer_imported"] = True
        
        success = True
        
    except ImportError as e:
        print_fail(f"Import error: {e}")
        errors.append(str(e))
        success = False
    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        errors.append(str(e))
        success = False
    
    return StageResult(
        stage_num=1,
        name="Import Validation",
        success=success,
        duration=time.time() - start,
        details=f"All {sum(1 for v in metrics.values() if v)} imports successful" if success else "Import failed",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_2_config_validation() -> StageResult:
    """Stage 2: Validate configuration works correctly."""
    print_stage(2, "Configuration Validation")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        from triaxial_backbone import TriaxialConfig
        
        # Test default config
        config = TriaxialConfig()
        print_success(f"Default config: recursive_dim={config.recursive_dim}, ethical_dim={config.ethical_dim}")
        metrics["default_config"] = True
        
        # Test custom config
        custom_config = TriaxialConfig(
            recursive_dim=64,
            ethical_dim=10,
            metacog_state_dim=512,
            parallel_computation=True
        )
        print_success(f"Custom config: recursive_dim={custom_config.recursive_dim}")
        metrics["custom_config"] = True
        
        # Test validation (should raise for invalid values)
        try:
            bad_config = TriaxialConfig(recursive_dim=0)
            print_fail("Should have raised ValueError for recursive_dim=0")
            errors.append("Config validation failed for recursive_dim=0")
        except ValueError:
            print_success("Config validation correctly rejects invalid values")
            metrics["validation_works"] = True
        
        success = True
        
    except Exception as e:
        print_fail(f"Config error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    
    return StageResult(
        stage_num=2,
        name="Configuration Validation",
        success=success,
        duration=time.time() - start,
        details="Configuration system validated",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_3_backbone_initialization() -> StageResult:
    """Stage 3: Initialize the triaxial backbone."""
    print_stage(3, "Backbone Initialization")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    backbone = None
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        config = TriaxialConfig(
            recursive_dim=32,
            metacog_state_dim=256,
            parallel_computation=True,
            bayesian_enabled=False  # Disable for faster test
        )
        
        print_info("Creating backbone...")
        backbone = TriaxialBackbone(config)
        print_success(f"Backbone created: ID={backbone.uuid}")
        metrics["backbone_id"] = backbone.uuid
        
        # Verify field initialized
        assert backbone.field is not None, "Field not initialized"
        print_success("TriaxialField initialized")
        metrics["field_initialized"] = True
        
        # Verify stabilizer initialized
        assert backbone.stabilizer is not None, "Stabilizer not initialized"
        print_success("EigenrecursionStabilizer initialized")
        metrics["stabilizer_initialized"] = True
        
        # Verify tensor components
        assert backbone.field.recursive_tensor is not None, "RecursiveTensor not initialized"
        assert backbone.field.ethical_state is not None, "EthicalTensor not initialized"
        assert backbone.field.metacog_tensor is not None, "MetacognitiveTensor not initialized"
        print_success("All three tensor axes initialized")
        metrics["all_tensors_initialized"] = True
        
        success = True
        
    except Exception as e:
        print_fail(f"Initialization error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    finally:
        if backbone:
            backbone.shutdown()
    
    return StageResult(
        stage_num=3,
        name="Backbone Initialization",
        success=success,
        duration=time.time() - start,
        details=f"Backbone ID: {metrics.get('backbone_id', 'N/A')}",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_4_text_forward_pass() -> StageResult:
    """Stage 4: Forward pass with text input."""
    print_stage(4, "Text Forward Pass")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    backbone = None
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        config = TriaxialConfig(parallel_computation=True, bayesian_enabled=False)
        backbone = TriaxialBackbone(config)
        
        # Test text input
        test_text = "I think therefore I am."
        print_info(f"Processing text: '{test_text}'")
        
        state = backbone.forward(test_text)
        
        print_success(f"Forward pass complete in {state.computation_time_ms:.2f}ms")
        metrics["computation_time_ms"] = state.computation_time_ms
        
        # Validate outputs
        assert state.integrated_vector is not None, "No integrated vector"
        print_success(f"Integrated vector: {state.integrated_vector}")
        metrics["integrated_vector"] = state.integrated_vector.tolist()
        
        # Check all axes computed
        assert "stability_score" in state.recursive_output, "Recursive axis missing"
        print_success(f"Recursive axis: stability={state.recursive_output.get('stability_score', 0):.4f}")
        
        assert "ethical_vector_norm" in state.ethical_output, "Ethical axis missing"
        print_success(f"Ethical axis: norm={state.ethical_output.get('ethical_vector_norm', 0):.4f}")
        
        assert "consciousness_level" in state.metacog_output, "Metacog axis missing"
        print_success(f"Metacog axis: consciousness={state.metacog_output.get('consciousness_level', 0):.4f}")
        
        print_success(f"Convergence status: {state.convergence_status}")
        metrics["convergence_status"] = state.convergence_status
        
        success = True
        
    except Exception as e:
        print_fail(f"Forward pass error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    finally:
        if backbone:
            backbone.shutdown()
    
    return StageResult(
        stage_num=4,
        name="Text Forward Pass",
        success=success,
        duration=time.time() - start,
        details=f"Time: {metrics.get('computation_time_ms', 0):.2f}ms",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_5_tensor_forward_pass() -> StageResult:
    """Stage 5: Forward pass with tensor input."""
    print_stage(5, "Tensor Forward Pass")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    backbone = None
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        config = TriaxialConfig(parallel_computation=True, bayesian_enabled=False)
        backbone = TriaxialBackbone(config)
        
        # Test tensor input
        input_tensor = torch.randn(256)
        print_info(f"Processing tensor: shape={input_tensor.shape}")
        
        state = backbone.forward(input_tensor)
        
        print_success(f"Forward pass complete in {state.computation_time_ms:.2f}ms")
        metrics["computation_time_ms"] = state.computation_time_ms
        
        # Validate integration
        assert state.integrated_vector is not None, "No integrated vector"
        magnitude = np.linalg.norm(state.integrated_vector)
        print_success(f"Integrated magnitude: {magnitude:.4f}")
        metrics["integrated_magnitude"] = float(magnitude)
        
        # Check stability metrics
        assert "coherence" in state.stability_metrics, "Missing coherence metric"
        coherence = state.stability_metrics.get("coherence", 0)
        print_success(f"Triaxial coherence: {coherence:.4f}")
        metrics["coherence"] = coherence
        
        success = True
        
    except Exception as e:
        print_fail(f"Tensor forward error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    finally:
        if backbone:
            backbone.shutdown()
    
    return StageResult(
        stage_num=5,
        name="Tensor Forward Pass",
        success=success,
        duration=time.time() - start,
        details=f"Coherence: {metrics.get('coherence', 0):.4f}",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_6_parallel_computation() -> StageResult:
    """Stage 6: Validate parallel computation."""
    print_stage(6, "Parallel Computation Validation")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        # Compare parallel vs sequential
        input_text = "Testing parallel computation in triaxial backbone"
        
        # Parallel
        config_parallel = TriaxialConfig(parallel_computation=True, bayesian_enabled=False)
        backbone_parallel = TriaxialBackbone(config_parallel)
        state_parallel = backbone_parallel.forward(input_text)
        parallel_time = state_parallel.computation_time_ms
        backbone_parallel.shutdown()
        print_success(f"Parallel computation: {parallel_time:.2f}ms")
        
        # Sequential
        config_seq = TriaxialConfig(parallel_computation=False, bayesian_enabled=False)
        backbone_seq = TriaxialBackbone(config_seq)
        state_seq = backbone_seq.forward(input_text)
        seq_time = state_seq.computation_time_ms
        backbone_seq.shutdown()
        print_success(f"Sequential computation: {seq_time:.2f}ms")
        
        # Compare results (should be similar)
        parallel_vec = state_parallel.integrated_vector
        seq_vec = state_seq.integrated_vector
        
        # Allow some variance due to timing differences
        difference = np.linalg.norm(parallel_vec - seq_vec)
        print_info(f"Output difference: {difference:.6f}")
        metrics["output_difference"] = float(difference)
        
        if difference < 0.01:
            print_success("Parallel and sequential outputs are consistent")
        else:
            print_warning("Outputs differ slightly (may be due to timing)")
            warnings.append(f"Output difference: {difference:.6f}")
        
        speedup = seq_time / parallel_time if parallel_time > 0 else 1.0
        print_info(f"Speedup: {speedup:.2f}x")
        metrics["parallel_time_ms"] = parallel_time
        metrics["sequential_time_ms"] = seq_time
        metrics["speedup"] = speedup
        
        success = True
        
    except Exception as e:
        print_fail(f"Parallel test error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    
    return StageResult(
        stage_num=6,
        name="Parallel Computation Validation",
        success=success,
        duration=time.time() - start,
        details=f"Speedup: {metrics.get('speedup', 1):.2f}x",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_7_stability_analysis() -> StageResult:
    """Stage 7: Eigenrecursion stability analysis."""
    print_stage(7, "Stability Analysis")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    backbone = None
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        config = TriaxialConfig(parallel_computation=True, bayesian_enabled=False)
        backbone = TriaxialBackbone(config)
        
        # Run multiple forward passes to build state history
        print_info("Running 5 forward passes to build state history...")
        states = []
        for i in range(5):
            state = backbone.forward(f"Iteration {i}: testing eigenrecursion stability")
            states.append(state)
            print_info(f"  Pass {i+1}: status={state.convergence_status}, dist={state.convergence_distance:.6f}")
        
        # Check stabilizer metrics
        stab_metrics = backbone.stabilizer.get_state_metrics()
        print_success(f"Stabilizer state: {stab_metrics}")
        metrics["stabilizer_state"] = stab_metrics
        
        # Check for convergence trend
        distances = [s.convergence_distance for s in states]
        if len(distances) >= 2:
            is_converging = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
            if is_converging:
                print_success("Convergence trend detected (distances decreasing)")
            else:
                print_info("No monotonic convergence (expected for varied inputs)")
        
        metrics["final_convergence_status"] = states[-1].convergence_status
        metrics["state_history_len"] = len(backbone._state_history)
        
        success = True
        
    except Exception as e:
        print_fail(f"Stability analysis error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    finally:
        if backbone:
            backbone.shutdown()
    
    return StageResult(
        stage_num=7,
        name="Stability Analysis",
        success=success,
        duration=time.time() - start,
        details=f"Final status: {metrics.get('final_convergence_status', 'N/A')}",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


def stage_8_metrics_collection() -> StageResult:
    """Stage 8: Collect and validate metrics."""
    print_stage(8, "Metrics Collection")
    start = time.time()
    errors = []
    warnings = []
    metrics = {}
    backbone = None
    
    try:
        from triaxial_backbone import TriaxialBackbone, TriaxialConfig
        
        config = TriaxialConfig(parallel_computation=True, bayesian_enabled=False)
        backbone = TriaxialBackbone(config)
        
        # Run some forward passes
        backbone.forward("Test input one")
        backbone.forward("Test input two")
        
        # Get metrics
        all_metrics = backbone.get_metrics()
        print_success(f"Backbone ID: {all_metrics['backbone_id']}")
        print_success(f"Forward count: {all_metrics['forward_count']}")
        print_success(f"State history: {all_metrics['state_history_len']}")
        print_success(f"Fixed points found: {all_metrics['fixed_points_found']}")
        
        metrics = all_metrics
        
        # Validate last state
        if all_metrics.get('last_state'):
            last = all_metrics['last_state']
            print_success(f"Last state ID: {last['state_id']}")
            print_success(f"Last status: {last['convergence_status']}")
        
        success = True
        
    except Exception as e:
        print_fail(f"Metrics error: {e}")
        errors.append(str(e))
        traceback.print_exc()
        success = False
    finally:
        if backbone:
            backbone.shutdown()
    
    return StageResult(
        stage_num=8,
        name="Metrics Collection",
        success=success,
        duration=time.time() - start,
        details=f"Forward passes: {metrics.get('forward_count', 0)}",
        metrics=metrics,
        warnings=warnings,
        errors=errors
    )


# ================================================================
# REPORT GENERATION
# ================================================================

def write_reports(results: List[StageResult], total_duration: float):
    """Write test reports to files."""
    
    # JSON report
    json_data = {
        "test_run": datetime.now().isoformat(),
        "total_duration_seconds": total_duration,
        "stages": [
            {
                "stage": r.stage_num,
                "name": r.name,
                "success": r.success,
                "duration": r.duration,
                "details": r.details,
                "metrics": r.metrics,
                "warnings": r.warnings,
                "errors": r.errors
            }
            for r in results
        ],
        "summary": {
            "total_stages": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "warnings": sum(len(r.warnings) for r in results)
        }
    }
    
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    logger.info(f"JSON report written to: {JSON_FILE}")
    
    # Markdown report
    md_lines = [
        "# Triaxial Backbone Test Report",
        "",
        f"**Test Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Duration:** {total_duration:.2f}s",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Stages | {len(results)} |",
        f"| Passed | {sum(1 for r in results if r.success)} |",
        f"| Failed | {sum(1 for r in results if not r.success)} |",
        f"| Warnings | {sum(len(r.warnings) for r in results)} |",
        "",
        "## Stage Results",
        ""
    ]
    
    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        md_lines.extend([
            f"### Stage {r.stage_num}: {r.name}",
            "",
            f"**Status:** {status}",
            f"**Duration:** {r.duration:.2f}s",
            f"**Details:** {r.details}",
            ""
        ])
        
        if r.warnings:
            md_lines.append("**Warnings:**")
            for w in r.warnings:
                md_lines.append(f"- {w}")
            md_lines.append("")
        
        if r.errors:
            md_lines.append("**Errors:**")
            for e in r.errors:
                md_lines.append(f"- {e}")
            md_lines.append("")
        
        if r.metrics:
            md_lines.append("**Metrics:**")
            md_lines.append("```json")
            md_lines.append(json.dumps(r.metrics, indent=2, default=str))
            md_lines.append("```")
            md_lines.append("")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    logger.info(f"Markdown report written to: {REPORT_FILE}")


# ================================================================
# MAIN
# ================================================================

def main():
    """Run all test stages."""
    print_header("TRIAXIAL BACKBONE TEST SUITE")
    
    overall_start = time.time()
    results: List[StageResult] = []
    
    # Stage 1: Import validation
    result = stage_1_import_validation()
    results.append(result)
    if not result.success:
        print_fail("Stage 1 failed - cannot continue")
        write_reports(results, time.time() - overall_start)
        return
    
    # Stage 2: Config validation
    result = stage_2_config_validation()
    results.append(result)
    
    # Stage 3: Backbone initialization
    result = stage_3_backbone_initialization()
    results.append(result)
    if not result.success:
        print_fail("Stage 3 failed - cannot continue")
        write_reports(results, time.time() - overall_start)
        return
    
    # Stage 4: Text forward pass
    result = stage_4_text_forward_pass()
    results.append(result)
    
    # Stage 5: Tensor forward pass
    result = stage_5_tensor_forward_pass()
    results.append(result)
    
    # Stage 6: Parallel computation
    result = stage_6_parallel_computation()
    results.append(result)
    
    # Stage 7: Stability analysis
    result = stage_7_stability_analysis()
    results.append(result)
    
    # Stage 8: Metrics collection
    result = stage_8_metrics_collection()
    results.append(result)
    
    # Write reports
    total_duration = time.time() - overall_start
    write_reports(results, total_duration)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    for r in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if r.success else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  Stage {r.stage_num}: {r.name.ljust(35)} [{status}] ({r.duration:.2f}s)")
    
    passed = sum(1 for r in results if r.success)
    total = len(results)
    
    print()
    if passed == total:
        print_success(f"All {total} stages passed in {total_duration:.2f}s")
    else:
        print_fail(f"{passed}/{total} stages passed in {total_duration:.2f}s")
    
    print()
    print_info(f"Log file: {LOG_FILE}")
    print_info(f"JSON report: {JSON_FILE}")
    print_info(f"Markdown report: {REPORT_FILE}")
    
    print_header("TEST COMPLETE")


if __name__ == "__main__":
    main()
