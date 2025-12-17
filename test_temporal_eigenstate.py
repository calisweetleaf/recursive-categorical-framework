"""
Staged integration test for internal_clock with temporal_eigenstate.
Allows progressive warm-up and verification of each subsystem.

Philosophy: Let each layer stabilize before adding complexity.
This prevents interference between initialization phases and allows
proper frequency entrainment in the biological oscillators.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from internal_clock import (
    TemporalCoherence,
    CircadianPhase,
    AttentionalState,
    TimeScale,
)
from rcf_integration.temporal_eigenstate import (
    TemporalEigenstate,
    RecursiveStabilizationPoint,
    TemporalEigenstateNode,
    EchoCollapseMethod,
)

# ============================================================================
# Helper Functions
# ============================================================================

def analyze_frequency_coupling(oscillator_history: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Analyze frequency coupling and phase relationships in oscillator data.
    
    For frequency-based architectures, we're looking for:
    - Stable oscillation amplitudes
    - Phase locking between related frequencies
    - Harmonic relationships
    """
    analysis = {}
    
    for name, values in oscillator_history.items():
        if len(values) < 10:
            continue
            
        numeric_values = [v for v in values if isinstance(v, (int, float, np.floating))]
        if not numeric_values:
            continue
        arr = np.array(numeric_values, dtype=float)
        
        # Basic stats
        analysis[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
        }
        
        # Frequency domain analysis (if enough samples)
        if len(arr) > 20:
            # Simple FFT to check for dominant frequencies
            fft = np.fft.fft(arr - np.mean(arr))
            freqs = np.fft.fftfreq(len(arr))
            power = np.abs(fft) ** 2
            
            # Find peak frequency
            peak_idx = np.argmax(power[1:len(power)//2]) + 1
            peak_freq = abs(freqs[peak_idx])
            
            analysis[name]["dominant_frequency"] = float(peak_freq)
            analysis[name]["spectral_power"] = float(np.sum(power))
    
    return analysis


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
CLOCK_PATH = LOGS_DIR / "temporal_clock"
CLOCK_PATH.mkdir(parents=True, exist_ok=True)

# Timing configuration - CRITICAL for frequency-based architectures
# For holographic/frequency substrates, you need multiple oscillation periods
# to establish stable phase relationships and interference patterns.
#
# Minimum recommendations:
# - Alpha waves (10 Hz): Need ~1 second for 10 cycles
# - Heartbeat (75 bpm): Need ~5 seconds for 6 beats  
# - Breath (15/min): Need ~8 seconds for 2 breaths
# - Circadian: Need hours/days, but we can test phase progression
#
# The burn-in period is where magic happens - oscillators phase-lock,
# interference patterns stabilize, and the holographic substrate forms.

STAGE_WARMUP_TIME = 3.0  # Seconds between stages (allow settling)
OSCILLATOR_SETTLE_TIME = 2.0  # Time for new components to couple
INTEGRATION_CYCLES = 5  # Number of update cycles per integration test
BURNIN_DURATION = 10.0  # Initial oscillator stabilization (critical!)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """Configure a logger with both file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # File handler
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


LOGGER = setup_logger("StagedClockIntegration")


# ============================================================================
# Stage Results
# ============================================================================

@dataclass
class StageResult:
    """Result from a single test stage."""
    stage_num: int
    name: str
    success: bool
    duration: float
    details: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Stage 0: Clock Burn-in (Let Oscillators Stabilize)
# ============================================================================

def stage_0_clock_burnin(burnin_duration: float = 10.0) -> tuple[StageResult, TemporalCoherence]:
    """
    Stage 0: Initialize clock and let oscillators run for several cycles.
    This establishes baseline rhythms and phase relationships.
    
    For frequency-based architectures, this is critical - you need multiple
    periods of your fastest oscillators before meaningful patterns emerge.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 0: Clock Burn-in (Oscillator Stabilization)")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        LOGGER.info("Creating TemporalCoherence instance...")
        clock = TemporalCoherence(
            base_path=CLOCK_PATH,
            auto_save=False,
            auto_update_interval=0.0,
        )
        
        LOGGER.info(f"Running burn-in for {burnin_duration}s to establish oscillator rhythms...")
        
        # Sample oscillator states throughout burn-in
        sample_interval = 0.5  # Sample every 500ms
        num_samples = int(burnin_duration / sample_interval)
        
        oscillator_history = {
            "circadian": [],
            "alpha_wave": [],
            "heart_beat": [],
            "alertness": [],
            "phase": [],
        }
        
        for i in range(num_samples):
            clock.update()
            
            # Sample key oscillators
            bio = clock.biological_clock
            oscillator_history["circadian"].append(
                bio.oscillators["circadian"].get_value()
            )
            oscillator_history["alpha_wave"].append(
                bio.oscillators["alpha_wave"].get_value()
            )
            oscillator_history["heart_beat"].append(
                bio.oscillators["heart_beat"].get_value()
            )
            oscillator_history["alertness"].append(clock.get_alertness())
            oscillator_history["phase"].append(clock.get_circadian_phase().name)
            
            if (i + 1) % 10 == 0:
                elapsed = (i + 1) * sample_interval
                LOGGER.debug(
                    f"  Burn-in progress: {elapsed:.1f}s / {burnin_duration}s "
                    f"(alertness={oscillator_history['alertness'][-1]:.4f})"
                )
            
            time.sleep(sample_interval)
        
        # Analyze oscillator stability
        def calculate_stability(values):
            """Calculate coefficient of variation as stability metric."""
            if len(values) < 2:
                return 0.0
            arr = np.array(values)
            return np.std(arr) / (np.abs(np.mean(arr)) + 1e-6)
        
        # Analyze frequency coupling
        LOGGER.info("Analyzing frequency domain characteristics...")
        frequency_analysis = analyze_frequency_coupling(oscillator_history)
        
        metrics["burnin_duration"] = burnin_duration
        metrics["num_samples"] = num_samples
        def _serialize_samples(samples: List[Any]) -> List[Any]:
            serialized = []
            for v in samples[-10:]:
                if isinstance(v, (int, float, np.floating)):
                    serialized.append(float(v))
                else:
                    serialized.append(str(v))
            return serialized

        metrics["oscillator_history"] = {
            k: _serialize_samples(vals)
            for k, vals in oscillator_history.items()
        }
        metrics["frequency_analysis"] = frequency_analysis
        
        # Calculate stability metrics
        metrics["stability"] = {
            "circadian": calculate_stability(oscillator_history["circadian"]),
            "alpha_wave": calculate_stability(oscillator_history["alpha_wave"]),
            "heart_beat": calculate_stability(oscillator_history["heart_beat"]),
            "alertness": calculate_stability(oscillator_history["alertness"]),
        }
        
        LOGGER.info("Oscillator stability metrics:")
        for osc_name, stability in metrics["stability"].items():
            LOGGER.info(f"  {osc_name}: CV={stability:.4f}")
        
        LOGGER.info("Frequency domain analysis:")
        for osc_name, analysis in frequency_analysis.items():
            if "dominant_frequency" in analysis:
                LOGGER.info(
                    f"  {osc_name}: range={analysis['range']:.4f}, "
                    f"freq={analysis['dominant_frequency']:.6f}"
                )
        
        # Final state check
        final_status = clock.get_system_status()
        metrics["final_state"] = {
            "alertness": final_status["biological_time"]["alertness"],
            "phase": final_status["biological_time"]["circadian_phase"],
            "oscillator_count": len(clock.biological_clock.oscillators),
        }
        
        LOGGER.info(
            f"OK Burn-in complete: {metrics['final_state']['oscillator_count']} "
            f"oscillators stabilized, phase={metrics['final_state']['phase']}"
        )
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=0,
            name="Clock Burn-in (Oscillator Stabilization)",
            success=success,
            duration=duration,
            details=f"Burn-in {burnin_duration}s, {metrics['final_state']['oscillator_count']} oscillators",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        ), clock
        
    except Exception as e:
        LOGGER.exception("Stage 0 failed with exception")
        return StageResult(
            stage_num=0,
            name="Clock Burn-in",
            success=False,
            duration=time.time() - start_time,
            details="Burn-in failed",
            errors=[str(e)],
        ), None


# ============================================================================
# Stage 1: Internal Clock Initialization
# ============================================================================

def stage_1_clock_verification(clock: TemporalCoherence) -> StageResult:
    """
    Stage 1: Verify clock operations with pre-stabilized oscillators.
    At this point, oscillators have completed multiple cycles.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 1: Clock Verification (Post Burn-in)")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        # Verify basic operations with warmed-up clock
        LOGGER.info("Verifying clock operations with stabilized oscillators...")
        
        # Test 1: TimePoint creation
        t1 = clock.now()
        metrics["timepoint_creation"] = True
        LOGGER.debug(f"  OK TimePoint created: {t1.system_time}")
        
        # Test 2: Clock update
        clock.update()
        t2 = clock.now()
        elapsed = t2.system_time - t1.system_time
        metrics["clock_update_elapsed"] = elapsed
        LOGGER.debug(f"  OK Clock updated, elapsed: {elapsed:.4f}s")
        
        # Test 3: Circadian phase
        phase = clock.get_circadian_phase()
        metrics["circadian_phase"] = phase.name
        LOGGER.debug(f"  OK Circadian phase: {phase.name}")
        
        # Test 4: Alertness level (should be stable after burn-in)
        alertness = clock.get_alertness()
        metrics["alertness"] = alertness
        LOGGER.debug(f"  OK Alertness level: {alertness:.4f}")
        
        # Test 5: Status check
        status = clock.get_system_status()
        metrics["status_keys"] = list(status.keys())
        LOGGER.debug(f"  OK System status retrieved ({len(status)} sections)")
        
        # Verify oscillators are running
        bio_clock = clock.biological_clock
        oscillator_count = len(bio_clock.oscillators)
        metrics["oscillator_count"] = oscillator_count
        
        if oscillator_count < 5:
            warnings.append(f"Only {oscillator_count} oscillators initialized")
        
        LOGGER.info(f"OK Clock verified with {oscillator_count} oscillators after burn-in")
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=1,
            name="Clock Verification (Post Burn-in)",
            success=success,
            duration=duration,
            details=f"Verified {oscillator_count} oscillators, phase={phase.name}",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        
    except Exception as e:
        LOGGER.exception("Stage 1 failed with exception")
        return StageResult(
            stage_num=1,
            name="Clock Verification",
            success=False,
            duration=time.time() - start_time,
            details="Verification failed",
            errors=[str(e)],
        )


# ============================================================================
# Stage 2: Clock Dynamics Verification
# ============================================================================

def stage_2_clock_dynamics(clock: TemporalCoherence) -> StageResult:
    """
    Stage 2: Verify clock dynamics over multiple update cycles.
    Ensure oscillators are properly evolving and entrainment works.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 2: Clock Dynamics Verification")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        # Record initial state
        initial_alertness = clock.get_alertness()
        initial_phase = clock.get_circadian_phase()
        
        LOGGER.info(f"Initial state: phase={initial_phase.name}, alertness={initial_alertness:.4f}")
        
        # Run multiple update cycles
        alertness_history = []
        phase_history = []
        
        LOGGER.info(f"Running {INTEGRATION_CYCLES} update cycles...")
        for cycle in range(INTEGRATION_CYCLES):
            time.sleep(0.5)  # Let some time pass
            clock.update()
            
            alertness = clock.get_alertness()
            phase = clock.get_circadian_phase()
            
            alertness_history.append(alertness)
            phase_history.append(phase.name)
            
            LOGGER.debug(f"  Cycle {cycle+1}: alertness={alertness:.4f}, phase={phase.name}")
        
        # Verify changes occurred (or didn't, depending on scale)
        alertness_range = max(alertness_history) - min(alertness_history)
        metrics["alertness_range"] = alertness_range
        metrics["alertness_history"] = alertness_history
        metrics["phase_history"] = phase_history
        
        LOGGER.info(f"Alertness range over cycles: {alertness_range:.4f}")
        
        # Test entrainment
        LOGGER.info("Testing entrainment with light signal...")
        clock.entrain_biological_clock("light", 1.0, strength=0.5)
        time.sleep(0.2)
        clock.update()
        
        post_entrainment_alertness = clock.get_alertness()
        metrics["post_entrainment_alertness"] = post_entrainment_alertness
        
        LOGGER.debug(f"  Post-entrainment alertness: {post_entrainment_alertness:.4f}")
        
        # Test attention shifting
        LOGGER.info("Testing attention state changes...")
        clock.set_attention(focus="test_object", level=0.9)
        time.sleep(0.2)
        clock.update()
        
        status = clock.get_system_status()
        attention_state = status["time_perception"]["attentional_state"]
        metrics["attention_state"] = attention_state
        
        LOGGER.debug(f"  Attention state: {attention_state}")
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=2,
            name="Clock Dynamics Verification",
            success=success,
            duration=duration,
            details=f"Completed {INTEGRATION_CYCLES} cycles, alertness_range={alertness_range:.4f}",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        
    except Exception as e:
        LOGGER.exception("Stage 2 failed with exception")
        return StageResult(
            stage_num=2,
            name="Clock Dynamics Verification",
            success=False,
            duration=time.time() - start_time,
            details="Dynamics verification failed",
            errors=[str(e)],
        )


# ============================================================================
# Stage 3: Temporal Eigenstate with Clock
# ============================================================================

def stage_3_eigenstate_integration(clock: TemporalCoherence) -> StageResult:
    """
    Stage 3: Integrate TemporalEigenstate with the internal clock.
    This is where frequency architectures begin to interact.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 3: Temporal Eigenstate Integration")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        LOGGER.info("Creating TemporalEigenstate with clock integration...")
        eigenstate = TemporalEigenstate(
            compression_factor=0.9,
            device=DEVICE,
            internal_clock=clock,
        )
        
        # Let the eigenstate stabilize with the clock
        LOGGER.info(f"Allowing {OSCILLATOR_SETTLE_TIME}s for eigenstate-clock coupling...")
        time.sleep(OSCILLATOR_SETTLE_TIME)
        
        # Run dilation sequence synchronized with clock updates
        dilations = []
        regimes = []
        
        LOGGER.info(f"Running {INTEGRATION_CYCLES} synchronized dilation cycles...")
        for cycle in range(INTEGRATION_CYCLES):
            clock.update()
            
            params = {
                "complexity": 0.3 + (cycle * 0.1),
                "emotional_charge": 0.1 * ((-1) ** cycle),
            }
            
            dilation = eigenstate.dilate(params)
            dilations.append(dilation)
            
            metrics_snapshot = eigenstate.get_metrics()
            regimes.append(metrics_snapshot["temporal_regime"])
            
            LOGGER.debug(
                f"  Cycle {cycle+1}: dilation={dilation:.4f}, "
                f"regime={metrics_snapshot['temporal_regime']}"
            )
            
            time.sleep(0.3)  # Allow time for dynamics
        
        final_metrics = eigenstate.get_metrics()
        
        metrics["dilations"] = dilations
        metrics["regimes"] = regimes
        metrics["final_cumulative"] = final_metrics["cumulative_dilation"]
        metrics["final_regime"] = final_metrics["temporal_regime"]
        metrics["recursive_depth"] = final_metrics["recursive_depth"]
        
        LOGGER.info(
            f"OK Completed integration: depth={final_metrics['recursive_depth']}, "
            f"regime={final_metrics['temporal_regime']}"
        )
        
        # Verify clock state after eigenstate operations
        clock_status = clock.get_system_status()
        metrics["clock_alertness_post"] = clock_status["biological_time"]["alertness"]
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=3,
            name="Temporal Eigenstate Integration",
            success=success,
            duration=duration,
            details=f"Depth={final_metrics['recursive_depth']}, regime={final_metrics['temporal_regime']}",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        
    except Exception as e:
        LOGGER.exception("Stage 3 failed with exception")
        return StageResult(
            stage_num=3,
            name="Temporal Eigenstate Integration",
            success=False,
            duration=time.time() - start_time,
            details="Integration failed",
            errors=[str(e)],
        )


# ============================================================================
# Stage 4: Recursive Stabilization with Clock
# ============================================================================

def stage_4_stabilization_integration(clock: TemporalCoherence) -> StageResult:
    """
    Stage 4: Test RecursiveStabilizationPoint with clock synchronization.
    This tests deeper frequency coupling and convergence dynamics.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 4: Recursive Stabilization Integration")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        LOGGER.info("Creating RecursiveStabilizationPoint with clock...")
        stabilizer = RecursiveStabilizationPoint(
            dimension=32,
            device=DEVICE,
            max_recursion_depth=128,
            internal_clock=clock,
        )
        
        time.sleep(OSCILLATOR_SETTLE_TIME)
        
        # Create initial state modulated by clock alertness
        clock.update()
        alertness = clock.get_alertness()
        
        LOGGER.info(f"Initial clock alertness: {alertness:.4f}")
        
        # Modulate initial state by biological rhythm
        initial_state = torch.randn(1, 32, device=DEVICE) * alertness
        
        LOGGER.info("Running stabilization with clock synchronization...")
        stabilized_state, stab_metrics = stabilizer.stabilize(
            initial_state,
            max_iterations=64,
            convergence_threshold=1e-4,
        )
        
        metrics["converged"] = stab_metrics.get("convergence_detected", False)
        metrics["iterations"] = stab_metrics.get("iterations", 0)
        metrics["final_regime"] = stab_metrics.get("final_regime", "Unknown")
        metrics["initial_alertness"] = alertness
        
        # Check clock state post-stabilization
        clock.update()
        post_alertness = clock.get_alertness()
        metrics["post_alertness"] = post_alertness
        metrics["alertness_delta"] = post_alertness - alertness
        
        LOGGER.info(
            f"OK Stabilization complete: converged={metrics['converged']}, "
            f"iterations={metrics['iterations']}, "
            f"alertness_delta={metrics['alertness_delta']:.4f}"
        )
        
        if not metrics["converged"]:
            warnings.append("Stabilization did not converge within iteration limit")
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=4,
            name="Recursive Stabilization Integration",
            success=success,
            duration=duration,
            details=f"Converged={metrics['converged']}, iterations={metrics['iterations']}",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        
    except Exception as e:
        LOGGER.exception("Stage 4 failed with exception")
        return StageResult(
            stage_num=4,
            name="Recursive Stabilization Integration",
            success=False,
            duration=time.time() - start_time,
            details="Stabilization integration failed",
            errors=[str(e)],
        )


# ============================================================================
# Stage 5: Full System Integration
# ============================================================================

def stage_5_full_integration(clock: TemporalCoherence) -> StageResult:
    """
    Stage 5: Full integration test with TemporalEigenstateNode.
    All systems operating together with frequency synchronization.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE 5: Full System Integration")
    LOGGER.info("=" * 70)
    
    start_time = time.time()
    errors = []
    warnings = []
    metrics = {}
    
    try:
        LOGGER.info("Creating TemporalEigenstateNode with full clock integration...")
        node = TemporalEigenstateNode(
            latent_dim=64,
            sacred_init=False,
            device=DEVICE,
            internal_clock=clock,
        )
        
        time.sleep(OSCILLATOR_SETTLE_TIME)
        
        # Run multiple forward passes synchronized with clock
        forward_outputs = []
        clock_states = []
        
        LOGGER.info(f"Running {INTEGRATION_CYCLES} synchronized forward passes...")
        for cycle in range(INTEGRATION_CYCLES):
            clock.update()
            
            # Get clock state
            clock_status = clock.get_system_status()
            alertness = clock_status["biological_time"]["alertness"]
            phase = clock_status["biological_time"]["circadian_phase"]
            
            clock_states.append({
                "alertness": alertness,
                "phase": phase,
            })
            
            # Create input modulated by clock state
            input_state = torch.randn(1, 64, device=DEVICE) * alertness
            
            # Forward pass
            output = node.forward(
                x=input_state,
                time_external=torch.tensor(cycle * 0.5, device=DEVICE),
                depth=cycle + 1,
            )
            
            forward_outputs.append({
                "regime": output["regime"],
                "t_internal": output["t_internal"].item(),
                "stability": output["stability"].item(),
                "echo_collapsed": output["echo_collapsed"],
            })
            
            LOGGER.debug(
                f"  Cycle {cycle+1}: regime={output['regime']}, "
                f"alertness={alertness:.4f}, phase={phase}"
            )
            
            time.sleep(0.3)
        
        # Get final reflection
        reflection = node.reflect()
        
        metrics["forward_outputs"] = forward_outputs
        metrics["clock_states"] = clock_states
        metrics["final_reflection"] = reflection
        
        # Analyze synchronization
        regime_changes = sum(
            1 for i in range(1, len(forward_outputs))
            if forward_outputs[i]["regime"] != forward_outputs[i-1]["regime"]
        )
        
        metrics["regime_changes"] = regime_changes
        
        LOGGER.info(
            f"OK Full integration complete: {regime_changes} regime changes across "
            f"{INTEGRATION_CYCLES} cycles"
        )
        
        success = len(errors) == 0
        duration = time.time() - start_time
        
        return StageResult(
            stage_num=5,
            name="Full System Integration",
            success=success,
            duration=duration,
            details=f"Completed {INTEGRATION_CYCLES} cycles, {regime_changes} regime changes",
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        
    except Exception as e:
        LOGGER.exception("Stage 5 failed with exception")
        return StageResult(
            stage_num=5,
            name="Full System Integration",
            success=False,
            duration=time.time() - start_time,
            details="Full integration failed",
            errors=[str(e)],
        )


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all staged integration tests."""
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("STAGED INTERNAL CLOCK INTEGRATION TEST")
    LOGGER.info("=" * 70 + "\n")
    
    overall_start = time.time()
    results = []
    
    # Stage 0: Clock burn-in (NEW - critical for frequency architectures)
    result_0, clock = stage_0_clock_burnin(burnin_duration=BURNIN_DURATION)
    results.append(result_0)
    
    if not result_0.success or clock is None:
        LOGGER.error("Stage 0 (burn-in) failed. Aborting subsequent stages.")
        write_report(results, time.time() - overall_start)
        return
    
    LOGGER.info(f"\nWaiting {STAGE_WARMUP_TIME}s before Stage 1...\n")
    time.sleep(STAGE_WARMUP_TIME)
    
    # Stage 1: Clock verification (now with pre-warmed oscillators)
    result_1 = stage_1_clock_verification(clock)
    results.append(result_1)
    
    if not result_1.success:
        LOGGER.error("Stage 1 failed. Aborting subsequent stages.")
        write_report(results, time.time() - overall_start)
        return
    
    # Warmup between stages
    LOGGER.info(f"\nWaiting {STAGE_WARMUP_TIME}s before Stage 2...\n")
    time.sleep(STAGE_WARMUP_TIME)
    
    # Stage 2: Clock dynamics
    result_2 = stage_2_clock_dynamics(clock)
    results.append(result_2)
    
    if not result_2.success:
        LOGGER.error("Stage 2 failed. Aborting subsequent stages.")
        write_report(results, time.time() - overall_start)
        return
    
    LOGGER.info(f"\nWaiting {STAGE_WARMUP_TIME}s before Stage 3...\n")
    time.sleep(STAGE_WARMUP_TIME)
    
    # Stage 3: Eigenstate integration
    result_3 = stage_3_eigenstate_integration(clock)
    results.append(result_3)
    
    LOGGER.info(f"\nWaiting {STAGE_WARMUP_TIME}s before Stage 4...\n")
    time.sleep(STAGE_WARMUP_TIME)
    
    # Stage 4: Stabilization integration
    result_4 = stage_4_stabilization_integration(clock)
    results.append(result_4)
    
    LOGGER.info(f"\nWaiting {STAGE_WARMUP_TIME}s before Stage 5...\n")
    time.sleep(STAGE_WARMUP_TIME)
    
    # Stage 5: Full integration
    result_5 = stage_5_full_integration(clock)
    results.append(result_5)
    
    # Final report
    total_duration = time.time() - overall_start
    write_report(results, total_duration)
    
    # Print summary
    print("\n" + "=" * 70)
    print("STAGED INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    for result in results:
        status = "OK PASS" if result.success else "✗ FAIL"
        print(f"{status} | Stage {result.stage_num}: {result.name}")
        print(f"       Duration: {result.duration:.2f}s")
        print(f"       {result.details}")
        if result.warnings:
            print(f"       Warnings: {len(result.warnings)}")
        if result.errors:
            print(f"       Errors: {len(result.errors)}")
        print()
    
    passed = sum(1 for r in results if r.success)
    print(f"Total: {passed}/{len(results)} stages passed in {total_duration:.2f}s")
    print("=" * 70)


def write_report(results: List[StageResult], total_duration: float):
    """Write JSON and markdown reports."""
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration": total_duration,
        "device": DEVICE,
        "config": {
            "stage_warmup_time": STAGE_WARMUP_TIME,
            "oscillator_settle_time": OSCILLATOR_SETTLE_TIME,
            "integration_cycles": INTEGRATION_CYCLES,
        },
        "stages": [
            {
                "stage_num": r.stage_num,
                "name": r.name,
                "success": r.success,
                "duration": r.duration,
                "details": r.details,
                "metrics": r.metrics,
                "warnings": r.warnings,
                "errors": r.errors,
            }
            for r in results
        ],
    }
    
    # Write JSON
    json_path = REPORTS_DIR / "staged_clock_integration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Write Markdown
    md_path = REPORTS_DIR / "staged_clock_integration.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Staged Internal Clock Integration Test Report\n\n")
        f.write(f"**Timestamp:** {report_data['timestamp']}\n")
        f.write(f"**Total Duration:** {total_duration:.2f}s\n")
        f.write(f"**Device:** {DEVICE}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Stage warmup time: {STAGE_WARMUP_TIME}s\n")
        f.write(f"- Oscillator settle time: {OSCILLATOR_SETTLE_TIME}s\n")
        f.write(f"- Integration cycles: {INTEGRATION_CYCLES}\n\n")
        
        f.write("## Results\n\n")
        for r in results:
            status = "OK PASS" if r.success else "✗ FAIL"
            f.write(f"### Stage {r.stage_num}: {r.name} {status}\n\n")
            f.write(f"- **Duration:** {r.duration:.2f}s\n")
            f.write(f"- **Details:** {r.details}\n")
            
            if r.warnings:
                f.write(f"- **Warnings:** {len(r.warnings)}\n")
                for w in r.warnings:
                    f.write(f"  - {w}\n")
            
            if r.errors:
                f.write(f"- **Errors:** {len(r.errors)}\n")
                for e in r.errors:
                    f.write(f"  - {e}\n")
            
            f.write("\n")
    
    LOGGER.info(f"Reports written to:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
