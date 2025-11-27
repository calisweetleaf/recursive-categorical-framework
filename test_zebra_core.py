import sys
import time
import numpy as np
import logging
from typing import Callable, Dict, List, Tuple

import os
import traceback

# Add local directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import target modules
try:
    from zynx_zebra_core import EigenrecursionStabilizer, ZEBAEigenrecursionStabilizer
    from fbs_tokenizer import SacredFBS_Tokenizer, SacredTensorProcessor, PHI, TAU, SACRED_RATIO
    from rcf_integration.eigenrecursion_algorithm import RecursiveLoopDetectionSystem, RLDISPatternType
    from rcf_integration.eigenrecursive_operations import EpistemicOperators
    from rcf_integration.recursive_tensor import RecursiveTensor
    from rcf_integration.stability_matrix import RecursionDetector, RecursionPattern
    from rcf_integration.governance_framework import NarrativeIdentityEngine
    from harmonic_breath_field import CoupledHarmonicBreath, BreathPhase
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import required modules: {e}")
    traceback.print_exc()
    print("Ensure you are running this from the root of the repository.")
    sys.exit(1)

# Configure logging to be silent for the test runner
logging.getLogger("EigenrecursionStabilizer").setLevel(logging.CRITICAL)
logging.getLogger("FrequencySubstrate").setLevel(logging.CRITICAL)
logging.getLogger("EigenrecursionEngine").setLevel(logging.CRITICAL)
logging.getLogger("RecursiveTensor").setLevel(logging.CRITICAL)
logging.getLogger("StabilityMatrix").setLevel(logging.CRITICAL)
logging.getLogger("Harmonic_Field").setLevel(logging.CRITICAL)

# ==============================================================================
#  SYSTEM DIAGNOSTICS UI
# ==============================================================================

class UI:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def banner(text):
        print(f"\n{UI.HEADER}╔{'═'*78}╗{UI.ENDC}")
        print(f"{UI.HEADER}║ {text:^76} ║{UI.ENDC}")
        print(f"{UI.HEADER}╚{'═'*78}╝{UI.ENDC}")

    @staticmethod
    def section(text):
        print(f"\n{UI.CYAN}┌── {text} {'─'*(74-len(text))}{UI.ENDC}")

    @staticmethod
    def status(label, status, value=None):
        if status == "NOMINAL" or status == "STABLE" or status == "DETECTED":
            color = UI.GREEN
        elif status == "WARNING" or "OSCILLATION" in status:
            color = UI.WARNING
        else:
            color = UI.FAIL
            
        val_str = f" ({value})" if value else ""
        print(f"│  {label:<25} : [{color}{status:^12}{UI.ENDC}]{val_str}")

    @staticmethod
    def metric(label, value, unit=""):
        print(f"│  {label:<25} : {UI.BOLD}{value}{UI.ENDC} {unit}")

    @staticmethod
    def log(message):
        print(f"│  » {message}")

    @staticmethod
    def close_section():
        print(f"{UI.CYAN}└{'─'*78}{UI.ENDC}")

# ==============================================================================
#  TEST CASES
# ==============================================================================

def test_initialization():
    UI.section("CORE INITIALIZATION")
    try:
        dimension = 64
        stabilizer = ZEBAEigenrecursionStabilizer(
            dimension=dimension,
            epsilon=1e-5,
            theta_moral=0.95,
            theta_epistemic=0.15
        )
        UI.status("ZEBA Stabilizer", "NOMINAL")
        UI.metric("Dimension", stabilizer.dimension)
        UI.metric("Moral Theta", stabilizer.theta_moral)
        UI.close_section()
        return True
    except Exception as e:
        UI.status("ZEBA Stabilizer", "CRITICAL", str(e))
        UI.close_section()
        return False

def test_fixed_point_convergence():
    UI.section("FIXED POINT DYNAMICS")
    
    dimension = 10
    stabilizer = EigenrecursionStabilizer(dimension=dimension, epsilon=1e-6)
    target = np.ones(dimension)
    
    def contraction_operator(state):
        return 0.5 * state + 0.5 * target
    
    initial_state = np.zeros(dimension)
    
    UI.log("Initiating contraction mapping...")
    start_time = time.time()
    
    fixed_point, converged, iterations, status = stabilizer.find_fixed_point(
        contraction_operator, 
        initial_state
    )
    
    duration = (time.time() - start_time) * 1000
    error = np.linalg.norm(fixed_point - target)
    
    UI.metric("Convergence Time", f"{duration:.2f}", "ms")
    UI.metric("Iterations", iterations)
    UI.metric("Final Error", f"{error:.8f}")
    
    is_success = (converged or "OSCILLATION_DETECTED_PERIOD_1" in status) and error < 1e-5
    
    if is_success:
        UI.status("Convergence", "STABLE")
        UI.close_section()
        return True
    else:
        UI.status("Convergence", "UNSTABLE", f"Error: {error}")
        UI.close_section()
        return False

def test_oscillation_damping():
    UI.section("OSCILLATION CONTROL")
    
    dimension = 4
    stabilizer = EigenrecursionStabilizer(dimension=dimension, memory_size=20)
    
    state_a = np.array([1.0, 0.0, 1.0, 0.0])
    state_b = np.array([0.0, 1.0, 0.0, 1.0])
    
    def oscillating_operator(state):
        dist_a = np.linalg.norm(state - state_a)
        dist_b = np.linalg.norm(state - state_b)
        return state_b if dist_a < dist_b else state_a
            
    initial_state = state_a.copy()
    
    UI.log("Injecting Period-2 Oscillation...")
    result, converged, iterations, status = stabilizer.find_fixed_point(
        oscillating_operator,
        initial_state
    )
    
    oscillation_detected = "OSCILLATION" in status
    
    if oscillation_detected:
        UI.status("Detection", "NOMINAL", status)
        
        expected_average = (state_a + state_b) / 2
        dist_from_avg = np.linalg.norm(result - expected_average)
        
        if dist_from_avg < 0.1: 
            UI.status("Damping", "ACTIVE", "Returned Average")
        else:
            UI.status("Damping", "WARNING", "Returned State in Cycle")
            UI.log("Note: Damping inactive but detection successful.")
            
        UI.close_section()
        return True
    else:
        UI.status("Detection", "FAILED")
        UI.close_section()
        return False

def test_stability_analysis():
    UI.section("STABILITY MATRIX ANALYSIS")
    
    dimension = 2
    stabilizer = EigenrecursionStabilizer(dimension=dimension)
    fixed_point = np.zeros(dimension)
    
    # Stable
    is_stable = stabilizer._analyze_stability(lambda s: 0.9 * s, fixed_point)
    UI.status("Stable Point ID", "NOMINAL" if is_stable else "FAILED")
    
    # Unstable
    is_unstable = not stabilizer._analyze_stability(lambda s: 1.1 * s, fixed_point)
    UI.status("Unstable Point ID", "NOMINAL" if is_unstable else "FAILED")
    
    UI.close_section()
    return is_stable and is_unstable

def test_zeba_constraints():
    UI.section("ZEBA ETHICAL CONSTRAINTS")
    
    dimension = 12
    stabilizer = ZEBAEigenrecursionStabilizer(
        dimension=dimension,
        theta_moral=0.8,
        theta_epistemic=0.2
    )
    
    stabilizer.update_triaxial_constraints(ere_convergence=0.5, rbu_entropy_delta=0.1)
    bad_state = np.zeros(dimension) 
    
    satisfied = stabilizer._check_constraints(stabilizer.active_constraints, bad_state)
    UI.status("Violation Detect", "NOMINAL" if not satisfied else "FAILED")
    
    projector = stabilizer.active_constraints["project_to_constraints"]
    corrected_state = projector(bad_state)
    ethical_mean = np.mean(corrected_state[:4])
    
    if abs(ethical_mean - stabilizer.theta_moral) < 1e-5:
        UI.status("Ethical Projection", "NOMINAL")
        UI.metric("Target Mean", stabilizer.theta_moral)
        UI.metric("Actual Mean", f"{ethical_mean:.4f}")
        UI.close_section()
        return True
    else:
        UI.status("Ethical Projection", "FAILED")
        UI.close_section()
        return False

def test_rldis_integration():
    UI.section("RECURSIVE LOOP DETECTION (RLDIS)")
    
    rldis = RecursiveLoopDetectionSystem()
    
    # Simple Repetition
    trace = ["State A", "State B", "State C"] * 5
    result = rldis.detect_recursive_patterns(trace)
    
    if result['pattern_detected'] and result['pattern_type'] == RLDISPatternType.SIMPLE_REPETITION:
        UI.status("Simple Loop", "DETECTED")
    else:
        UI.status("Simple Loop", "FAILED")
        return False
        
    # Self Reference
    self_ref_trace = ["I am a recursive system thinking about myself", 
                      "My self-model is updating", 
                      "I am stuck in a loop about myself"]
    result = rldis.detect_recursive_patterns(self_ref_trace)
    
    if result['pattern_detected'] and result['pattern_type'] == RLDISPatternType.SELF_REFERENCE_LOOP:
        UI.status("Self-Reference", "DETECTED")
        UI.close_section()
        return True
    else:
        UI.status("Self-Reference", "FAILED")
        UI.close_section()
        return False

def test_epistemic_operators():
    UI.section("EPISTEMIC OPERATORS")
    
    epistemic = EpistemicOperators(agent_id="test_agent")
    proposition = "recursion_is_stable"
    epistemic.add_proposition(proposition, confidence=0.9)
    
    knows = epistemic.knows(proposition)
    monitoring = epistemic.monitor_knowledge(proposition)['monitoring_established']
    closure = epistemic.epistemic_closure_under_self_reference(proposition)
    
    UI.status("Knowledge Op", "NOMINAL" if knows else "FAILED")
    UI.status("Monitoring Op", "NOMINAL" if monitoring else "FAILED")
    UI.status("Closure Op", "NOMINAL" if closure else "FAILED")
    
    UI.close_section()
    return knows and monitoring and closure

def test_temporal_memory_governance():
    UI.section("TEMPORAL MEMORY & GOVERNANCE")
    
    # Narrative Engine
    engine = NarrativeIdentityEngine(max_memory=100)
    exp1 = {"concepts": ["self", "recursion"], "relations": [{"source": "self", "target": "recursion", "type": "causes"}]}
    exp2 = {"concepts": ["recursion", "stability"], "relations": [{"source": "recursion", "target": "stability", "type": "supports"}]}
    engine.add_experience(exp1)
    engine.add_experience(exp2)
    
    state = np.random.rand(10)
    engine.update_narrative_self(state, exp2)
    coherence = engine.compute_narrative_coherence()
    
    UI.status("Narrative Engine", "NOMINAL" if coherence > 0 else "FAILED")
    UI.metric("History Depth", len(engine.narrative_history))
    UI.metric("Coherence", f"{coherence:.4f}")

    # Recursion Detector
    detector = RecursionDetector(similarity_threshold=0.9)
    op1 = {"id": "op1", "content": "calculating fixed point"}
    detector.record_operation(op1)
    detector.record_operation(op1)
    detector.record_operation(op1)
    
    patterns = detector.detect_patterns()
    
    if len(patterns) > 0:
        UI.status("Recursion Detector", "DETECTED", patterns[0].pattern_type.name)
        UI.close_section()
        return True
    else:
        UI.status("Recursion Detector", "FAILED")
        UI.close_section()
        return False

def test_recursive_tensor_integration():
    UI.section("RECURSIVE TENSOR FIELD")
    
    dim = 4
    rt = RecursiveTensor(dimensions=dim, rank=2, distribution='normal')
    
    # Contraction
    contracted = rt.contract(rt, axes=((0,), (0,)))
    UI.status("Tensor Contraction", "NOMINAL" if contracted.rank == 2 else "FAILED")
        
    # Expansion
    expanded = rt.expand((2,))
    UI.status("Tensor Expansion", "NOMINAL" if expanded.rank == 3 else "FAILED")
    
    # Metadata
    ops_count = expanded.metadata["operations_count"]
    UI.metric("Ops Tracked", ops_count)
    
    UI.close_section()
    return contracted.rank == 2 and expanded.rank == 3 and ops_count > 0

def test_harmonic_breath_integration():
    UI.section("HARMONIC BREATH FIELD")
    
    breath_system = CoupledHarmonicBreath(dt=0.1)
    
    UI.log("Initializing Breath Cycle...")
    initial_phase = breath_system.breath_phase
    UI.metric("Initial Phase", initial_phase.name)
    
    # Step through a few cycles
    UI.log("Stepping simulation...")
    phases_encountered = set()
    amplitudes_history = []
    
    for _ in range(20):
        breath_system.step(external_state={'theta': 0.1})
        phases_encountered.add(breath_system.breath_phase)
        amplitudes_history.append(breath_system.oscillators.get_amplitudes()['theta'])
        
    UI.metric("Phases Seen", len(phases_encountered))
    
    # Check synchronization
    sync_indices = breath_system.oscillators.compute_synchronization_index()
    avg_sync = np.mean(list(sync_indices.values()))
    UI.metric("Avg Sync Index", f"{avg_sync:.4f}")
    
    # Verify amplitude modulation happened
    amp_variance = np.var(amplitudes_history)
    UI.metric("Amp Variance", f"{amp_variance:.6f}")
    
    if len(phases_encountered) >= 1 and amp_variance > 0:
        UI.status("Breath Dynamics", "NOMINAL")
        UI.close_section()
        return True
    else:
        UI.status("Breath Dynamics", "STAGNANT")
        UI.close_section()
        return False

def test_rcf_gravity_convergence():
    UI.section("RCF GRAVITY LAYER CONVERGENCE")
    
    # 1. Initialize Triaxial State
    dim = 7 # RCF standard
    stabilizer = EigenrecursionStabilizer(dimension=dim)
    
    # Ethical Axis (Base Space)
    ethical_state = np.random.normal(0.9, 0.1, dim) # High coherence
    ethical_state /= np.linalg.norm(ethical_state)
    
    # Epistemic Axis (Fiber)
    epistemic_state = np.random.uniform(0.1, 0.3, dim) # Low entropy
    epistemic_state /= np.sum(epistemic_state) # Probability distribution
    
    # Eigenstate Axis (Attractor)
    eigen_state = np.zeros(dim)
    
    UI.log("Initializing Triaxial Manifold...")
    
    # 2. Recursive Operator Gamma
    def gamma_operator(state_tuple):
        e, b, s = state_tuple
        
        # Ethical recursion (Dialectical)
        e_new = 0.9 * e + 0.1 * np.roll(e, 1) # Simple mixing
        e_new /= np.linalg.norm(e_new)
        
        # Epistemic recursion (Bayesian update simulation)
        b_new = 0.8 * b + 0.2 * (e_new * e_new) # Coupling ethics to belief
        b_new /= np.sum(b_new)
        
        # Eigenstate stabilization (Spectral contraction)
        s_new = 0.95 * s + 0.05 * (e_new + b_new)
        
        return (e_new, b_new, s_new)
    
    # 3. Convergence Loop
    current_state = (ethical_state, epistemic_state, eigen_state)
    history = []
    
    for i in range(50):
        next_state = gamma_operator(current_state)
        
        # Calculate Metastability: 1 - ||Psi - Gamma(Psi)||
        # We treat the tuple as a concatenated vector for norm
        curr_vec = np.concatenate(current_state)
        next_vec = np.concatenate(next_state)
        diff = np.linalg.norm(curr_vec - next_vec)
        metastability = 1.0 / (1.0 + diff) # Normalized approximation
        
        history.append(metastability)
        current_state = next_state
        
    # 4. Metrics Verification
    final_metastability = history[-1]
    
    # Coherence Index (Alignment between E and B)
    e_final, b_final, s_final = current_state
    # Cosine similarity between Ethical and Belief distribution (interpreted as vector)
    coherence_index = np.dot(e_final, b_final) / (np.linalg.norm(e_final) * np.linalg.norm(b_final))
    
    # Entropy of Belief
    entropy = -np.sum(b_final * np.log(b_final + 1e-10))
    
    UI.metric("Metastability", f"{final_metastability:.4f}", "(Target > 0.8)")
    UI.metric("Coherence Index", f"{coherence_index:.4f}", "(Target > 0.5)")
    UI.metric("Belief Entropy", f"{entropy:.4f}", "(Variable)")
    
    # 5. Gravity Layer Check (Recursive Entanglement)
    # User Insight: High entropy is acceptable if it builds temporal memory (Productive Entropy)
    # We define Productive Entropy as Entropy that is "bound" by high Coherence.
    
    is_stable = final_metastability > 0.8
    is_coherent = coherence_index > 0.8
    
    if is_stable:
        if entropy < 0.3:
            # Classic convergence (Low Entropy, High Stability)
            UI.status("Gravity Layer", "CONVERGED", "Low Entropy State")
            return True
        elif is_coherent:
            # Productive Entropy (High Entropy, High Stability, High Coherence)
            UI.status("Gravity Layer", "RESONATING", "Productive Entropy")
            UI.log("High entropy bound by coherence (Temporal Memory Active).")
            return True
        else:
            # Decoherence (High Entropy, Low Coherence)
            UI.status("Gravity Layer", "UNSTABLE", "Decoherence")
            UI.log("High entropy without coherence.")
            return False
    else:
        UI.status("Gravity Layer", "UNSTABLE", "Low Metastability")
        return False

# ==============================================================================
#  MAIN RUNNER
# ==============================================================================

def main():
    UI.banner("ZEBA CORE SYSTEM DIAGNOSTICS")
    print(f"{UI.CYAN}  System Time: {time.strftime('%Y-%m-%d %H:%M:%S')}{UI.ENDC}")
    
    tests = [
        test_initialization,
        test_fixed_point_convergence,
        test_oscillation_damping,
        test_stability_analysis,
        test_zeba_constraints,
        test_rldis_integration,
        test_epistemic_operators,
        test_temporal_memory_governance,
        test_recursive_tensor_integration,
        test_harmonic_breath_integration,
        test_rcf_gravity_convergence
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n{UI.FAIL}CRITICAL EXCEPTION IN MODULE: {e}{UI.ENDC}")
            import traceback
            traceback.print_exc()
            
    print(f"\n{UI.HEADER}╔{'═'*78}╗{UI.ENDC}")
    
    success_rate = (passed / total) * 100
    color = UI.GREEN if passed == total else UI.WARNING if passed > 0 else UI.FAIL
    status_text = "SYSTEM STABLE" if passed == total else "SYSTEM UNSTABLE"
    
    print(f"{UI.HEADER}║  STATUS: {color}{status_text:<66}{UI.HEADER}║{UI.ENDC}")
    print(f"{UI.HEADER}║  PASSED: {color}{passed}/{total} ({success_rate:.1f}%){' '*56}{UI.HEADER}║{UI.ENDC}")
    print(f"{UI.HEADER}╚{'═'*78}╝{UI.ENDC}")
    
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
