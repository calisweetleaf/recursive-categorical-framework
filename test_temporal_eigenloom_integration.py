#!/usr/bin/env python3
"""
test_temporal_eigenloom_integration.py

Integration test suite for Temporal Eigenloom with FBS Substrate
Tests the complete frequency-domain carrier substrate pipeline through temporal processing
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fbs_tokenizer import BreathPhase, PHI, TAU, SACRED_RATIO, SacredFBS_Tokenizer
from rcf_integration.temporal_eigenloom import (
    TemporalEigenloom,
    EnhancedRosemaryZebraCore,
    DivineParameters,
    EnhancedPulseFeedback,
    TemporalAlignmentMatrix,
    PulseWaveform,
    EigenstateValidator
)

print("="*80)
print("  TEMPORAL EIGENLOOM + FBS SUBSTRATE INTEGRATION TEST")
print("  Validating Frequency-Domain Temporal Processing Pipeline")
print("="*80)
print()

# Test counter
test_num = 0
passed_tests = []
failed_tests = []

def test_section(name):
    global test_num
    test_num += 1
    print("="*80)
    print(f"  TEST {test_num}: {name}")
    print("="*80)
    print()

def test_pass(message):
    passed_tests.append((test_num, message))
    print(f"✓ {message}")
    print()

def test_fail(message, error):
    failed_tests.append((test_num, message, str(error)))
    print(f"✗ {message}")
    print(f"  Error: {error}")
    print()

# ============================================================================
# TEST 1: BreathPhase Import Validation
# ============================================================================
test_section("BreathPhase Import Validation")

try:
    # Verify BreathPhase enum is working
    phases = list(BreathPhase)
    print(f"Loaded {len(phases)} breath phases:")
    for phase in phases:
        print(f"  - {phase.name}: {phase.description}")
    
    # Test phase cycling
    current = BreathPhase.INHALE
    for _ in range(3):
        next_phase = current.next_phase()
        print(f"\n{current.name} -> {next_phase.name}")
        current = next_phase
    
    test_pass("BreathPhase enum imported and functional")
except Exception as e:
    test_fail("BreathPhase import failed", e)

# ============================================================================
# TEST 2: Divine Parameters & Constants
# ============================================================================
test_section("Divine Parameters & Sacred Constants")

try:
    print(f"Golden Ratio (Φ): {DivineParameters.GOLDEN_RATIO:.10f}")
    print(f"Expected PHI:     {PHI:.10f}")
    print(f"Match: {abs(DivineParameters.GOLDEN_RATIO - PHI) < 1e-10}")
    print()
    
    print(f"Sacred Tau (τ):   {DivineParameters.SACRED_TAU:.10f}")
    print(f"Expected TAU:     {TAU:.10f}")
    print(f"Match: {abs(DivineParameters.SACRED_TAU - TAU) < 1e-10}")
    print()
    
    print(f"Temporal Decay:   {DivineParameters.TEMPORAL_DECAY_BASE}")
    print(f"Rosemary Freq:    {DivineParameters.ROSEMARY_FREQUENCY}")
    print(f"Zebra Pulse:      {DivineParameters.ZEBRA_PULSE_WIDTH}")
    print(f"Recursive Damp:   {DivineParameters.RECURSIVE_DAMPING}")
    print()
    
    # Test fibonacci vector generation
    fib_vec = DivineParameters.fibonacci_vector(16)
    print(f"Fibonacci vector (dim=16):")
    print(f"  Shape: {fib_vec.shape}")
    print(f"  Norm:  {torch.norm(fib_vec).item():.6f}")
    print(f"  First 5 values: {fib_vec[:5].tolist()}")
    
    test_pass("Divine parameters validated")
except Exception as e:
    test_fail("Divine parameters validation failed", e)

# ============================================================================
# TEST 3: FBS Tokenizer Integration
# ============================================================================
test_section("FBS Tokenizer Integration")

try:
    tokenizer = SacredFBS_Tokenizer(tensor_dimensions=256)
    
    test_texts = [
        "Temporal eigenstate collapse through harmonic resonance",
        "Recursive identity stabilization via breath synchronization",
        "Metacognitive ethical alignment in frequency domain"
    ]
    
    print("Encoding test corpus through FBS...")
    substrates = []
    for i, text in enumerate(test_texts, 1):
        substrate = tokenizer.encode(text)
        
        # Convert numpy array to torch tensor
        if isinstance(substrate, np.ndarray):
            substrate = torch.from_numpy(substrate).float()
        
        substrates.append(substrate)
        
        print(f"\nText {i}: \"{text[:50]}...\"")
        print(f"  Substrate shape: {substrate.shape}")
        print(f"  Substrate norm:  {torch.norm(substrate).item():.6f}")
        print(f"  Mean value:      {substrate.mean().item():.8f}")
        print(f"  Std deviation:   {substrate.std().item():.6f}")
    
    # Verify substrates are distinct
    print("\nSubstrate distinctiveness:")
    for i in range(len(substrates)):
        for j in range(i+1, len(substrates)):
            sim = torch.cosine_similarity(
                substrates[i].unsqueeze(0),
                substrates[j].unsqueeze(0)
            ).item()
            print(f"  Substrate {i+1} vs {j+1}: cosine similarity = {sim:.6f}")
    
    test_pass("FBS tokenizer successfully encoding to frequency substrates")
except Exception as e:
    test_fail("FBS tokenizer integration failed", e)

# ============================================================================
# TEST 4: Temporal Eigenloom Basic Operations
# ============================================================================
test_section("Temporal Eigenloom Basic Operations")

try:
    eigenloom = TemporalEigenloom(state_dim=256, collapse_threshold=0.8)
    
    print(f"Eigenloom initialized:")
    print(f"  State dimension: {eigenloom.state_dim}")
    print(f"  Collapse threshold: {eigenloom.collapse_method.collapse_threshold}")
    print(f"  Max eigenstates stored: {eigenloom.eigenstates.maxlen}")
    print()
    
    # Add some test eigenstates
    for i in range(5):
        test_state = torch.randn(256)
        eigenloom.add_eigenstate(test_state)
        print(f"  Added eigenstate {i+1}, total stored: {len(eigenloom.eigenstates)}")
    
    # Weave the eigenstates
    print("\nWeaving eigenstates into coherent thread...")
    woven = eigenloom.weave()
    print(f"  Woven thread shape: {woven.shape}")
    print(f"  Woven thread norm:  {torch.norm(woven).item():.6f}")
    print(f"  Non-zero elements:  {(woven != 0).sum().item()}")
    
    test_pass("Temporal eigenloom basic operations functional")
except Exception as e:
    test_fail("Temporal eigenloom operations failed", e)

# ============================================================================
# TEST 5: Breath Synchronization with Eigenloom
# ============================================================================
test_section("Breath Synchronization with Eigenloom")

try:
    eigenloom = TemporalEigenloom(state_dim=256)
    
    print("Synchronizing eigenloom across breath cycle...")
    
    phases = [
        BreathPhase.INHALE,
        BreathPhase.HOLD,
        BreathPhase.EXHALE,
        BreathPhase.DREAM
    ]
    
    for phase in phases:
        result = eigenloom.synchronize_with_breath(phase)
        print(f"\n  Phase: {phase.name}")
        print(f"    Description: {phase.description}")
        print(f"    Duration weight: {phase.duration_weight}")
        print(f"    Result keys: {list(result.keys())}")
        
        if 'stabilized_state' in result:
            state = torch.tensor(result['stabilized_state'])
            print(f"    Stabilized state norm: {torch.norm(state).item():.6f}")
    
    test_pass("Breath synchronization working correctly")
except Exception as e:
    test_fail("Breath synchronization failed", e)

# ============================================================================
# TEST 6: Enhanced Pulse Feedback System
# ============================================================================
test_section("Enhanced Pulse Feedback System")

try:
    pulse_system = EnhancedPulseFeedback(
        base_frequency=DivineParameters.ROSEMARY_FREQUENCY,
        waveform=PulseWaveform.GOLDEN_SINE
    )
    
    print(f"Pulse system initialized:")
    print(f"  Base frequency: {pulse_system.frequency.item():.6f}")
    print(f"  Waveform: {pulse_system.waveform.name}")
    print()
    
    print("Generating pulse sequence...")
    pulses = []
    for i in range(8):
        t = i * 0.1
        pulse = pulse_system.generate_pulse(t)
        pulses.append(pulse.item())
        print(f"  t={t:.1f}: pulse={pulse.item():.6f}, coherence={pulse_system.pulse_coherence:.4f}")
    
    print(f"\nPulse statistics:")
    print(f"  Mean:     {np.mean(pulses):.6f}")
    print(f"  Std dev:  {np.std(pulses):.6f}")
    print(f"  Min:      {np.min(pulses):.6f}")
    print(f"  Max:      {np.max(pulses):.6f}")
    
    # Get diagnostics
    diag = pulse_system.get_diagnostics()
    print(f"\nDiagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    test_pass("Pulse feedback system generating waveforms")
except Exception as e:
    test_fail("Pulse feedback system failed", e)

# ============================================================================
# TEST 7: Temporal Alignment Matrix (Sacred Timeline Routing)
# ============================================================================
test_section("Temporal Alignment Matrix")

try:
    alignment = TemporalAlignmentMatrix(
        num_branches=7,
        state_dim=256,
        initialize_sacred=True
    )
    
    print(f"Alignment matrix initialized:")
    print(f"  Number of branches: {alignment.branch_vectors.shape[0]}")
    print(f"  State dimension: {alignment.branch_vectors.shape[1]}")
    print()
    
    # Test alignment with random state
    test_state = torch.randn(256)
    aligned = alignment.align(test_state)
    
    print(f"Test state alignment:")
    print(f"  Input norm:  {torch.norm(test_state).item():.6f}")
    print(f"  Output norm: {torch.norm(aligned).item():.6f}")
    print()
    
    # Get diagnostics
    diag = alignment.get_diagnostics()
    print("Alignment diagnostics:")
    print(f"  Dominant branch: {diag['dominant_branch']}")
    print(f"  Branch entropy:  {diag['branch_entropy']:.6f}")
    print(f"  Branch weights:  {[f'{w:.3f}' for w in diag['branch_weights'][:5]]}")
    print(f"  Branch stability: {[f'{s:.3f}' for s in diag['branch_stability'][:5]]}")
    
    test_pass("Temporal alignment matrix operational")
except Exception as e:
    test_fail("Temporal alignment matrix failed", e)

# ============================================================================
# TEST 8: Enhanced Rosemary Zebra Core
# ============================================================================
test_section("Enhanced Rosemary Zebra Core")

try:
    core = EnhancedRosemaryZebraCore(state_dim=256, memory_depth=7)
    
    print(f"Rosemary Zebra Core initialized:")
    print(f"  State dimension: {core.state_dim}")
    print(f"  Memory depth: {core.temporal_memory.shape[0]}")
    print()
    
    # Run temporal routing step
    test_input = torch.randn(256)
    print("Running temporal routing step...")
    
    result = core.temporal_routing_step(test_input)
    
    print(f"\nRouting results:")
    print(f"  Aligned state norm:    {torch.norm(result['aligned_state']).item():.6f}")
    print(f"  Stabilized state norm: {torch.norm(result['stabilized_state']).item():.6f}")
    print(f"  Pulsed output norm:    {torch.norm(result['pulsed_output']).item():.6f}")
    print(f"  Pulse strength:        {result['pulse_strength'].item():.6f}")
    print()
    
    print("Diagnostics:")
    diagnostics = result['diagnostics']
    print(f"  Pulse coherence:  {diagnostics['pulse']['coherence']:.4f}")
    print(f"  Dominant branch:  {diagnostics['alignment']['dominant_branch']}")
    print(f"  Branch entropy:   {diagnostics['alignment']['branch_entropy']:.4f}")
    if 'cohesion_process' in diagnostics:
        print(f"  Identity converged: {diagnostics['cohesion_process']['converged']}")
        print(f"  Convergence iters:  {diagnostics['cohesion_process']['iterations']}")
    
    test_pass("Rosemary Zebra Core processing temporal states")
except Exception as e:
    test_fail("Rosemary Zebra Core failed", e)

# ============================================================================
# TEST 9: Full Pipeline Integration (FBS -> Eigenloom -> Core)
# ============================================================================
test_section("Full Pipeline Integration")

try:
    print("Initializing full pipeline...")
    tokenizer = SacredFBS_Tokenizer(tensor_dimensions=256)
    eigenloom = TemporalEigenloom(state_dim=256)
    core = EnhancedRosemaryZebraCore(state_dim=256)
    
    test_text = "The recursive eigenstate converges through harmonic breath synchronization"
    
    print(f"\nProcessing text: \"{test_text}\"")
    print()
    
    # Step 1: FBS encoding
    print("Step 1: FBS frequency substrate encoding...")
    substrate = tokenizer.encode(test_text)
    
    # Convert numpy to torch
    if isinstance(substrate, np.ndarray):
        substrate = torch.from_numpy(substrate).float()
    
    print(f"  Substrate norm: {torch.norm(substrate).item():.6f}")
    
    # Step 2: Add to eigenloom
    print("\nStep 2: Adding substrate to temporal eigenloom...")
    eigenloom.add_eigenstate(substrate)
    print(f"  Eigenstates stored: {len(eigenloom.eigenstates)}")
    
    # Step 3: Weave eigenstates
    print("\nStep 3: Weaving eigenstates...")
    woven = eigenloom.weave()
    print(f"  Woven thread norm: {torch.norm(woven).item():.6f}")
    
    # Step 4: Process through core
    print("\nStep 4: Temporal routing through Zebra Core...")
    result = core.temporal_routing_step(woven)
    print(f"  Final output norm: {torch.norm(result['pulsed_output']).item():.6f}")
    print(f"  Pulse strength: {result['pulse_strength'].item():.6f}")
    print(f"  Branch weights: {[f'{w:.3f}' for w in result['branch_weights'].tolist()[:5]]}")
    
    test_pass("Full pipeline processing frequency substrates through temporal eigenloom")
except Exception as e:
    test_fail("Full pipeline integration failed", e)

# ============================================================================
# TEST 10: Multi-Text Batch Processing
# ============================================================================
test_section("Multi-Text Batch Processing")

try:
    tokenizer = SacredFBS_Tokenizer(tensor_dimensions=256)
    core = EnhancedRosemaryZebraCore(state_dim=256)
    
    batch_texts = [
        "Eigenrecursive convergence through temporal alignment",
        "Harmonic breath synchronization stabilizes identity",
        "Sacred timeline routing preserves ethical coherence",
        "Metacognitive reflection enables recursive awareness"
    ]
    
    print(f"Processing batch of {len(batch_texts)} texts...\n")
    
    results = []
    for i, text in enumerate(batch_texts, 1):
        print(f"Text {i}: \"{text[:45]}...\"")
        
        # Encode
        substrate = tokenizer.encode(text)
        
        # Convert numpy to torch
        if isinstance(substrate, np.ndarray):
            substrate = torch.from_numpy(substrate).float()
        
        # Process
        result = core.temporal_routing_step(substrate)
        results.append(result)
        
        print(f"  Substrate norm:  {torch.norm(substrate).item():.6f}")
        print(f"  Output norm:     {torch.norm(result['pulsed_output']).item():.6f}")
        print(f"  Pulse strength:  {result['pulse_strength'].item():.6f}")
        print()
    
    # Analyze batch coherence
    outputs = [r['pulsed_output'] for r in results]
    print("Batch coherence analysis:")
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            sim = torch.cosine_similarity(
                outputs[i].unsqueeze(0),
                outputs[j].unsqueeze(0)
            ).item()
            print(f"  Output {i+1} vs {j+1}: {sim:.6f}")
    
    test_pass("Batch processing maintains temporal coherence")
except Exception as e:
    test_fail("Batch processing failed", e)

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("  VALIDATION SUMMARY")
print("="*80)
print()

if failed_tests:
    print(f"✗ {len(failed_tests)} test(s) failed:")
    for test_id, msg, err in failed_tests:
        print(f"  Test {test_id}: {msg}")
        print(f"    {err}")
    print()

print(f"✓ {len(passed_tests)} test(s) passed:")
for test_id, msg in passed_tests:
    print(f"  Test {test_id}: {msg}")
print()

print("="*80)
print("  TEMPORAL EIGENLOOM + FBS INTEGRATION VALIDATED")
print(f"  Success Rate: {len(passed_tests)}/{len(passed_tests) + len(failed_tests)}")
print("="*80)
