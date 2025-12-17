(.venv) PS C:\Users\treyr\Desktop\recursive-categorical-framework> python test_temporal_eigenstate.py
23:46:07 [INFO] StagedClockIntegration: 
======================================================================
2025-12-16 23:46:07 [INFO] (Temporal Coherence) 
======================================================================
23:46:07 [INFO] StagedClockIntegration: STAGED INTERNAL CLOCK INTEGRATION TEST
2025-12-16 23:46:07 [INFO] (Temporal Coherence) STAGED INTERNAL CLOCK INTEGRATION TEST
23:46:07 [INFO] StagedClockIntegration: ======================================================================

2025-12-16 23:46:07 [INFO] (Temporal Coherence) ======================================================================

23:46:07 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:07 [INFO] (Temporal Coherence) ======================================================================
23:46:07 [INFO] StagedClockIntegration: STAGE 0: Clock Burn-in (Oscillator Stabilization)
2025-12-16 23:46:07 [INFO] (Temporal Coherence) STAGE 0: Clock Burn-in (Oscillator Stabilization)
23:46:07 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:07 [INFO] (Temporal Coherence) ======================================================================
23:46:07 [INFO] StagedClockIntegration: Creating TemporalCoherence instance...
2025-12-16 23:46:07 [INFO] (Temporal Coherence) Creating TemporalCoherence instance...
23:46:07 [INFO] StagedClockIntegration: Running burn-in for 10.0s to establish oscillator rhythms...
2025-12-16 23:46:07 [INFO] (Temporal Coherence) Running burn-in for 10.0s to establish oscillator rhythms...
2025-12-16 23:46:12 [DEBUG] (Temporal Coherence)   Burn-in progress: 5.0s / 10.0s (alertness=0.6844)
2025-12-16 23:46:17 [DEBUG] (Temporal Coherence)   Burn-in progress: 10.0s / 10.0s (alertness=0.6736)
23:46:17 [INFO] StagedClockIntegration: Analyzing frequency domain characteristics...
2025-12-16 23:46:17 [INFO] (Temporal Coherence) Analyzing frequency domain characteristics...
23:46:17 [INFO] StagedClockIntegration: Oscillator stability metrics:
2025-12-16 23:46:17 [INFO] (Temporal Coherence) Oscillator stability metrics:
23:46:17 [INFO] StagedClockIntegration:   circadian: CV=0.0001
2025-12-16 23:46:17 [INFO] (Temporal Coherence)   circadian: CV=0.0001
23:46:17 [INFO] StagedClockIntegration:   alpha_wave: CV=1.5194
2025-12-16 23:46:17 [INFO] (Temporal Coherence)   alpha_wave: CV=1.5194
23:46:17 [INFO] StagedClockIntegration:   heart_beat: CV=14.6107
2025-12-16 23:46:17 [INFO] (Temporal Coherence)   heart_beat: CV=14.6107
23:46:17 [INFO] StagedClockIntegration:   alertness: CV=0.0155
2025-12-16 23:46:17 [INFO] (Temporal Coherence)   alertness: CV=0.0155
23:46:17 [INFO] StagedClockIntegration: Frequency domain analysis:
2025-12-16 23:46:17 [INFO] (Temporal Coherence) Frequency domain analysis:
23:46:17 [INFO] StagedClockIntegration: OK Burn-in complete: 11 oscillators stabilized, phase=NIGHT
2025-12-16 23:46:17 [INFO] (Temporal Coherence) OK Burn-in complete: 11 oscillators stabilized, phase=NIGHT
23:46:17 [INFO] StagedClockIntegration:
Waiting 3.0s before Stage 1...

2025-12-16 23:46:17 [INFO] (Temporal Coherence)
Waiting 3.0s before Stage 1...

23:46:20 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:20 [INFO] (Temporal Coherence) ======================================================================
23:46:20 [INFO] StagedClockIntegration: STAGE 1: Clock Verification (Post Burn-in)
2025-12-16 23:46:20 [INFO] (Temporal Coherence) STAGE 1: Clock Verification (Post Burn-in)
23:46:20 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:20 [INFO] (Temporal Coherence) ======================================================================
23:46:20 [INFO] StagedClockIntegration: Verifying clock operations with stabilized oscillators...
2025-12-16 23:46:20 [INFO] (Temporal Coherence) Verifying clock operations with stabilized oscillators...
2025-12-16 23:46:20 [DEBUG] (Temporal Coherence)   OK TimePoint created: 1765950380.6661103
2025-12-16 23:46:20 [DEBUG] (Temporal Coherence)   OK Clock updated, elapsed: 0.0000s
2025-12-16 23:46:20 [DEBUG] (Temporal Coherence)   OK Circadian phase: NIGHT
2025-12-16 23:46:20 [DEBUG] (Temporal Coherence)   OK Alertness level: 0.6722
2025-12-16 23:46:20 [DEBUG] (Temporal Coherence)   OK System status retrieved (5 sections)
23:46:20 [INFO] StagedClockIntegration: OK Clock verified with 11 oscillators after burn-in
2025-12-16 23:46:20 [INFO] (Temporal Coherence) OK Clock verified with 11 oscillators after burn-in
23:46:20 [INFO] StagedClockIntegration:
Waiting 3.0s before Stage 2...

2025-12-16 23:46:20 [INFO] (Temporal Coherence)
Waiting 3.0s before Stage 2...

23:46:23 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:23 [INFO] (Temporal Coherence) ======================================================================
23:46:23 [INFO] StagedClockIntegration: STAGE 2: Clock Dynamics Verification
2025-12-16 23:46:23 [INFO] (Temporal Coherence) STAGE 2: Clock Dynamics Verification
23:46:23 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:23 [INFO] (Temporal Coherence) ======================================================================
23:46:23 [INFO] StagedClockIntegration: Initial state: phase=NIGHT, alertness=0.6713
2025-12-16 23:46:23 [INFO] (Temporal Coherence) Initial state: phase=NIGHT, alertness=0.6713
23:46:23 [INFO] StagedClockIntegration: Running 5 update cycles...
2025-12-16 23:46:23 [INFO] (Temporal Coherence) Running 5 update cycles...
2025-12-16 23:46:24 [DEBUG] (Temporal Coherence)   Cycle 1: alertness=0.6709, phase=NIGHT
2025-12-16 23:46:24 [DEBUG] (Temporal Coherence)   Cycle 2: alertness=0.6705, phase=NIGHT
2025-12-16 23:46:25 [DEBUG] (Temporal Coherence)   Cycle 3: alertness=0.6701, phase=NIGHT
2025-12-16 23:46:25 [DEBUG] (Temporal Coherence)   Cycle 4: alertness=0.6697, phase=NIGHT
2025-12-16 23:46:26 [DEBUG] (Temporal Coherence)   Cycle 5: alertness=0.6694, phase=NIGHT
23:46:26 [INFO] StagedClockIntegration: Alertness range over cycles: 0.0015
2025-12-16 23:46:26 [INFO] (Temporal Coherence) Alertness range over cycles: 0.0015
23:46:26 [INFO] StagedClockIntegration: Testing entrainment with light signal...
2025-12-16 23:46:26 [INFO] (Temporal Coherence) Testing entrainment with light signal...
2025-12-16 23:46:26 [DEBUG] (Temporal Coherence)   Post-entrainment alertness: 0.6735
23:46:26 [INFO] StagedClockIntegration: Testing attention state changes...
2025-12-16 23:46:26 [INFO] (Temporal Coherence) Testing attention state changes...
2025-12-16 23:46:26 [DEBUG] (Temporal Coherence)   Attention state: NORMAL
23:46:26 [INFO] StagedClockIntegration: 
Waiting 3.0s before Stage 3...

2025-12-16 23:46:26 [INFO] (Temporal Coherence)
Waiting 3.0s before Stage 3...

23:46:29 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:29 [INFO] (Temporal Coherence) ======================================================================
23:46:29 [INFO] StagedClockIntegration: STAGE 3: Temporal Eigenstate Integration
2025-12-16 23:46:29 [INFO] (Temporal Coherence) STAGE 3: Temporal Eigenstate Integration
23:46:29 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:29 [INFO] (Temporal Coherence) ======================================================================
23:46:29 [INFO] StagedClockIntegration: Creating TemporalEigenstate with clock integration...
2025-12-16 23:46:29 [INFO] (Temporal Coherence) Creating TemporalEigenstate with clock integration...
23:46:29 [INFO] StagedClockIntegration: Allowing 2.0s for eigenstate-clock coupling...
2025-12-16 23:46:29 [INFO] (Temporal Coherence) Allowing 2.0s for eigenstate-clock coupling...
23:46:31 [INFO] StagedClockIntegration: Running 5 synchronized dilation cycles...
2025-12-16 23:46:31 [INFO] (Temporal Coherence) Running 5 synchronized dilation cycles...
2025-12-16 23:46:31 [DEBUG] (Temporal Coherence)   Cycle 1: dilation=0.7615, regime=Compression
2025-12-16 23:46:31 [DEBUG] (Temporal Coherence)   Cycle 2: dilation=0.7795, regime=Compression
2025-12-16 23:46:32 [DEBUG] (Temporal Coherence)   Cycle 3: dilation=0.7975, regime=Compression
2025-12-16 23:46:32 [DEBUG] (Temporal Coherence)   Cycle 4: dilation=0.8155, regime=Compression
2025-12-16 23:46:32 [DEBUG] (Temporal Coherence)   Cycle 5: dilation=0.8335, regime=Compression
23:46:33 [INFO] StagedClockIntegration: OK Completed integration: depth=5, regime=Compression
2025-12-16 23:46:33 [INFO] (Temporal Coherence) OK Completed integration: depth=5, regime=Compression
23:46:33 [INFO] StagedClockIntegration:
Waiting 3.0s before Stage 4...

2025-12-16 23:46:33 [INFO] (Temporal Coherence)
Waiting 3.0s before Stage 4...

23:46:36 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:36 [INFO] (Temporal Coherence) ======================================================================
23:46:36 [INFO] StagedClockIntegration: STAGE 4: Recursive Stabilization Integration
2025-12-16 23:46:36 [INFO] (Temporal Coherence) STAGE 4: Recursive Stabilization Integration
23:46:36 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:36 [INFO] (Temporal Coherence) ======================================================================
23:46:36 [INFO] StagedClockIntegration: Creating RecursiveStabilizationPoint with clock...
2025-12-16 23:46:36 [INFO] (Temporal Coherence) Creating RecursiveStabilizationPoint with clock...
2025-12-16 23:46:36 [INFO] (Temporal Coherence) Initialized RecursiveStabilizationPoint system with dimension 32
23:46:38 [INFO] StagedClockIntegration: Initial clock alertness: 0.6708
2025-12-16 23:46:38 [INFO] (Temporal Coherence) Initial clock alertness: 0.6708
23:46:38 [INFO] StagedClockIntegration: Running stabilization with clock synchronization...
2025-12-16 23:46:38 [INFO] (Temporal Coherence) Running stabilization with clock synchronization...
2025-12-16 23:46:38 [INFO] (Temporal Coherence) Temporal eigenstate reset to initial values
2025-12-16 23:46:38 [INFO] (Temporal Coherence) Reached critical recursive depth 7: First Harmonic
2025-12-16 23:46:38 [INFO] (Temporal Coherence) Maximum iterations reached without eigenstate detection
23:46:38 [INFO] StagedClockIntegration: OK Stabilization complete: converged=False, iterations=64, alertness_delta=-0.0002
2025-12-16 23:46:38 [INFO] (Temporal Coherence) OK Stabilization complete: converged=False, iterations=64, alertness_delta=-0.0002
23:46:38 [INFO] StagedClockIntegration:
Waiting 3.0s before Stage 5...

2025-12-16 23:46:38 [INFO] (Temporal Coherence)
Waiting 3.0s before Stage 5...

23:46:41 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:41 [INFO] (Temporal Coherence) ======================================================================
23:46:41 [INFO] StagedClockIntegration: STAGE 5: Full System Integration
2025-12-16 23:46:41 [INFO] (Temporal Coherence) STAGE 5: Full System Integration
23:46:41 [INFO] StagedClockIntegration: ======================================================================
2025-12-16 23:46:41 [INFO] (Temporal Coherence) ======================================================================
23:46:41 [INFO] StagedClockIntegration: Creating TemporalEigenstateNode with full clock integration...
2025-12-16 23:46:41 [INFO] (Temporal Coherence) Creating TemporalEigenstateNode with full clock integration...
23:46:43 [INFO] StagedClockIntegration: Running 5 synchronized forward passes...
2025-12-16 23:46:43 [INFO] (Temporal Coherence) Running 5 synchronized forward passes...
2025-12-16 23:46:43 [DEBUG] (Temporal Coherence)   Cycle 1: regime=Compression, alertness=0.6699, phase=NIGHT
2025-12-16 23:46:43 [DEBUG] (Temporal Coherence)   Cycle 2: regime=Compression, alertness=0.6697, phase=NIGHT
2025-12-16 23:46:43 [DEBUG] (Temporal Coherence)   Cycle 3: regime=Compression, alertness=0.6696, phase=NIGHT
2025-12-16 23:46:44 [DEBUG] (Temporal Coherence)   Cycle 4: regime=Compression, alertness=0.6695, phase=NIGHT
2025-12-16 23:46:44 [DEBUG] (Temporal Coherence)   Cycle 5: regime=Compression, alertness=0.6694, phase=NIGHT
23:46:44 [INFO] StagedClockIntegration: OK Full integration complete: 0 regime changes across 5 cycles
2025-12-16 23:46:44 [INFO] (Temporal Coherence) OK Full integration complete: 0 regime changes across 5 cycles
23:46:44 [INFO] StagedClockIntegration: Reports written to:
  - C:\Users\treyr\Desktop\recursive-categorical-framework\reports\staged_clock_integration.json
  - C:\Users\treyr\Desktop\recursive-categorical-framework\reports\staged_clock_integration.md
2025-12-16 23:46:44 [INFO] (Temporal Coherence) Reports written to:
  - C:\Users\treyr\Desktop\recursive-categorical-framework\reports\staged_clock_integration.json
  - C:\Users\treyr\Desktop\recursive-categorical-framework\reports\staged_clock_integration.md

======================================================================
STAGED INTEGRATION TEST SUMMARY
======================================================================
OK PASS | Stage 0: Clock Burn-in (Oscillator Stabilization)
       Duration: 10.04s
       Burn-in 10.0s, 11 oscillators

OK PASS | Stage 1: Clock Verification (Post Burn-in)
       Duration: 0.00s
       Verified 11 oscillators, phase=NIGHT

OK PASS | Stage 2: Clock Dynamics Verification
       Duration: 2.91s
       Completed 5 cycles, alertness_range=0.0015

OK PASS | Stage 3: Temporal Eigenstate Integration
       Duration: 3.51s
       Depth=5, regime=Compression

OK PASS | Stage 4: Recursive Stabilization Integration
       Duration: 2.07s
       Converged=False, iterations=64
       Warnings: 1

OK PASS | Stage 5: Full System Integration
       Duration: 3.75s
       Completed 5 cycles, 0 regime changes

Total: 6/6 stages passed in 37.29s
======================================================================