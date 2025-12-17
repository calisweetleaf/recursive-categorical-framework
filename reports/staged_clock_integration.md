# Staged Internal Clock Integration Test Report

**Timestamp:** 2025-12-16 23:46:44
**Total Duration:** 37.29s
**Device:** cpu

## Configuration

- Stage warmup time: 3.0s
- Oscillator settle time: 2.0s
- Integration cycles: 5

## Results

### Stage 0: Clock Burn-in (Oscillator Stabilization) OK PASS

- **Duration:** 10.04s
- **Details:** Burn-in 10.0s, 11 oscillators

### Stage 1: Clock Verification (Post Burn-in) OK PASS

- **Duration:** 0.00s
- **Details:** Verified 11 oscillators, phase=NIGHT

### Stage 2: Clock Dynamics Verification OK PASS

- **Duration:** 2.91s
- **Details:** Completed 5 cycles, alertness_range=0.0015

### Stage 3: Temporal Eigenstate Integration OK PASS

- **Duration:** 3.51s
- **Details:** Depth=5, regime=Compression

### Stage 4: Recursive Stabilization Integration OK PASS

- **Duration:** 2.07s
- **Details:** Converged=False, iterations=64
- **Warnings:** 1
  - Stabilization did not converge within iteration limit

### Stage 5: Full System Integration OK PASS

- **Duration:** 3.75s
- **Details:** Completed 5 cycles, 0 regime changes

