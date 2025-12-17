# Triaxial Backbone Test Report

**Test Run:** 2025-12-17 00:47:51
**Total Duration:** 2.06s

## Summary

| Metric | Value |
|--------|-------|
| Total Stages | 8 |
| Passed | 8 |
| Failed | 0 |
| Warnings | 0 |

## Stage Results

### Stage 1: Import Validation

**Status:** ✅ PASS
**Duration:** 1.96s
**Details:** All 6 imports successful

**Metrics:**
```json
{
  "backbone_imported": true,
  "recursive_tensor_imported": true,
  "ethical_tensor_imported": true,
  "metacog_tensor_imported": true,
  "bayesian_imported": true,
  "stabilizer_imported": true
}
```

### Stage 2: Configuration Validation

**Status:** ✅ PASS
**Duration:** 0.00s
**Details:** Configuration system validated

**Metrics:**
```json
{
  "default_config": true,
  "custom_config": true,
  "validation_works": true
}
```

### Stage 3: Backbone Initialization

**Status:** ✅ PASS
**Duration:** 0.03s
**Details:** Backbone ID: a0261d21

**Metrics:**
```json
{
  "backbone_id": "a0261d21",
  "field_initialized": true,
  "stabilizer_initialized": true,
  "all_tensors_initialized": true
}
```

### Stage 4: Text Forward Pass

**Status:** ✅ PASS
**Duration:** 0.01s
**Details:** Time: 6.52ms

**Metrics:**
```json
{
  "computation_time_ms": 6.522655487060547,
  "integrated_vector": [
    0.10940052568912506,
    0.26295992732048035,
    0.02056431770324707
  ],
  "convergence_status": "ACTIVE"
}
```

### Stage 5: Tensor Forward Pass

**Status:** ✅ PASS
**Duration:** 0.01s
**Details:** Coherence: 0.3794

**Metrics:**
```json
{
  "computation_time_ms": 3.515005111694336,
  "integrated_magnitude": 1.0148629952037116,
  "coherence": 0.3793519214688522
}
```

### Stage 6: Parallel Computation Validation

**Status:** ✅ PASS
**Duration:** 0.01s
**Details:** Speedup: 0.51x

**Metrics:**
```json
{
  "output_difference": 0.0,
  "parallel_time_ms": 1.0075569152832031,
  "sequential_time_ms": 0.5166530609130859,
  "speedup": 0.512778040700426
}
```

### Stage 7: Stability Analysis

**Status:** ✅ PASS
**Duration:** 0.02s
**Details:** Final status: CONVERGING

**Metrics:**
```json
{
  "stabilizer_state": {
    "iterations_to_convergence": null,
    "stability_gradient": null,
    "oscillation_index": 0.0,
    "num_fixed_points_found": 0,
    "convergence_achieved": false
  },
  "final_convergence_status": "CONVERGING",
  "state_history_len": 5
}
```

### Stage 8: Metrics Collection

**Status:** ✅ PASS
**Duration:** 0.01s
**Details:** Forward passes: 2

**Metrics:**
```json
{
  "backbone_id": "25880611",
  "forward_count": 2,
  "state_history_len": 2,
  "fixed_points_found": 0,
  "stabilizer_metrics": {
    "iterations_to_convergence": null,
    "stability_gradient": null,
    "oscillation_index": 0.0,
    "num_fixed_points_found": 0,
    "convergence_achieved": false
  },
  "config": {
    "recursive_dim": 16,
    "ethical_dim": 5,
    "metacog_dim": 256,
    "parallel": true
  },
  "last_state": {
    "state_id": "abb2facc",
    "timestamp": 1765954071.3172967,
    "computation_time_ms": 0.5056858062744141,
    "convergence_status": "ACTIVE",
    "convergence_distance": 0.0008293743332023416,
    "recursive": {
      "transformed": [
        0.10020443797111511,
        0.12048390507698059,
        0.13718463480472565,
        0.13837754726409912,
        0.038173120468854904,
        0.1252555400133133,
        0.13122008740901947,
        0.13360591232776642,
        0.1395704597234726,
        0.13837754726409912,
        0.038173120468854904,
        0.13837754726409912,
        0.14195628464221954,
        0.13241299986839294,
        0.0,
        0.0
      ],
      "spectral_radius": 2.2873971462249756,
      "norm": 1.5124143362045288,
      "density": 0.875,
      "stability_score": 0.10333582758903503
    },
    "ethical": {
      "breath_phase": "INHALE",
      "resonance_norm": 17.48086316041387,
      "ethical_vector_norm": 0.0005852667964063585
    },
    "metacog": {
      "coherence": 0.6666674017906189,
      "consciousness_proxy": 0.6666674017906189,
      "self_reference_ratio": 0.0,
      "consciousness_level": 0.0
    },
    "stability_metrics": {
      "magnitude": 0.10333748497391326,
      "coherence": 0.0,
      "ere": 0.10333582758903503,
      "rbu": 0.0005852667964063585,
      "es": 0.0,
      "unified_state_norm": 0.46007147431373596
    },
    "has_integrated_vector": true,
    "integrated_norm": 0.10333748497391326
  }
}
```
