(.venv) PS C:\Users\treyr\Desktop\recursive-categorical-framework> python triaxial_backbone.py
2025-12-17 00:40:40 | WARNING | TriaxialBackbone | bayesian_config_orchestrator not available - Bayesian features disabled
======================================================================
TRIAXIAL BACKBONE - Sanity Check
======================================================================
2025-12-17 00:40:40 | INFO | TriaxialBackbone | Initialized RecursiveTensor: dim=32, rank=2
2025-12-17 00:40:40 | INFO | EthicalTensors | Initialized QuantumBreathAdapter with 5 ethical dimensions
2025-12-17 00:40:40 | INFO | EthicalTensors | Initialized SymbolicQuantumState with field shape (16, 16)
2025-12-17 00:40:40 | INFO | EthicalTensors | Initialized QuantumBreathAdapter with 5 ethical dimensions
2025-12-17 00:40:40 | INFO | TriaxialBackbone | Initialized EthicalTensor: field_shape=(16, 16), dim=5
2025-12-17 00:40:40 | INFO | TriaxialBackbone | Initialized MetacognitiveTensor: state_dim=256, layers=3
2025-12-17 00:40:40 | INFO | TriaxialBackbone | TriaxialBackbone[f5369119]: Field initialized
2025-12-17 00:40:40 | INFO | EigenrecursionStabilizer | Initialized Eigenrecursion Stabilizer with dimension 32
2025-12-17 00:40:40 | INFO | TriaxialBackbone | TriaxialBackbone[f5369119]: Stabilizer initialized
2025-12-17 00:40:40 | WARNING | TriaxialBackbone | Could not initialize Bayesian orchestrator: 'NoneType' object is not callable
2025-12-17 00:40:40 | INFO | TriaxialBackbone | TriaxialBackbone.forward #1
2025-12-17 00:40:40 | INFO | EthicalTensors | Applied symbol 'I think therefore I am.' at position (0.5, 0.5) with intensity 1.00
2025-12-17 00:40:40 | INFO | TriaxialBackbone | Triaxial computation #1: time=3.55ms, status=COLD_START

Conv Status: COLD_START
Computation: 3.55ms
Integrated Vector: [0.06096046 0.0007633  0.        ]
Stability: {'magnitude': 0.06096523904779523, 'coherence': 0.0, 'ere': 0.06096046045422554, 'rbu': 0.0007633042405359447, 'es': 0.0, 'unified_state_norm': 0.4299820065498352}

Backbone Metrics:
{
  "backbone_id": "f5369119",
  "forward_count": 1,
  "state_history_len": 1,
  "fixed_points_found": 0,
  "stabilizer_metrics": {
    "iterations_to_convergence": null,
    "stability_gradient": null,
    "oscillation_index": 0.0,
    "num_fixed_points_found": 0,
    "convergence_achieved": false
  },
  "config": {
    "recursive_dim": 32,
    "ethical_dim": 5,
    "metacog_dim": 256,
    "parallel": true
  },
  "last_state": {
    "state_id": "5d3deb43",
    "timestamp": 1765953640.9064028,
    "computation_time_ms": 3.548860549926758,
    "convergence_status": "COLD_START",
    "convergence_distance": 0.9648017028245456,
    "recursive": {
      "transformed": [
        0.07008052617311478,
        0.03072023205459118,
        0.11136084049940109,
        0.09984075278043747,
        0.1008007600903511,
        0.10560079663991928,
        0.10272077471017838,
        0.03072023205459118,
        0.11136084049940109,
        0.09984075278043747,
        0.09696073085069656,
        0.10944082587957382,
        0.09696073085069656,
        0.0979207381606102,
        0.10656080394983292,
        0.10944082587957382,
        0.09696073085069656,
        0.03072023205459118,
        0.07008052617311478,
        0.03072023205459118,
        0.09312070161104202,
        0.10464078933000565,
        0.04416033253073692,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "spectral_radius": 3.0849366188049316,
      "norm": 1.7563987970352173,
      "density": 0.71875,
      "stability_score": 0.06096046045422554
    },
    "ethical": {
      "breath_phase": "INHALE",
      "resonance_norm": 6.130765844964513,
      "ethical_vector_norm": 0.0007633042405359447
    },
    "metacog": {
      "coherence": 0.6666674613952637,
      "consciousness_proxy": 0.6666674613952637,
      "self_reference_ratio": 0.0,
      "consciousness_level": 0.0
    },
    "stability_metrics": {
      "magnitude": 0.06096523904779523,
      "coherence": 0.0,
      "ere": 0.06096046045422554,
      "rbu": 0.0007633042405359447,
      "es": 0.0,
      "unified_state_norm": 0.4299820065498352
    },
    "has_integrated_vector": true,
    "integrated_norm": 0.06096523904779523
  }
}
2025-12-17 00:40:40 | INFO | TriaxialBackbone | TriaxialField shutdown complete
2025-12-17 00:40:40 | INFO | TriaxialBackbone | TriaxialBackbone[f5369119] shutdown complete

======================================================================
SANITY CHECK COMPLETE
======================================================================