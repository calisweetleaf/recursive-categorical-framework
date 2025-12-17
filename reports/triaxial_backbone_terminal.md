(.venv) PS C:\Users\treyr\Desktop\recursive-categorical-framework> .\.venv\Scripts\Activate.ps1; python test_triaxial_backbone.py

======================================================================
                     TRIAXIAL BACKBONE TEST SUITE
======================================================================


[STAGE 1] Import Validation
------------------------------------------------------------
✓ Imported triaxial_backbone module
✓ Imported RecursiveTensor
✓ Imported SymbolicQuantumState (EthicalTensor)
✓ Imported MetacognitiveTensor
✓ Imported BayesianConfigurationOrchestrator
✓ Imported EigenrecursionStabilizer

[STAGE 2] Configuration Validation
------------------------------------------------------------
✓ Default config: recursive_dim=16, ethical_dim=5
✓ Custom config: recursive_dim=64
✓ Config validation correctly rejects invalid values

[STAGE 3] Backbone Initialization
------------------------------------------------------------
ℹ Creating backbone...
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=32, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a0261d21]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 32
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a0261d21]: Stabilizer initialized
✓ Backbone created: ID=a0261d21
✓ TriaxialField initialized
✓ EigenrecursionStabilizer initialized
✓ All three tensor axes initialized
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a0261d21] shutdown complete

[STAGE 4] Text Forward Pass
------------------------------------------------------------
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a1ee6fda]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a1ee6fda]: Stabilizer initialized
ℹ Processing text: 'I think therefore I am.'
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] EthicalTensors: Applied symbol 'I think therefore I am.' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=6.52ms, status=ACTIVE
✓ Forward pass complete in 6.52ms
✓ Integrated vector: [0.10940053 0.26295993 0.02056432]
✓ Recursive axis: stability=0.1094
✓ Ethical axis: norm=0.2630
✓ Metacog axis: consciousness=0.0206
✓ Convergence status: ACTIVE
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[a1ee6fda] shutdown complete

[STAGE 5] Tensor Forward Pass
------------------------------------------------------------
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[7d174efd]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[7d174efd]: Stabilizer initialized
ℹ Processing tensor: shape=torch.Size([256])
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=3.52ms, status=CONVERGING
✓ Forward pass complete in 3.52ms
✓ Integrated magnitude: 1.0149
✓ Triaxial coherence: 0.3794
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[7d174efd] shutdown complete

[STAGE 6] Parallel Computation Validation
------------------------------------------------------------
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[c61e8f51]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[c61e8f51]: Stabilizer initialized
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] EthicalTensors: Applied symbol 'Testing parallel computation in triaxial backbone' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=1.01ms, status=CONVERGING
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[c61e8f51] shutdown complete
✓ Parallel computation: 1.01ms
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[75ecd12d]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[75ecd12d]: Stabilizer initialized
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] EthicalTensors: Applied symbol 'Testing parallel computation in triaxial backbone' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=0.52ms, status=CONVERGING
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[75ecd12d] shutdown complete
✓ Sequential computation: 0.52ms
ℹ Output difference: 0.000000
✓ Parallel and sequential outputs are consistent
ℹ Speedup: 0.51x

[STAGE 7] Stability Analysis
------------------------------------------------------------
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[f099f2f4]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[f099f2f4]: Stabilizer initialized
ℹ Running 5 forward passes to build state history...
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] EthicalTensors: Applied symbol 'Iteration 0: testing eigenrecursion stability' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=1.01ms, status=CONVERGING
ℹ   Pass 1: status=CONVERGING, dist=0.711311
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #2
00:47:51 [INFO] EthicalTensors: Applied symbol 'Iteration 1: testing eigenrecursion stability' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #2: time=1.01ms, status=CONVERGING
ℹ   Pass 2: status=CONVERGING, dist=0.000433
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #3
00:47:51 [INFO] EthicalTensors: Applied symbol 'Iteration 2: testing eigenrecursion stability' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #3: time=1.01ms, status=CONVERGING
ℹ   Pass 3: status=CONVERGING, dist=0.000444
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #4
00:47:51 [INFO] EthicalTensors: Applied symbol 'Iteration 3: testing eigenrecursion stability' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #4: time=1.01ms, status=CONVERGING
ℹ   Pass 4: status=CONVERGING, dist=0.000874
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #5
00:47:51 [INFO] EthicalTensors: Applied symbol 'Iteration 4: testing eigenrecursion stability' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #5: time=1.00ms, status=CONVERGING
ℹ   Pass 5: status=CONVERGING, dist=0.000921
✓ Stabilizer state: {'iterations_to_convergence': None, 'stability_gradient': None, 'oscillation_index': 0.0, 'num_fixed_points_found': 0, 'convergence_achieved': False}
ℹ No monotonic convergence (expected for varied inputs)
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[f099f2f4] shutdown complete

[STAGE 8] Metrics Collection
------------------------------------------------------------
00:47:51 [INFO] TriaxialBackbone: Initialized RecursiveTensor: dim=16, rank=2
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] EthicalTensors: Initialized SymbolicQuantumState with field shape (16, 16)
00:47:51 [INFO] EthicalTensors: Initialized QuantumBreathAdapter with 5 ethical dimensions
00:47:51 [INFO] TriaxialBackbone: Initialized EthicalTensor: field_shape=(16, 16), dim=5
00:47:51 [INFO] TriaxialBackbone: Initialized MetacognitiveTensor: state_dim=256, layers=3
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[25880611]: Field initialized
00:47:51 [INFO] EigenrecursionStabilizer: Initialized Eigenrecursion Stabilizer with dimension 16
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[25880611]: Stabilizer initialized
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #1
00:47:51 [INFO] EthicalTensors: Applied symbol 'Test input one' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #1: time=1.01ms, status=ACTIVE
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone.forward #2
00:47:51 [INFO] EthicalTensors: Applied symbol 'Test input two' at position (0.5, 0.5) with intensity 1.00
00:47:51 [INFO] TriaxialBackbone: Triaxial computation #2: time=0.51ms, status=ACTIVE
✓ Backbone ID: 25880611
✓ Forward count: 2
✓ State history: 2
✓ Fixed points found: 0
✓ Last state ID: abb2facc
✓ Last status: ACTIVE
00:47:51 [INFO] TriaxialBackbone: TriaxialField shutdown complete
00:47:51 [INFO] TriaxialBackbone: TriaxialBackbone[25880611] shutdown complete
00:47:51 [INFO] TriaxialBackboneTest: JSON report written to: C:\Users\treyr\Desktop\recursive-categorical-framework\logs\triaxial_backbone_20251217_004749.json
00:47:51 [INFO] TriaxialBackboneTest: Markdown report written to: C:\Users\treyr\Desktop\recursive-categorical-framework\reports\triaxial_backbone_report.md

======================================================================
                             TEST SUMMARY
======================================================================

  Stage 1: Import Validation                   [PASS] (1.96s)
  Stage 2: Configuration Validation            [PASS] (0.00s)
  Stage 3: Backbone Initialization             [PASS] (0.03s)
  Stage 4: Text Forward Pass                   [PASS] (0.01s)
  Stage 5: Tensor Forward Pass                 [PASS] (0.01s)
  Stage 6: Parallel Computation Validation     [PASS] (0.01s)
  Stage 7: Stability Analysis                  [PASS] (0.02s)
  Stage 8: Metrics Collection                  [PASS] (0.01s)

✓ All 8 stages passed in 2.06s

ℹ Log file: C:\Users\treyr\Desktop\recursive-categorical-framework\logs\triaxial_backbone_20251217_004749.log
ℹ JSON report: C:\Users\treyr\Desktop\recursive-categorical-framework\logs\triaxial_backbone_20251217_004749.json
ℹ Markdown report: C:\Users\treyr\Desktop\recursive-categorical-framework\reports\triaxial_backbone_report.md

======================================================================
                            TEST COMPLETE
======================================================================
