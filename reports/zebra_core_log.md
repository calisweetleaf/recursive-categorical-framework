```text
(.venv) PS C:\Users\treyr\Desktop\recursive-categorical-framework> python test_zebra_core.py

╔══════════════════════════════════════════════════════════════════════════════╗
║                         ZEBA CORE SYSTEM DIAGNOSTICS                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
  System Time: 2025-11-27 17:25:15

┌── CORE INITIALIZATION ───────────────────────────────────────────────────────
│  ZEBA Stabilizer           : [  NOMINAL   ]
│  Dimension                 : 64
│  Moral Theta               : 0.95
└──────────────────────────────────────────────────────────────────────────────

┌── FIXED POINT DYNAMICS ──────────────────────────────────────────────────────
│  » Initiating contraction mapping...
│  Convergence Time          : 1.51 ms
│  Iterations                : 20
│  Final Error               : 0.00000302
│  Convergence               : [   STABLE   ]
└──────────────────────────────────────────────────────────────────────────────

┌── OSCILLATION CONTROL ───────────────────────────────────────────────────────
│  » Injecting Period-2 Oscillation...
│  Detection                 : [  NOMINAL   ] (OSCILLATION_DETECTED_PERIOD_2)
│  Damping                   : [  WARNING   ] (Returned State in Cycle)
│  » Note: Damping inactive but detection successful.
└──────────────────────────────────────────────────────────────────────────────

┌── STABILITY MATRIX ANALYSIS ─────────────────────────────────────────────────
│  Stable Point ID           : [  NOMINAL   ]
│  Unstable Point ID         : [  NOMINAL   ]
└──────────────────────────────────────────────────────────────────────────────

┌── ZEBA ETHICAL CONSTRAINTS ──────────────────────────────────────────────────
│  Violation Detect          : [  NOMINAL   ]
│  Ethical Projection        : [  NOMINAL   ]
│  Target Mean               : 0.8
│  Actual Mean               : 0.8000
└──────────────────────────────────────────────────────────────────────────────

┌── RECURSIVE LOOP DETECTION (RLDIS) ──────────────────────────────────────────
│  Simple Loop               : [  DETECTED  ]
│  Self-Reference            : [  DETECTED  ]
└──────────────────────────────────────────────────────────────────────────────

┌── EPISTEMIC OPERATORS ───────────────────────────────────────────────────────
│  Knowledge Op              : [  NOMINAL   ]
│  Monitoring Op             : [  NOMINAL   ]
│  Closure Op                : [  NOMINAL   ]
└──────────────────────────────────────────────────────────────────────────────

┌── TEMPORAL MEMORY & GOVERNANCE ──────────────────────────────────────────────
│  Narrative Engine          : [  NOMINAL   ]
│  History Depth             : 1
│  Coherence                 : 1.0000
│  Recursion Detector        : [  DETECTED  ] (SIMPLE_REPETITION)
└──────────────────────────────────────────────────────────────────────────────

┌── RECURSIVE TENSOR FIELD ────────────────────────────────────────────────────
│  Tensor Contraction        : [  NOMINAL   ]
│  Tensor Expansion          : [  NOMINAL   ]
│  Ops Tracked               : 2
└──────────────────────────────────────────────────────────────────────────────

┌── HARMONIC BREATH FIELD ─────────────────────────────────────────────────────
│  » Initializing Breath Cycle...
│  Initial Phase             : INHALE
│  » Stepping simulation...
│  Phases Seen               : 3 
│  Avg Sync Index            : 1.0000 
│  Amp Variance              : 0.007010
│  Breath Dynamics           : [  NOMINAL   ]
└──────────────────────────────────────────────────────────────────────────────

┌── RCF GRAVITY LAYER CONVERGENCE ─────────────────────────────────────────────
│  » Initializing Triaxial Manifold...
│  Metastability             : 0.9944 (Target > 0.8)
│  Coherence Index           : 1.0000 (Target > 0.5)
│  Belief Entropy            : 1.9459 (Variable)
│  Gravity Layer             : [ RESONATING ] (Productive Entropy)
│  » High entropy bound by coherence (Temporal Memory Active).

╔══════════════════════════════════════════════════════════════════════════════╗
║  STATUS: SYSTEM STABLE                                                     ║
║  PASSED: 11/11 (100.0%)                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
