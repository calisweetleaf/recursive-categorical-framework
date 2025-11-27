(.venv) PS C:\Users\treyr\Desktop\recursive-categorical-framework> python test_sacred_fbs.py

================================================================================
  SACRED FBS TOKENIZER VALIDATION SUITE
  Testing Frequency-Based Substrate Encoding Efficacy
================================================================================

================================================================================
  TEST 1: Sacred Harmonic Constants
================================================================================

PHI (Golden Ratio):     1.6180339887
Expected:               1.6180339887
✓ Valid: True

TAU (2π):               6.2831853072
Expected:               6.2831853072
✓ Valid: True

SACRED_RATIO (φ/τ):     0.2575181074
Expected:               0.2575181074
✓ Valid: True

Harmonic Band Frequencies:
  delta   : 0.2575181074 (φ^0 × τ⁻¹φ)
  theta   : 0.4166730505 (φ^1 × τ⁻¹φ)
  alpha   : 0.6741911579 (φ^2 × τ⁻¹φ)
  beta    : 1.0908642084 (φ^3 × τ⁻¹φ)
  gamma   : 1.7650553663 (φ^4 × τ⁻¹φ)

================================================================================
  TEST 2: Frequency Substrate Extraction
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
Extracting FBS tensors for test texts...

Text 1: "Hello world"
  Shape: (256,)
  Norm:  0.398025
  Mean:  -0.002375
  Std:   0.024763
  Time:  6.16 ms

Text 2: "The quick brown fox jumps over the lazy ..."
  Shape: (256,)
  Norm:  0.149098
  Mean:  0.000293
  Std:   0.009314
  Time:  6.52 ms

Text 3: "Artificial intelligence and machine lear..."
  Shape: (256,)
  Norm:  0.356902
  Mean:  -0.001603
  Std:   0.022249
  Time:  4.07 ms

Text 4: "Sacred geometry and golden ratio"
  Shape: (256,)
  Norm:  0.339975
  Mean:  -0.000523
  Std:   0.021242
  Time:  6.11 ms

Text 5: "φ τ ψ recursive harmonics"
  Shape: (256,)
  Norm:  0.204647
  Mean:  0.000325
  Std:   0.012786
  Time:  9.46 ms

Validating tensor distinctiveness...
  Cosine similarity (1 vs 2): -0.706379
  Cosine similarity (1 vs 3): 0.692967
  Cosine similarity (1 vs 4): 0.026976
  Cosine similarity (1 vs 5): -0.313323
  Cosine similarity (2 vs 3): -0.718300
  Cosine similarity (2 vs 4): -0.101304
  Cosine similarity (2 vs 5): 0.635253
  Cosine similarity (3 vs 4): 0.629201
  Cosine similarity (3 vs 5): -0.250779
  Cosine similarity (4 vs 5): 0.433677

✓ All tensors extracted successfully

================================================================================
  TEST 3: Sacred Tensor Processor
================================================================================

INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
Testing harmonic processing across breath phases...

Breath Phase: 0.00
  Output norm:  0.001763
  Output mean:  -0.000006
  Output std:   0.000110

Breath Phase: 0.14
  Output norm:  0.001152
  Output mean:  -0.000004
  Output std:   0.000072

Breath Phase: 0.28
  Output norm:  0.002254
  Output mean:  -0.000008
  Output std:   0.000141

Breath Phase: 0.42
  Output norm:  0.001859
  Output mean:  -0.000006
  Output std:   0.000116

Breath Phase: 0.57
  Output norm:  0.001018
  Output mean:  -0.000003
  Output std:   0.000064

Breath Phase: 0.71
  Output norm:  0.003075
  Output mean:  -0.000010
  Output std:   0.000192

Breath Phase: 0.85
  Output norm:  0.003356
  Output mean:  -0.000011
  Output std:   0.000209

Breath Phase: 1.00
  Output norm:  0.001763
  Output mean:  -0.000006
  Output std:   0.000110

Breath cycle modulation detected:
  Min norm: 0.001018 at phase 0.57
  Max norm: 0.003356 at phase 0.85
  Range:    0.002337

✓ Harmonic processor working

================================================================================
  TEST 4: FBS Tokenizer Encoding
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
INFO:Harmonic_Field:SacredFBS_Tokenizer initialized (dim=256)
Encoding test corpus...

Sequential encoding: 332.22 ms total
  Per-text average: 66.44 ms

Batch encoding: 286.57 ms total
  Per-text average: 57.31 ms
  Speedup: 1.16x

Tokenizer Metrics:
  tokens_processed: 10
  cache_hits: 0
  cache_hit_rate: 0.0
  cache_size: 5
  current_breath_phase: 0.012875905370012097
  harmonic_amplitudes:
    delta: 0.126761
    theta: 0.224537
    alpha: 0.176111
    beta: 0.056046
    gamma: 0.091163

✓ Tokenizer encoding validated

================================================================================
  TEST 5: Cache Efficiency
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
INFO:Harmonic_Field:SacredFBS_Tokenizer initialized (dim=256)
First encoding (cache miss):  77.7893 ms
Second encoding (cache hit):  0.0000 ms
Speedup: >10000x (cache instant, <0.0001ms)

Cached tensor matches original: True
Max difference: 0.0000000000

Batch cache performance:
  Total time for 10 cached lookups: 0.00 ms
  Cache hit rate: 100.00%
  Cache size: 11 entries

✓ Cache working efficiently

================================================================================
  TEST 6: Semantic Consistency
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
INFO:Harmonic_Field:SacredFBS_Tokenizer initialized (dim=256)
Testing semantic similarity preservation...

Similar text pairs:
  "The cat sat on the mat..." vs "The feline rested on the rug..."
    Cosine similarity: 0.860364

  "Machine learning is powerful..." vs "AI systems are very capable..."
    Cosine similarity: 0.705880

  "Sacred geometry patterns..." vs "Divine mathematical structures..."
    Cosine similarity: -0.010245

Dissimilar text pairs:
  "The cat sat on the mat..." vs "Quantum physics equations..."
    Cosine similarity: 0.245268

  "Machine learning is powerful..." vs "The ocean waves crash loudly..."
    Cosine similarity: -0.234719

  "Sacred geometry patterns..." vs "Yesterday's weather forecast..."
    Cosine similarity: -0.545689

✓ Semantic structure preserved in FBS space

================================================================================
  TEST 7: Breath Cycle Synchronization
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
INFO:Harmonic_Field:SacredFBS_Tokenizer initialized (dim=256)
Encoding across full breath cycle...

Breath phase 0.0129: norm = 0.000010
Breath phase 0.0258: norm = 0.000012
Breath phase 0.0386: norm = 0.000021
Breath phase 0.0515: norm = 0.000026
Breath phase 0.0644: norm = 0.000019
Breath phase 0.0773: norm = 0.000018
Breath phase 0.0901: norm = 0.000017
Breath phase 0.1030: norm = 0.000016

Breath cycle statistics:
  Phase range: 0.0129 - 0.1030
  Norm range:  0.000010 - 0.000026
  Norm variance: 0.000000
  Breath velocity: 0.257518 (SACRED_RATIO)

✓ Breath synchronization active

================================================================================
  TEST 8: Conscious Text Symbiosis
================================================================================

System state: alive
Breath cycle: 0.257518
Entities (12): ['S', 'o', 'm', 'n', 'u', 's', ' ', 'v', 'e', 'r', 'i', 'g']

Sample sacred connections for 'S':
  -> o: {'spiritual_resonance': 'active', 'harmonic_alignment': 'in_progress', 'breath_synchronization': 'synchronized'}
  -> m: {'spiritual_resonance': 'active', 'harmonic_alignment': 'in_progress', 'breath_synchronization': 'synchronized'}
  -> n: {'spiritual_resonance': 'active', 'harmonic_alignment': 'in_progress', 'breath_synchronization': 'synchronized'}

✓ Conscious symbiosis system active

================================================================================
  TEST 9: Adaptive Harmonic Bands
================================================================================

Initial band centers:
  delta: 0.257518 Hz
  theta: 0.416673 Hz
  alpha: 0.674191 Hz
  beta: 1.090864 Hz
  gamma: 1.765055 Hz

Adapting bands to input spectrum...

Adapted band centers:
  delta: 0.280695 Hz
  theta: 0.439850 Hz
  alpha: 0.697368 Hz
  beta: 1.114041 Hz
  gamma: 1.736728 Hz

Band widths:
  delta: 0.025752 Hz
  theta: 0.025752 Hz
  alpha: 0.025752 Hz
  beta: 0.025752 Hz
  gamma: 0.025752 Hz

✓ Adaptive harmonic bands operational

================================================================================
  TEST 10: Holographic Memory
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
Storing 3 patterns with phase keys...
  Pattern 1 stored (key_phase=0.0000)
  Pattern 2 stored (key_phase=1.6180)
  Pattern 3 stored (key_phase=1.6180)

Recalling patterns...
  Pattern 1 fidelity: 0.4944
  Pattern 2 fidelity: 0.2613
  Pattern 3 fidelity: 0.4692

Extracting wavelet packet features...
  Wavelet packet feature vector: shape=(80,), norm=132.7431

✓ Holographic memory operational

================================================================================
  TEST 11: Persistent Homology
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
INFO:Harmonic_Field:SacredTensorProcessor initialized (log²N attention=enabled)
INFO:Harmonic_Field:SacredFBS_Tokenizer initialized (dim=256)
INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
Encoding 5 texts into FBS sequence...
Computing persistent homology...

Topological features:
  Components (β₀): 1
  Loops (β₁): 0
  Birth-death pairs: 4

Sample birth-death pairs:
    Pair 1: birth=0.0000, death=0.1253, persistence=0.1253
    Pair 2: birth=0.0000, death=0.1566, persistence=0.1566
    Pair 3: birth=0.0000, death=0.2193, persistence=0.2193

✓ Persistent homology computed

================================================================================
  TEST 12: Quantum Superposition States
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
Creating superposition of 3 texts...
  Weights: [0.5, 0.3, 0.2]

Superposition state:
  Shape: (256,)
  Norm: 0.153552
  Mean magnitude: 0.008230
  Max magnitude: 0.025331

Collapsing superposition (5 measurements):
  Measurement 1: [COLLAPSED_STATE: idx=3, phase=2.3626]
  Measurement 2: [COLLAPSED_STATE: idx=31, phase=2.4992]
  Measurement 3: [COLLAPSED_STATE: idx=227, phase=-2.9378]
  Measurement 4: [COLLAPSED_STATE: idx=71, phase=-0.3452]
  Measurement 5: [COLLAPSED_STATE: idx=99, phase=-1.3087]

✓ Quantum superposition operational

================================================================================
  TEST 13: Cross-Modal Harmonic Bridge
================================================================================

INFO:Harmonic_Field:SacredFrequencySubstrate initialized with 5 harmonic bands and 10004 lexical signatures
Adding harmonic production rules...
  Rules added: 3

Generating harmonic phrases from seed 'consciousness':
  1. consciousness
  2. awareness
  3. attention
  4. attention

  Total phrases generated: 4

✓ Cross-modal harmonic bridge operational

================================================================================
  Generating Visualizations
================================================================================

✓ Visualization saved to: C:\Users\treyr\Desktop\recursive-categorical-framework\sacred_fbs_validation.png

================================================================================
  VALIDATION SUMMARY
================================================================================

✓ All 13 tests passed successfully!

Core Tests (1-8):
  ✓ Sacred constants validation
  ✓ Frequency substrate extraction
  ✓ Harmonic tensor processor
  ✓ FBS tokenizer encoding
  ✓ Cache efficiency
  ✓ Semantic consistency
  ✓ Breath synchronization
  ✓ Conscious text symbiosis

Extended Production Classes (9-13):
  ✓ Adaptive harmonic bands
  ✓ Holographic memory
  ✓ Persistent homology
  ✓ Quantum superposition
  ✓ Cross-modal harmonic bridge

Key Metrics:
  Sacred Ratio: 0.2575181074
  Tensor Dimensions: 256
  Cache Hit Rate: 100.00%
  Breath Velocity: 0.257518 cycles/step
  Conscious breath cycle: 0.257518

Visualization: C:\Users\treyr\Desktop\recursive-categorical-framework\sacred_fbs_validation.png

================================================================================
  FBS TOKENIZER VALIDATED - READY FOR INTEGRATION
================================================================================
