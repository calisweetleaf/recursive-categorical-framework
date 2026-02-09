(.venv) PS C:\Users\treyr\Desktop\ursmif> python ursmif_alpha1_stress_benchmark.py --max-kb 100000 --max-outputs 1000
==============================================================================

URSMIF Alpha-1 Stress Benchmark Suite
==============================================================================

Run ID: 20260108_190546
  Timestamp: 2026-01-08 19:05:46
  Max KB: 100000
  Max Time (s): 2.0
  Max Memory (MB): 512.0
==============================================================================

TEST 1: Scaling Saturation
==============================================================================

Case n100_d1
OK n100_d1 -> 2.95 ms
Case n100_d3
OK n100_d3 -> 2.94 ms
Case n100_d5
OK n100_d5 -> 9.10 ms
Case n100_d10
OK n100_d10 -> 21.88 ms
Case n100_d20
OK n100_d20 -> 98.43 ms
Case n100_d50
OK n100_d50 -> 748.31 ms
Case n1000_d1
OK n1000_d1 -> 9.23 ms
Case n1000_d3
OK n1000_d3 -> 10.92 ms
Case n1000_d5
OK n1000_d5 -> 19.77 ms
Case n1000_d10
OK n1000_d10 -> 42.16 ms
Case n1000_d20
OK n1000_d20 -> 129.33 ms
OK n1000_d10 -> 42.16 ms
Case n1000_d20
OK n1000_d20 -> 129.33 ms
Case n1000_d20
OK n1000_d20 -> 129.33 ms
OK n1000_d20 -> 129.33 ms
Case n1000_d50
OK n1000_d50 -> 765.49 ms
Case n10000_d1
OK n10000_d1 -> 89.78 ms
Case n10000_d3
OK n10000_d3 -> 117.48 ms
Case n1000_d50
OK n1000_d50 -> 765.49 ms
Case n10000_d1
OK n10000_d1 -> 89.78 ms
Case n10000_d3
OK n10000_d3 -> 117.48 ms
Case n10000_d5
OK n10000_d5 -> 250.45 ms
Case n10000_d10
OK n10000_d10 -> 437.90 ms
OK n10000_d1 -> 89.78 ms
Case n10000_d3
OK n10000_d3 -> 117.48 ms
Case n10000_d5
OK n10000_d5 -> 250.45 ms
Case n10000_d10
OK n10000_d10 -> 437.90 ms
Case n10000_d5
OK n10000_d5 -> 250.45 ms
Case n10000_d10
OK n10000_d10 -> 437.90 ms
Case n10000_d20
OK n10000_d20 -> 1272.60 ms
Case n10000_d50
WARN n10000_d50 -> limit time_limit
Case n10000_d20
OK n10000_d20 -> 1272.60 ms
Case n10000_d50
WARN n10000_d50 -> limit time_limit
Case n100000_d1
Case n100000_d1
OK n100000_d1 -> 1248.70 ms
Case n100000_d3
Case n100000_d3
OK n100000_d3 -> 1735.14 ms
Case n100000_d5
WARN n100000_d5 -> limit time_limit
Case n100000_d10
WARN n100000_d10 -> limit time_limit
Case n100000_d20
WARN n100000_d20 -> limit time_limit
Case n100000_d50
WARN n100000_d50 -> limit time_limit
Case n250000_d1
OK n250000_d1 -> 1652.57 ms
Case n250000_d3
OK n250000_d3 -> 1903.00 ms
Case n250000_d5
OK n250000_d5 -> 2288.53 ms
Case n250000_d10
WARN n250000_d10 -> limit time_limit
Case n250000_d20
WARN n250000_d20 -> limit time_limit
Case n250000_d50
WARN n250000_d50 -> limit time_limit
Case n500000_d1
OK n500000_d1 -> 694.79 ms
Case n500000_d3
OK n500000_d3 -> 1212.55 ms
Case n500000_d5
OK n500000_d5 -> 1931.07 ms
Case n500000_d10
WARN n500000_d10 -> limit time_limit
Case n500000_d20
WARN n500000_d20 -> limit time_limit
Case n500000_d50
WARN n500000_d50 -> limit time_limit
Case n1000000_d1
OK n1000000_d1 -> 862.46 ms
Case n1000000_d3
OK n1000000_d3 -> 1308.39 ms
Case n1000000_d5
OK n1000000_d5 -> 1927.80 ms
Case n1000000_d10
WARN n1000000_d10 -> limit time_limit
Case n1000000_d20
WARN n1000000_d20 -> limit time_limit
Case n1000000_d50
WARN n1000000_d50 -> limit time_limit
==============================================================================

TEST 2: Recursion Storm
==============================================================================

Case depth_5
OK depth_5 -> 146.50 ms
Case depth_10
OK depth_10 -> 214.66 ms
Case depth_25
OK depth_25 -> 826.19 ms
Case depth_50
WARN depth_50 -> limit time_limit
Case depth_75
WARN depth_75 -> limit time_limit
Case depth_100
WARN depth_100 -> limit time_limit
==============================================================================

TEST 3: Contradiction Cascade
==============================================================================

Case kb1000_r0
OK kb1000_r0 -> 8.36 ms
Case kb1000_r10
OK kb1000_r10 -> 6.52 ms
Case kb1000_r30
OK kb1000_r30 -> 7.88 ms
Case kb1000_r60
OK kb1000_r60 -> 6.61 ms
Case kb1000_r90
OK kb1000_r90 -> 5.35 ms
Case kb5000_r0
OK kb5000_r0 -> 39.16 ms
Case kb5000_r10
OK kb5000_r10 -> 38.48 ms
Case kb5000_r30
OK kb5000_r30 -> 58.58 ms
Case kb5000_r60
OK kb5000_r60 -> 30.45 ms
Case kb5000_r90
OK kb5000_r90 -> 26.74 ms
Case kb20000_r0
OK kb20000_r0 -> 143.90 ms
Case kb20000_r10
OK kb20000_r10 -> 162.38 ms
Case kb20000_r30
OK kb20000_r30 -> 153.70 ms
Case kb20000_r60
OK kb20000_r60 -> 131.99 ms
Case kb20000_r90
OK kb20000_r90 -> 116.33 ms
Case kb50000_r0
OK kb50000_r0 -> 382.63 ms
Case kb50000_r10
OK kb50000_r10 -> 398.86 ms
Case kb50000_r30
OK kb50000_r30 -> 343.79 ms
Case kb50000_r60
OK kb50000_r60 -> 306.99 ms
Case kb50000_r90
OK kb50000_r90 -> 352.51 ms
==============================================================================

TEST 4: Entropy Flood
==============================================================================

Case entropy_10
OK entropy_10 -> entropy 3.3219
Case entropy_100
OK entropy_100 -> entropy 6.6439
Case entropy_500
OK entropy_500 -> entropy 8.9658
Case entropy_1000
OK entropy_1000 -> entropy 9.9658
Case entropy_2500
OK entropy_2500 -> entropy 11.2877
Case entropy_5000
OK entropy_5000 -> entropy 12.2877
==============================================================================

TEST 5: Pattern Saturation
==============================================================================

Case iterations_2000
WARN iterations_2000 -> limit time_limit
==============================================================================

TEST 6: State Fuzzing
==============================================================================

WARN fuzz_300 -> limit time_limit
==============================================================================

Generating Output Files
==============================================================================

OK JSON manifest: results\stress_manifest_20260108_190546.json
OK Markdown report: results\stress_report_20260108_190546.md
OK Event log: results\stress_event_log_20260108_190546.json
==============================================================================

Stress Benchmark Complete
==============================================================================

OK Total Cases: 76
