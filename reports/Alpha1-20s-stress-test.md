(.venv) PS C:\Users\treyr\Desktop\ursmif> python ursmif_alpha1_stress_benchmark.py --max-kb 100000 --max-outputs 1000 --max-time-s 20
==============================================================================

URSMIF Alpha-1 Stress Benchmark Suite
==============================================================================

Run ID: 20260108_195843
  Timestamp: 2026-01-08 19:58:43
  Max KB: 100000
  Max Time (s): 20.0
  Max Memory (MB): 512.0
==============================================================================

TEST 1: Scaling Saturation
==============================================================================

Case n100_d1
OK n100_d1 -> 3.00 ms
Case n100_d3
OK n100_d3 -> 6.28 ms
Case n100_d5
OK n100_d5 -> 12.15 ms
Case n100_d10
OK n100_d10 -> 29.05 ms
Case n100_d20
OK n100_d20 -> 93.79 ms
Case n100_d50
OK n100_d50 -> 713.06 ms
Case n1000_d1
OK n1000_d1 -> 10.45 ms
Case n1000_d3
OK n1000_d3 -> 13.20 ms
Case n1000_d5
OK n1000_d5 -> 19.75 ms
Case n1000_d10
OK n1000_d10 -> 44.39 ms
Case n1000_d20
OK n1000_d20 -> 110.34 ms
Case n1000_d50
OK n1000_d50 -> 751.22 ms
Case n10000_d1
OK n10000_d1 -> 84.86 ms
Case n10000_d3
OK n10000_d3 -> 127.21 ms
Case n10000_d5
OK n10000_d5 -> 196.01 ms
Case n10000_d10
OK n10000_d10 -> 472.34 ms
Case n10000_d20
OK n10000_d20 -> 1082.99 ms
Case n10000_d50
OK n10000_d50 -> 4548.24 ms
Case n100000_d1
OK n100000_d1 -> 1247.48 ms
Case n100000_d3
OK n100000_d3 -> 1242.96 ms
Case n100000_d5
OK n100000_d5 -> 2182.42 ms
Case n100000_d10
OK n100000_d10 -> 6466.84 ms
Case n100000_d20
OK n100000_d20 -> 12227.35 ms
Case n100000_d50
WARN n100000_d50 -> limit time_limit
Case n250000_d1
OK n250000_d1 -> 674.32 ms
Case n250000_d3
OK n250000_d3 -> 877.14 ms
Case n250000_d5
OK n250000_d5 -> 1511.62 ms
Case n250000_d10
OK n250000_d10 -> 4346.64 ms
Case n250000_d20
OK n250000_d20 -> 13182.89 ms
Case n250000_d50
WARN n250000_d50 -> limit time_limit
Case n500000_d1
OK n500000_d1 -> 789.10 ms
Case n500000_d3
OK n500000_d3 -> 1117.43 ms
Case n500000_d5
OK n500000_d5 -> 2246.22 ms
Case n500000_d10
OK n500000_d10 -> 6922.49 ms
Case n500000_d20
OK n500000_d20 -> 13322.14 ms
Case n500000_d50
WARN n500000_d50 -> limit time_limit
Case n1000000_d1
OK n1000000_d1 -> 794.19 ms
Case n1000000_d3
OK n1000000_d3 -> 1029.66 ms
Case n1000000_d5
OK n1000000_d5 -> 1640.57 ms
Case n1000000_d10
OK n1000000_d10 -> 4714.51 ms
Case n1000000_d20
OK n1000000_d20 -> 12949.74 ms
Case n1000000_d50
WARN n1000000_d50 -> limit time_limit
==============================================================================

TEST 2: Recursion Storm
==============================================================================

Case depth_5
OK depth_5 -> 92.38 ms
Case depth_10
OK depth_10 -> 218.83 ms
Case depth_25
OK depth_25 -> 726.86 ms
Case depth_50
OK depth_50 -> 2585.75 ms
Case depth_75
OK depth_75 -> 5354.08 ms
Case depth_100
OK depth_100 -> 11202.83 ms
==============================================================================

TEST 3: Contradiction Cascade
==============================================================================

Case kb1000_r0
OK kb1000_r0 -> 6.08 ms
Case kb1000_r10
OK kb1000_r10 -> 5.76 ms
Case kb1000_r30
OK kb1000_r30 -> 7.51 ms
Case kb1000_r60
OK kb1000_r60 -> 8.40 ms
Case kb1000_r90
OK kb1000_r90 -> 3.71 ms
Case kb5000_r0
OK kb5000_r0 -> 36.57 ms
Case kb5000_r10
OK kb5000_r10 -> 27.49 ms
Case kb5000_r30
OK kb5000_r30 -> 24.32 ms
Case kb5000_r60
OK kb5000_r60 -> 23.27 ms
Case kb5000_r90
OK kb5000_r90 -> 20.09 ms
Case kb20000_r0
OK kb20000_r0 -> 126.31 ms
Case kb20000_r10
OK kb20000_r10 -> 122.53 ms
Case kb20000_r30
OK kb20000_r30 -> 113.97 ms
Case kb20000_r60
OK kb20000_r60 -> 97.89 ms
Case kb20000_r90
OK kb20000_r90 -> 83.71 ms
Case kb50000_r0
OK kb50000_r0 -> 320.05 ms
Case kb50000_r10
OK kb50000_r10 -> 313.22 ms
Case kb50000_r30
OK kb50000_r30 -> 285.39 ms
Case kb50000_r60
OK kb50000_r60 -> 273.10 ms
Case kb50000_r90
OK kb50000_r90 -> 209.88 ms
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

OK JSON manifest: results\stress_manifest_20260108_195843.json
OK Markdown report: results\stress_report_20260108_195843.md
OK Event log: results\stress_event_log_20260108_195843.json
==============================================================================

Stress Benchmark Complete
==============================================================================

OK Total Cases: 76
