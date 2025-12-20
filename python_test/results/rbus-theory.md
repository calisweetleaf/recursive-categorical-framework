(.venv) PS C:\Users\treyr\Desktop\RCF-v2\python_test> python rbus-theory.py
======================================================================
RCF Recursive Bayesian Updating System (RBUS) Test
======================================================================


Verifying Posterior Tracking (Epistemic Update Only)
----------------------------------------------------------------------
True bias: 0.700
Estimated bias: 0.716
Error: 0.0157 (threshold: 0.15)
Final entropy: 1.2981 (initial: 3.04)
Evidence count: 100
Tracking accurate: ✓ YES
Entropy reduced: ✓ YES
Note: Full convergence requires triaxial ERE-RBU-ES system
Verification: ✓ PASSED

Verifying Bayesian Coherence
----------------------------------------------------------------------
Updates performed: 50
Coherence violations: 0
Final posterior sum: 1.0000000000
All probs in [0,1]: True
Coherent: ✓ YES
Verification: ✓ PASSED

Verifying Information Gain Monotonicity
----------------------------------------------------------------------
Initial entropy: 2.9957
Final entropy: 0.9542
Total information gain: 2.0416
Updates: 30
Positive gains: 22/30
Majority positive: ✓ YES
Overall decrease: ✓ YES
Note: Minor fluctuations expected without ES stabilization
Verification: ✓ PASSED

Verifying Recursive Depth Convergence
----------------------------------------------------------------------
Recursion depth: 10
Max depth: 50
Within limit: ✓ YES
Final posterior: [9.99999946e-01 5.37047498e-08 9.37055097e-13]
H1 posterior: 1.0000
Correctly concentrated: ✓ YES
Verification: ✓ PASSED

Verifying Likelihood Sensitivity
----------------------------------------------------------------------
Posterior 1: [0.89863843 0.04992436 0.05143722]
Posterior 2: [0.04992436 0.89863843 0.05143722]
KL divergence: 2.4531
Dominant H1: H1
Dominant H2: H2
Different dominant: ✓ YES
Sensitive: ✓ YES
Verification: ✓ PASSED

Verifying Prior Influence Decay
----------------------------------------------------------------------
  Evidence count:   1 → KL divergence: 7.914556
  Evidence count:   5 → KL divergence: 5.659776
  Evidence count:  20 → KL divergence: 3.344382
  Evidence count: 100 → KL divergence: 0.950359
KL divergence trend: [7.915, 5.660, 3.344, 0.950]
Prior influence reduced: 88.0%
Decreasing: ✓ YES
Substantial reduction (>80%): ✓ YES
Verification: ✓ PASSED

======================================================================
VERIFICATION SUMMARY
======================================================================
  ✓ PASS   Posterior Tracking (Epistemic Update)
  ✓ PASS   Bayesian Coherence
  ✓ PASS   Information Gain (Overall Decrease)
  ✓ PASS   Recursive Depth Convergence
  ✓ PASS   Likelihood Sensitivity
  ✓ PASS   Prior Influence Decay

Total: 6/6 verified
Overall: ✓ ALL PROPERTIES VERIFIED

Note: RBUS is the epistemic axis of triaxial recursion (ERE-RBU-ES).
      Full convergence requires eigenrecursion stabilization (ES).
      These tests verify Bayesian updating mechanics only.

======================================================================
Test Complete
======================================================================
Results saved to: results\rbus_theory_test.json