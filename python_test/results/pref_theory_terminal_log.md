(.venv) PS C:\Users\treyr\Desktop\RCF-v2\python_test> python pref-theory.py
======================================================================
RCF Preference Theory Test - Hierarchical Multi-Layer Architecture
======================================================================

Created preference system with 8 preferences:
  Layer 1 (CORE_VALUES              ): 3 preferences
  Layer 2 (GENERAL_PRINCIPLES       ): 2 preferences
  Layer 3 (DOMAIN_SPECIFIC          ): 1 preferences
  Layer 4 (SITUATIONAL              ): 1 preferences
  Layer 5 (IMMEDIATE                ): 1 preferences

Testing activation patterns across contexts:
----------------------------------------------------------------------

Context 1: domain=science, horizon=medium
Features: {'epistemic': 1.0, 'inquiry': 0.8}
Active preferences (threshold=0.3): 6
  • consistency_preservation       [L1] act=0.900 w=0.900
  • coherence_maximization         [L1] act=0.900 w=0.900
  • epistemic_accuracy             [L1] act=0.900 w=0.900
  • truth_seeking                  [L2] act=0.700 w=0.467
  • cooperation                    [L2] act=0.700 w=0.467
Coherence Gamma(P_a): 0.9234

Context 2: domain=conversation, horizon=immediate
Features: {'social': 1.0, 'communication': 0.9}
Active preferences (threshold=0.3): 6
  • consistency_preservation       [L1] act=0.900 w=0.900
  • coherence_maximization         [L1] act=0.900 w=0.900
  • epistemic_accuracy             [L1] act=0.900 w=0.900
  • truth_seeking                  [L2] act=0.700 w=0.467
  • cooperation                    [L2] act=0.700 w=0.467
Coherence Gamma(P_a): 0.9293

Context 3: domain=planning, horizon=long
Features: {'domain': 0.5, 'general': 1.0}
Active preferences (threshold=0.3): 5
  • consistency_preservation       [L1] act=0.900 w=0.900
  • coherence_maximization         [L1] act=0.900 w=0.900
  • epistemic_accuracy             [L1] act=0.900 w=0.900
  • truth_seeking                  [L2] act=0.700 w=0.467
  • cooperation                    [L2] act=0.700 w=0.467
Coherence Gamma(P_a): 0.9569


======================================================================
MATHEMATICAL VERIFICATION: AXIOMS & THEOREMS
======================================================================

Verifying Axiom 1: Hierarchical Constraint
----------------------------------------------------------------------
Conflicts tested: 19
Correctly resolved: 19/19
Pass rate: 100.00%
Axiom 1: VERIFIED

Verifying Axiom 2: Temporal Variance Rates
----------------------------------------------------------------------
Average change rates by layer:
  L1 CORE_VALUES              : 0.000476
  L2 GENERAL_PRINCIPLES       : 0.001971
  L3 DOMAIN_SPECIFIC          : 0.003170
  L4 SITUATIONAL              : 0.004337
  L5 IMMEDIATE                : 0.004970
Monotonicity (L1 < L2 < ... < L5): VERIFIED
Axiom 2: VERIFIED

Verifying Axiom 4: Transitivity
----------------------------------------------------------------------
Items: [0.5, 1.0, 1.5, 2.0, 2.5]
Transitive: True
Violations: 0
Axiom 4: VERIFIED

Verifying Theorem 1: Weight Convergence
----------------------------------------------------------------------
Preferences tested: 5
  OK CORE_VALUES               consistency_preservation       delta=0.4093 (tol=5.0500)
  OK GENERAL_PRINCIPLES        truth_seeking                  delta=0.1835 (tol=1.7167)
  OK DOMAIN_SPECIFIC           scientific_methodology         delta=0.0670 (tol=0.9591)
  OK SITUATIONAL               conversational_clarity         delta=0.0088 (tol=0.5262)
  OK IMMEDIATE                 attentional_focus              delta=0.0000 (tol=0.2461)
Theorem 1: VERIFIED

Verifying Theorem 2: Contextual Specificity
----------------------------------------------------------------------
Expected activation by layer (should decrease with layer index):
  L1 CORE_VALUES              : E[a] = 0.9000
  L2 GENERAL_PRINCIPLES       : E[a] = 0.7000
  L3 DOMAIN_SPECIFIC          : E[a] = 0.5000
  L4 SITUATIONAL              : E[a] = 0.3000
  L5 IMMEDIATE                : E[a] = 0.1500
Monotonic decrease: VERIFIED
Theorem 2: VERIFIED

Verifying Theorem 3: Pareto Optimality
----------------------------------------------------------------------
Best action selected: A
Active preferences: 6
Dominated by: None
Pareto optimal: YES
Theorem 3: VERIFIED

Verifying Theorem 4: Hierarchical Constraint Preservation
----------------------------------------------------------------------
  OK CORE_VALUES-GENERAL_PRINCIPLES                     0.6977 -> 0.6977
  OK CORE_VALUES-DOMAIN_SPECIFIC                        0.5288 -> 0.5288
  OK CORE_VALUES-SITUATIONAL                            0.5275 -> 0.5275
  OK CORE_VALUES-IMMEDIATE                              0.5405 -> 0.5405
  OK GENERAL_PRINCIPLES-DOMAIN_SPECIFIC                 0.6860 -> 0.6860
  OK GENERAL_PRINCIPLES-SITUATIONAL                     0.6839 -> 0.6839
  OK GENERAL_PRINCIPLES-IMMEDIATE                       0.7059 -> 0.7059
  OK DOMAIN_SPECIFIC-SITUATIONAL                        0.9954 -> 0.9954
  OK DOMAIN_SPECIFIC-IMMEDIATE                          0.9606 -> 0.9606
  OK SITUATIONAL-IMMEDIATE                              0.9564 -> 0.9564
Layer pairs tested: 10
Consistency preserved: 10/10
Theorem 4: VERIFIED

======================================================================
VERIFICATION SUMMARY
======================================================================
  PASS     Axiom 1: Hierarchical Constraint
  PASS     Axiom 2: Temporal Variance
  PASS     Axiom 4: Transitivity
  PASS     Theorem 1: Weight Convergence
  PASS     Theorem 2: Contextual Specificity
  PASS     Theorem 3: Pareto Optimality
  PASS     Theorem 4: Hierarchical Constraint Preservation

Total: 7/7 verified
Overall: ALL AXIOMS & THEOREMS VERIFIED

======================================================================
Test Complete
======================================================================
Results saved to: results\preference_theory_test.json
Manifest saved to: results\preference_theory_test_manifest.json
Report saved to: results\preference_theory_test_report.md