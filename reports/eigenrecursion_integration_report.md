# Eigenrecursion Integration Report

- Generated: 2025-11-27T23:26:20.858638Z
- Overall Status: FAILURE
- Total Tests: 25 (passed=23, failed=1, errors=1, skipped=0)

| Stage | Module | Tests | Passed | Failed | Errors | Skipped | Duration (s) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eigenrecursion Algorithm | `rcf_integration.eigenrecursion_algorithm` | 7 | 6 | 1 | 0 | 0 | 2.092 | FAILED |
| Eigenrecursive Operations | `rcf_integration.eigenrecursive_operations` | 8 | 7 | 0 | 1 | 0 | 0.058 | FAILED |
| Governance Framework | `rcf_integration.governance_framework` | 3 | 3 | 0 | 0 | 0 | 0.009 | PASSED |
| Epistemic Operators | `rcf_integration.eigenrecursive_operations` | 2 | 2 | 0 | 0 | 0 | 0.000 | PASSED |
| Formula Validation | `rcf_integration.governance_framework` | 5 | 5 | 0 | 0 | 0 | 0.002 | PASSED |

## Eigenrecursion Algorithm

Validates recursive convergence, RLDIS monitoring, and algorithmic resilience.

```
test_basic_eigenrecursion_convergence (__main__.TestEigenrecursionAlgorithm.test_basic_eigenrecursion_convergence)
Test basic eigenrecursion finds fixed point. ... FAIL
test_bayesian_intervention_selector (__main__.TestEigenrecursionAlgorithm.test_bayesian_intervention_selector)
Test Bayesian intervention selection framework. ... ok
test_gradient_contradiction_resolver (__main__.TestEigenrecursionAlgorithm.test_gradient_contradiction_resolver)
Test gradient-based contradiction resolution. ... ok
test_meta_cognition_amplifier (__main__.TestEigenrecursionAlgorithm.test_meta_cognition_amplifier)
Test meta-cognition amplification. ... ok
test_pattern_analysis_entropy_detection (__main__.TestEigenrecursionAlgorithm.test_pattern_analysis_entropy_detection)
Test entropy-based pattern detection. ... ok
test_rldis_pattern_detection (__main__.TestEigenrecursionAlgorithm.test_rldis_pattern_detection)
Test RLDIS pattern detection system. ... ok
test_semantic_analysis_topological (__main__.TestEigenrecursionAlgorithm.test_semantic_analysis_topological)
Test topological phase space analysis. ... ok

======================================================================
FAIL: test_basic_eigenrecursion_convergence (__main__.TestEigenrecursionAlgorithm.test_basic_eigenrecursion_convergence)
Test basic eigenrecursion finds fixed point.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\test_eigenrecursion_integration.py", line 127, in test_basic_eigenrecursion_convergence
    self.assertEqual(result['status'], ConvergenceStatus.CONVERGED)
AssertionError: <ConvergenceStatus.ERROR: 'Error occurred during recursion'> != <ConvergenceStatus.CONVERGED: 'Convergence achieved within tolerance'>

----------------------------------------------------------------------
Ran 7 tests in 2.092s

FAILED (failures=1)
```

## Eigenrecursive Operations

Exercises eigenstate convergence, contradiction tension, and operator stacks.

```
test_consciousness_eigenoperator (__main__.TestEigenrecursiveOperations.test_consciousness_eigenoperator)
Test consciousness eigenoperator with full metrics. ... ok
test_contradiction_tension_engine (__main__.TestEigenrecursiveOperations.test_contradiction_tension_engine)
Test contradiction tension computation and minimization. ... ok
test_eigenstate_convergence_engine (__main__.TestEigenrecursiveOperations.test_eigenstate_convergence_engine)
Test full eigenstate convergence with all integrated components. ... ERROR
test_free_energy_minimizer (__main__.TestEigenrecursiveOperations.test_free_energy_minimizer)
Test free energy minimization. ... ok
test_information_geometry (__main__.TestEigenrecursiveOperations.test_information_geometry)
Test information geometry and natural gradient descent. ... ok
test_information_theoretic_detector (__main__.TestEigenrecursiveOperations.test_information_theoretic_detector)
Test information-theoretic pattern detection. ... ok
test_quantum_cognition_model (__main__.TestEigenrecursiveOperations.test_quantum_cognition_model)
Test quantum cognition model for identity representation. ... ok
test_topological_analyzer (__main__.TestEigenrecursiveOperations.test_topological_analyzer)
Test topological phase space analysis. ... ok

======================================================================
ERROR: test_eigenstate_convergence_engine (__main__.TestEigenrecursiveOperations.test_eigenstate_convergence_engine)
Test full eigenstate convergence with all integrated components.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\test_eigenrecursion_integration.py", line 398, in test_eigenstate_convergence_engine
    result = engine.converge_to_eigenstate(initial_state, operator)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\rcf_integration\eigenrecursive_operations.py", line 2029, in converge_to_eigenstate
    if self.recursion_tracer and len(self.recursion_tracer.trace) >= 3:
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'EigenrecursionTracer' object has no attribute 'trace'

----------------------------------------------------------------------
Ran 8 tests in 0.058s

FAILED (errors=1)
```

## Governance Framework

Ensures homeostatic control, narrative identity, and policy metrics behave stably.

```
test_governance_framework (__main__.TestGovernanceFramework.test_governance_framework)
Test governance framework for human-AI interaction. ... ok
test_homeostatic_controller (__main__.TestGovernanceFramework.test_homeostatic_controller)
Test homeostatic control theory implementation. ... ok
test_narrative_identity_engine (__main__.TestGovernanceFramework.test_narrative_identity_engine)
Test narrative identity formation. ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.009s

OK
```

## Epistemic Operators

Checks epistemic inference, modal logic operators, and belief revision dynamics.

```
test_epistemic_operators (__main__.TestEpistemicOperators.test_epistemic_operators)
Test epistemic operators: K_a φ → φ, M_a φ → K_a(K_a φ ∨ ¬K_a φ) ... ok
test_modal_logic_operators (__main__.TestEpistemicOperators.test_modal_logic_operators)
Test modal logic operators for recursive reasoning. ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK
```

## Formula Validation

Confirms analytical formulas align with published theoretical specifications.

```
test_autonomy_authority_ratio_formula (__main__.TestFormulaValidation.test_autonomy_authority_ratio_formula)
Validate autonomy-authority ratio: AAR = DA/HA ... ok
test_entropy_formula (__main__.TestFormulaValidation.test_entropy_formula)
Validate entropy formula: H(O) = -Σ_i p(o_i) log p(o_i) ... ok
test_lyapunov_exponent_formula (__main__.TestFormulaValidation.test_lyapunov_exponent_formula)
Validate Lyapunov exponent: λ = lim_{t→∞} (1/t) ln(|δΦ(t)|/|δΦ(0)|) ... ok
test_mutual_information_formula (__main__.TestFormulaValidation.test_mutual_information_formula)
Validate mutual information: I(O_t; O_{t-1}) = H(O_t) + H(O_{t-1}) - H(O_t, O_{t-1}) ... ok
test_transparency_obligation_formula (__main__.TestFormulaValidation.test_transparency_obligation_formula)
Validate transparency obligation: TO(DA) = k·DA^α ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.002s

OK
```
