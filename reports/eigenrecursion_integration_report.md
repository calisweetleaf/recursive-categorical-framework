# Eigenrecursion Integration Report
- Generated: 2025-11-14T04:40:43.547017Z
- Overall Status: FAILURE
- Total Tests: 25 (passed=24, failed=0, errors=1, skipped=0)

| Stage | Module | Tests | Passed | Failed | Errors | Skipped | Duration (s) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eigenrecursion Algorithm | `rcf_integration.eigenrecursion_algorithm` | 7 | 7 | 0 | 0 | 0 | 0.507 | PASSED |
| Eigenrecursive Operations | `rcf_integration.eigenrecursive_operations` | 8 | 7 | 0 | 1 | 0 | 0.211 | FAILED |
| Governance Framework | `rcf_integration.governance_framework` | 3 | 3 | 0 | 0 | 0 | 0.012 | PASSED |
| Epistemic Operators | `rcf_integration.eigenrecursive_operations` | 2 | 2 | 0 | 0 | 0 | 0.001 | PASSED |
| Formula Validation | `rcf_integration.governance_framework` | 5 | 5 | 0 | 0 | 0 | 0.003 | PASSED |

## Eigenrecursion Algorithm
Validates recursive convergence, RLDIS monitoring, and algorithmic resilience.

```
test_basic_eigenrecursion_convergence (__main__.TestEigenrecursionAlgorithm.test_basic_eigenrecursion_convergence)
Test basic eigenrecursion finds fixed point. ... ok
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

----------------------------------------------------------------------
Ran 7 tests in 0.507s

OK
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
  File "c:\Users\treyr\Desktop\recursive-categorical-framework\test_eigenrecursion_integration.py", line 398, in test_eigenstate_convergence_engine
    result = engine.converge_to_eigenstate(initial_state, operator)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\treyr\Desktop\recursive-categorical-framework\rcf_integration\eigenrecursive_operations.py", line 1952, in converge_to_eigenstate
    next_state = self.contradiction_tension_engine.minimize_tension_gradient_descent(next_state)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\treyr\Desktop\recursive-categorical-framework\rcf_integration\eigenrecursive_operations.py", line 394, in minimize_tension_gradient_descent
    tension.backward()
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\.venv\Lib\site-packages\torch\_tensor.py", line 625, in backward
    torch.autograd.backward(
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\.venv\Lib\site-packages\torch\autograd\__init__.py", line 347, in backward
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\treyr\Desktop\recursive-categorical-framework\.venv\Lib\site-packages\torch\autograd\__init__.py", line 207, in _make_grads
    raise RuntimeError(msg)
RuntimeError: grad can be implicitly created only for real scalar outputs but got torch.complex64

----------------------------------------------------------------------
Ran 8 tests in 0.211s

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
Ran 3 tests in 0.012s

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
Ran 2 tests in 0.001s

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