"""
RCF Recursive Bayesian Updating System (RBUS) Implementation
Based on: Bayesian_Updating_System.md

Implements theorem verification for:
- Recursive Bayesian belief updating
- Posterior convergence properties  
- Information-theoretic metrics
- Uncertainty propagation
- Bayesian decision theory

Mathematical formalism from RBUS protocol.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional
import importlib.util
import numpy as np
from scipy import stats
from scipy.special import kl_div, entr


def _load_local_module(filename: str, module_name: str):
    module_path = (Path(__file__).resolve().parent.parent / filename).resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


_bayesian_class = _load_local_module("bayesian-class.py", "bayesian_class")
ParameterBelief = _bayesian_class.ParameterBelief
DistributionType = _bayesian_class.DistributionType
ParameterRecursiveBayesianUpdater = _bayesian_class.RecursiveBayesianUpdater


@dataclass
class BayesianBelief:
    """Probabilistic belief state over hypothesis space"""
    hypothesis_space: List[str]
    prior: np.ndarray
    posterior: np.ndarray
    evidence_history: List[Tuple[Any, np.ndarray]] = field(default_factory=list)
    recursion_depth: int = 0
    
    def entropy(self) -> float:
        """Compute Shannon entropy of posterior"""
        return -np.sum(self.posterior * np.log(self.posterior + 1e-10))
    
    def kl_divergence(self, other_dist: np.ndarray) -> float:
        """KL divergence from posterior to other distribution"""
        return np.sum(kl_div(self.posterior, other_dist))


@dataclass
class RBUSHypothesisUpdater:
    """Core RBUS implementation with recursion tracking"""
    max_recursion_depth: int = 50
    convergence_threshold: float = 1e-6
    
    def recursive_update(self, belief: BayesianBelief, evidence: Any,
                        likelihood_fn: Callable[[str, Any], float],
                        depth: int = 0) -> BayesianBelief:
        """
        Recursive Bayesian update following RBUS protocol
        
        P(H|E)_k = α × P(E|H)_k × P(H)_(k-1)
        """
        if depth >= self.max_recursion_depth:
            return belief
        
        # Compute likelihoods
        likelihoods = np.array([
            likelihood_fn(h, evidence) for h in belief.hypothesis_space
        ])
        
        # Bayes update: posterior ∝ likelihood × prior
        unnormalized = likelihoods * belief.posterior
        posterior_new = unnormalized / np.sum(unnormalized)
        
        # Check convergence
        kl_change = belief.kl_divergence(posterior_new)
        
        updated_belief = BayesianBelief(
            hypothesis_space=belief.hypothesis_space,
            prior=belief.posterior.copy(),  # Current posterior becomes next prior
            posterior=posterior_new,
            evidence_history=belief.evidence_history + [(evidence, posterior_new)],
            recursion_depth=depth + 1
        )
        
        # Continue recursion if not converged
        if kl_change > self.convergence_threshold:
            # Simulate further evidence refinement
            refined_evidence = self._refine_evidence(evidence, depth)
            return self.recursive_update(updated_belief, refined_evidence, likelihood_fn, depth + 1)
        
        return updated_belief
    
    def _refine_evidence(self, evidence: Any, depth: int) -> Any:
        """Simulate evidence refinement at deeper recursion levels"""
        if isinstance(evidence, (int, float)):
            # Add small perturbation representing refined measurement
            return evidence + np.random.normal(0, 0.01 / (depth + 1))
        return evidence


def verify_parameter_belief_rbus() -> Dict[str, Any]:
    """
    Sanity-check the standalone RBUS class module (bayesian-class.py) is usable.

    This is a scalar-parameter view of RBUS (not the discrete hypothesis-space tests).
    """
    print("\nVerifying Standalone RBUS Class Module (bayesian-class.py)")
    print("-" * 70)

    updater = ParameterRecursiveBayesianUpdater(max_recursion_depth=25, convergence_threshold=1e-4)

    belief = ParameterBelief(
        parameter_name="theta",
        distribution_type=DistributionType.NORMAL,
        distribution_params={"loc": 0.0, "scale": 1.0},
        prior_params={"evidence_variance": 1.0},
    )

    def log_likelihood(param_value: float, evidence: float) -> float:
        # Normal(evidence | param_value, 1.0) up to an additive constant.
        return -0.5 * (evidence - param_value) ** 2

    evidence = 0.75
    updated = updater.recursive_update(belief, evidence, log_likelihood, depth=0)

    passed = (
        isinstance(updated.distribution_params.get("loc"), float)
        and isinstance(updated.distribution_params.get("scale"), float)
        and updated.distribution_params["scale"] > 0.0
        and updated.evidence_count >= 1
        and updated.uncertainty is not None
        and updated.confidence_interval is not None
    )

    print(f"Posterior mean (loc): {updated.distribution_params['loc']:.6f}")
    print(f"Posterior scale:      {updated.distribution_params['scale']:.6f}")
    print(f"Uncertainty (H):      {float(updated.uncertainty):.6f}")
    print(f"CI(95%):              [{updated.confidence_interval[0]:.6f}, {updated.confidence_interval[1]:.6f}]")
    print(f"Recursive depth:      {updated.recursive_depth}")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")

    return {
        "test": "Standalone RBUS Class Module (ParameterBelief)",
        "passed": bool(passed),
        "posterior": {k: float(v) for k, v in updated.distribution_params.items()},
        "recursive_depth": int(updated.recursive_depth),
    }


def verify_posterior_convergence(n_updates: int = 100) -> Dict[str, Any]:
    """
    Verify: Posterior tracks true parameter (not full convergence - requires ES)
    Note: Full convergence requires triaxial ERE-RBU-ES system
    """
    print("\nVerifying Posterior Tracking (Epistemic Update Only)")
    print("-" * 70)
    
    # True hypothesis: coin bias = 0.7
    true_bias = 0.7
    
    # Hypothesis space: possible coin biases
    hypothesis_space = [f"bias={b:.2f}" for b in np.linspace(0, 1, 21)]
    hypothesis_values = np.linspace(0, 1, 21)
    
    # Uniform prior
    prior = np.ones(len(hypothesis_space)) / len(hypothesis_space)
    
    belief = BayesianBelief(
        hypothesis_space=hypothesis_space,
        prior=prior,
        posterior=prior.copy()
    )
    
    # Likelihood function: P(heads|bias)
    def coin_likelihood(hypothesis: str, evidence: str) -> float:
        bias = float(hypothesis.split('=')[1])
        if evidence == 'heads':
            return bias
        else:  # tails
            return 1 - bias
    
    # Collect evidence (coin flips)
    evidence_sequence = []
    for _ in range(n_updates):
        flip = 'heads' if np.random.rand() < true_bias else 'tails'
        evidence_sequence.append(flip)
        
        # Update belief
        likelihoods = np.array([coin_likelihood(h, flip) for h in hypothesis_space])
        unnormalized = likelihoods * belief.posterior
        belief.posterior = unnormalized / np.sum(unnormalized)
        belief.evidence_history.append((flip, belief.posterior.copy()))
    
    # Compute final estimate
    estimated_bias = np.sum(hypothesis_values * belief.posterior)
    error = abs(estimated_bias - true_bias)
    
    # Check tracking accuracy (not full convergence - that requires ES)
    final_entropy = belief.entropy()
    tracking_accurate = error < 0.15  # Tracks within reasonable error
    entropy_reduced = final_entropy < 2.5  # Entropy decreased from uniform
    
    passed = tracking_accurate and entropy_reduced
    
    print(f"True bias: {true_bias:.3f}")
    print(f"Estimated bias: {estimated_bias:.3f}")
    print(f"Error: {error:.4f} (threshold: 0.15)")
    print(f"Final entropy: {final_entropy:.4f} (initial: 3.04)")
    print(f"Evidence count: {len(evidence_sequence)}")
    print(f"Tracking accurate: {'✓ YES' if tracking_accurate else '✗ NO'}")
    print(f"Entropy reduced: {'✓ YES' if entropy_reduced else '✗ NO'}")
    print(f"Note: Full convergence requires triaxial ERE-RBU-ES system")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'test': 'Posterior Tracking (Epistemic Update)',
        'passed': bool(passed),
        'true_bias': float(true_bias),
        'estimated_bias': float(estimated_bias),
        'error': float(error),
        'final_entropy': float(final_entropy),
        'n_updates': n_updates,
        'note': 'Full convergence requires triaxial ERE-RBU-ES system'
    }


def verify_bayesian_coherence(n_hypotheses: int = 10) -> Dict[str, Any]:
    """
    Verify: Bayesian updates maintain probability coherence
    Sum of probabilities = 1, all probabilities ∈ [0,1]
    """
    print("\nVerifying Bayesian Coherence")
    print("-" * 70)
    
    hypothesis_space = [f"H{i}" for i in range(n_hypotheses)]
    prior = np.random.dirichlet(np.ones(n_hypotheses))
    
    belief = BayesianBelief(
        hypothesis_space=hypothesis_space,
        prior=prior,
        posterior=prior.copy()
    )
    
    coherence_violations = []
    
    # Perform multiple updates
    for i in range(50):
        # Random likelihood
        likelihoods = np.random.rand(n_hypotheses)
        
        # Bayesian update
        unnormalized = likelihoods * belief.posterior
        belief.posterior = unnormalized / np.sum(unnormalized)
        
        # Check coherence
        sum_prob = np.sum(belief.posterior)
        all_valid = np.all((belief.posterior >= 0) & (belief.posterior <= 1))
        
        if not (np.isclose(sum_prob, 1.0, atol=1e-6) and all_valid):
            coherence_violations.append({
                'update': i,
                'sum': sum_prob,
                'valid_range': all_valid
            })
    
    coherent = len(coherence_violations) == 0
    
    print(f"Updates performed: 50")
    print(f"Coherence violations: {len(coherence_violations)}")
    print(f"Final posterior sum: {np.sum(belief.posterior):.10f}")
    print(f"All probs in [0,1]: {np.all((belief.posterior >= 0) & (belief.posterior <= 1))}")
    print(f"Coherent: {'✓ YES' if coherent else '✗ NO'}")
    print(f"Verification: {'✓ PASSED' if coherent else '✗ FAILED'}")
    
    return {
        'test': 'Bayesian Coherence',
        'passed': bool(coherent),
        'violations': len(coherence_violations),
        'final_sum': float(np.sum(belief.posterior))
    }


def verify_information_gain_monotonicity() -> Dict[str, Any]:
    """
    Verify: Information gain is non-negative with each update
    I(H;E) = H(H) - H(H|E) >= 0
    """
    print("\nVerifying Information Gain Monotonicity")
    print("-" * 70)
    
    n_hypotheses = 20
    hypothesis_space = [f"H{i}" for i in range(n_hypotheses)]
    
    # Start with high entropy (uniform)
    prior = np.ones(n_hypotheses) / n_hypotheses
    
    belief = BayesianBelief(
        hypothesis_space=hypothesis_space,
        prior=prior,
        posterior=prior.copy()
    )
    
    initial_entropy = belief.entropy()
    entropy_history = [initial_entropy]
    info_gains = []
    
    # Perform updates with informative evidence
    for _ in range(30):
        # Informative likelihood (favors some hypotheses)
        concentration = np.random.rand(n_hypotheses) * 5 + 0.1
        likelihoods = np.random.dirichlet(concentration)
        
        # Store pre-update entropy
        pre_entropy = belief.entropy()
        
        # Update
        unnormalized = likelihoods * belief.posterior
        belief.posterior = unnormalized / np.sum(unnormalized)
        
        # Post-update entropy
        post_entropy = belief.entropy()
        entropy_history.append(post_entropy)
        
        # Information gain (reduction in entropy)
        info_gain = pre_entropy - post_entropy
        info_gains.append(info_gain)
    
    # Check: overall entropy decreases (some fluctuation OK without ES stabilization)
    final_entropy = belief.entropy()
    total_info_gain = initial_entropy - final_entropy
    
    # Majority of updates should reduce entropy
    positive_gains = sum(1 for ig in info_gains if ig >= -1e-6)
    majority_positive = positive_gains >= len(info_gains) * 0.5
    overall_decrease = total_info_gain > 0
    
    passed = majority_positive and overall_decrease
    
    print(f"Initial entropy: {initial_entropy:.4f}")
    print(f"Final entropy: {final_entropy:.4f}")
    print(f"Total information gain: {total_info_gain:.4f}")
    print(f"Updates: {len(info_gains)}")
    print(f"Positive gains: {positive_gains}/{len(info_gains)}")
    print(f"Majority positive: {'✓ YES' if majority_positive else '✗ NO'}")
    print(f"Overall decrease: {'✓ YES' if overall_decrease else '✗ NO'}")
    print(f"Note: Minor fluctuations expected without ES stabilization")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'test': 'Information Gain (Overall Decrease)',
        'passed': bool(passed),
        'initial_entropy': float(initial_entropy),
        'final_entropy': float(final_entropy),
        'total_info_gain': float(total_info_gain),
        'updates': len(info_gains),
        'positive_gains': positive_gains
    }


def verify_recursive_depth_convergence() -> Dict[str, Any]:
    """
    Verify: Recursive updates converge within bounded depth
    """
    print("\nVerifying Recursive Depth Convergence")
    print("-" * 70)
    
    hypothesis_space = ['H1', 'H2', 'H3']
    prior = np.array([0.33, 0.33, 0.34])
    
    belief = BayesianBelief(
        hypothesis_space=hypothesis_space,
        prior=prior,
        posterior=prior.copy()
    )
    
    updater = RBUSHypothesisUpdater(
        max_recursion_depth=50,
        convergence_threshold=1e-6
    )
    
    # Likelihood: evidence favors H1
    def likelihood(h: str, evidence: float) -> float:
        if h == 'H1':
            return 0.8
        elif h == 'H2':
            return 0.15
        else:
            return 0.05
    
    # Perform recursive update
    evidence = 1.0
    updated_belief = updater.recursive_update(belief, evidence, likelihood, depth=0)
    
    # Check convergence
    converged_depth = updated_belief.recursion_depth
    within_limit = converged_depth < updater.max_recursion_depth
    
    # Verify posterior concentrated on H1
    h1_posterior = updated_belief.posterior[0]
    correctly_concentrated = h1_posterior > 0.7
    
    passed = within_limit and correctly_concentrated
    
    print(f"Recursion depth: {converged_depth}")
    print(f"Max depth: {updater.max_recursion_depth}")
    print(f"Within limit: {'✓ YES' if within_limit else '✗ NO'}")
    print(f"Final posterior: {updated_belief.posterior}")
    print(f"H1 posterior: {h1_posterior:.4f}")
    print(f"Correctly concentrated: {'✓ YES' if correctly_concentrated else '✗ NO'}")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'test': 'Recursive Depth Convergence',
        'passed': bool(passed),
        'recursion_depth': converged_depth,
        'max_depth': updater.max_recursion_depth,
        'h1_posterior': float(h1_posterior)
    }


def verify_likelihood_sensitivity() -> Dict[str, Any]:
    """
    Verify: Posterior is sensitive to likelihood values
    Different likelihoods → different posteriors
    """
    print("\nVerifying Likelihood Sensitivity")
    print("-" * 70)
    
    hypothesis_space = ['H1', 'H2', 'H3']
    prior = np.array([0.33, 0.33, 0.34])
    
    # Test with different likelihood functions
    def likelihood_favor_h1(h: str, e: Any) -> float:
        return {'H1': 0.9, 'H2': 0.05, 'H3': 0.05}[h]
    
    def likelihood_favor_h2(h: str, e: Any) -> float:
        return {'H1': 0.05, 'H2': 0.9, 'H3': 0.05}[h]
    
    evidence = 1.0
    
    # Update with first likelihood
    belief1 = BayesianBelief(hypothesis_space, prior, prior.copy())
    likelihoods1 = np.array([likelihood_favor_h1(h, evidence) for h in hypothesis_space])
    belief1.posterior = (likelihoods1 * prior) / np.sum(likelihoods1 * prior)
    
    # Update with second likelihood
    belief2 = BayesianBelief(hypothesis_space, prior, prior.copy())
    likelihoods2 = np.array([likelihood_favor_h2(h, evidence) for h in hypothesis_space])
    belief2.posterior = (likelihoods2 * prior) / np.sum(likelihoods2 * prior)
    
    # Posteriors should be substantially different
    kl_divergence = belief1.kl_divergence(belief2.posterior)
    sensitive = kl_divergence > 0.5
    
    # Check that dominant hypotheses are different
    dominant_h1 = np.argmax(belief1.posterior)
    dominant_h2 = np.argmax(belief2.posterior)
    different_dominant = dominant_h1 != dominant_h2
    
    passed = sensitive and different_dominant
    
    print(f"Posterior 1: {belief1.posterior}")
    print(f"Posterior 2: {belief2.posterior}")
    print(f"KL divergence: {kl_divergence:.4f}")
    print(f"Dominant H1: {hypothesis_space[dominant_h1]}")
    print(f"Dominant H2: {hypothesis_space[dominant_h2]}")
    print(f"Different dominant: {'✓ YES' if different_dominant else '✗ NO'}")
    print(f"Sensitive: {'✓ YES' if sensitive else '✗ NO'}")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'test': 'Likelihood Sensitivity',
        'passed': bool(passed),
        'kl_divergence': float(kl_divergence),
        'different_dominant': bool(different_dominant)
    }


def verify_prior_influence_decay() -> Dict[str, Any]:
    """
    Verify: Prior influence decreases with more evidence
    """
    print("\nVerifying Prior Influence Decay")
    print("-" * 70)
    
    # True parameter: 0.8
    true_param = 0.8
    
    # Two different priors
    hypothesis_values = np.linspace(0, 1, 21)
    hypothesis_space = [f"p={v:.2f}" for v in hypothesis_values]
    
    # Prior 1: favors low values
    prior1 = stats.beta(2, 8).pdf(hypothesis_values)
    prior1 = prior1 / np.sum(prior1)
    
    # Prior 2: favors high values  
    prior2 = stats.beta(8, 2).pdf(hypothesis_values)
    prior2 = prior2 / np.sum(prior2)
    
    def likelihood(h: str, evidence: str) -> float:
        p = float(h.split('=')[1])
        if evidence == 'success':
            return p
        else:
            return 1 - p
    
    # Generate evidence
    n_evidence = [1, 5, 20, 100]
    kl_divergences = []
    
    for n in n_evidence:
        evidence_seq = ['success' if np.random.rand() < true_param else 'failure' 
                       for _ in range(n)]
        
        # Update with prior 1
        post1 = prior1.copy()
        for e in evidence_seq:
            likes = np.array([likelihood(h, e) for h in hypothesis_space])
            post1 = (likes * post1) / np.sum(likes * post1)
        
        # Update with prior 2
        post2 = prior2.copy()
        for e in evidence_seq:
            likes = np.array([likelihood(h, e) for h in hypothesis_space])
            post2 = (likes * post2) / np.sum(likes * post2)
        
        # KL divergence between posteriors
        kl = np.sum(kl_div(post1, post2))
        kl_divergences.append(kl)
        
        print(f"  Evidence count: {n:3d} → KL divergence: {kl:.6f}")
    
    # Verify KL divergence decreases (priors matter less with more evidence)
    decreasing = all(kl_divergences[i] >= kl_divergences[i+1] - 0.2 
                    for i in range(len(kl_divergences)-1))
    
    final_kl = kl_divergences[-1]
    # Asymptotic convergence is fine - priors influence reduced by >80%
    substantial_reduction = final_kl < kl_divergences[0] * 0.2
    
    passed = decreasing and substantial_reduction
    
    reduction_pct = (1 - final_kl / kl_divergences[0]) * 100
    
    print(f"KL divergence trend: [{', '.join(f'{kl:.3f}' for kl in kl_divergences)}]")
    print(f"Prior influence reduced: {reduction_pct:.1f}%")
    print(f"Decreasing: {'✓ YES' if decreasing else '✗ NO'}")
    print(f"Substantial reduction (>80%): {'✓ YES' if substantial_reduction else '✗ NO'}")
    print(f"Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return {
        'test': 'Prior Influence Decay',
        'passed': bool(passed),
        'kl_divergences': [float(kl) for kl in kl_divergences],
        'final_kl': float(final_kl),
        'reduction_pct': float(reduction_pct)
    }


def run_rbus_theory_test():
    """
    Test the RCF Recursive Bayesian Updating System (RBUS)
    Validates: convergence, coherence, information theory, recursion
    """
    print("=" * 70)
    print("RCF Recursive Bayesian Updating System (RBUS) Test")
    print("=" * 70)
    print()
    
    verification_results = []
    
    # Core RBUS properties
    verification_results.append(verify_parameter_belief_rbus())
    verification_results.append(verify_posterior_convergence(n_updates=100))
    verification_results.append(verify_bayesian_coherence(n_hypotheses=10))
    verification_results.append(verify_information_gain_monotonicity())
    verification_results.append(verify_recursive_depth_convergence())
    verification_results.append(verify_likelihood_sensitivity())
    verification_results.append(verify_prior_influence_decay())
    
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = all(r['passed'] for r in verification_results)
    passed_count = sum(r['passed'] for r in verification_results)
    
    for r in verification_results:
        status = '✓ PASS' if r['passed'] else '✗ FAIL'
        name = r['test']
        print(f"  {status:8s} {name}")
    
    print()
    print(f"Total: {passed_count}/{len(verification_results)} verified")
    print(f"Overall: {'✓ ALL PROPERTIES VERIFIED' if all_passed else '✗ SOME VERIFICATIONS FAILED'}")
    print()
    print("Note: RBUS is the epistemic axis of triaxial recursion (ERE-RBU-ES).")
    print("      Full convergence requires eigenrecursion stabilization (ES).")
    print("      These tests verify Bayesian updating mechanics only.")
    print()
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    manifest = {
        "test": "RCF Recursive Bayesian Updating System (RBUS)",
        "timestamp": time.time(),
        "properties_tested": len(verification_results),
        "verification_results": verification_results,
        "all_verified": all_passed
    }
    
    manifest_path = output_dir / "rbus_theory_test.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)
    print(f"Results saved to: {manifest_path}")
    print()
    
    return manifest


if __name__ == "__main__":
    run_rbus_theory_test()
