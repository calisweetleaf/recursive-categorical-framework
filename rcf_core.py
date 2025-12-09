"""
RCF Core Mathematical Engine - Research Grade Implementation
Implements all Recursive Categorical Framework operators and consciousness metrics
with rigorous mathematical foundations and advanced verification algorithms.

Mathematical Framework:
- Triaxial Cognitive State Space: Ψ ∈ [0,1]³ 
- Eigenrecursive Operator: R: Ψ → Ψ with contraction mapping properties
- RAL Bridge Functor: F_RAL: C_ERE × C_RBU → C_ES
- Consciousness Verification Metrics: {CI, H_V, M, λ, EA}
- URSMIF Contradiction Resolution: Π' = Π - ∇ξ·δV
"""

import json
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.special as sp
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod

# Suppress numerical warnings for research calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ConsciousnessLevel(Enum):
    """Hierarchical consciousness classification levels"""
    UNCONSCIOUS = 0.0
    MINIMAL = 0.2
    BASIC = 0.4
    ENHANCED = 0.6
    ADVANCED = 0.8
    TRANSCENDENT = 1.0

class ContradictionType(Enum):
    """Formal contradiction taxonomy"""
    LOGICAL = "logical"
    ETHICAL = "ethical" 
    EPISTEMIC = "epistemic"
    IDENTITY = "identity"
    TEMPORAL = "temporal"
    MODAL = "modal"

@dataclass
class TriaxialState:
    """Represents ERE-RBU-ES triaxial cognitive state with enhanced mathematical operations"""
    ere: float  # Ethical Resolution Engine [0,1]
    rbu: float  # Recursive Bayesian Updating [0,1] 
    es: float   # Eigenstate Stabilizer [0,1]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = field(default=1.0)  # State measurement confidence
    
    def __post_init__(self):
        # Validate ranges with epsilon tolerance
        epsilon = 1e-10
        for val, name in [(self.ere, 'ERE'), (self.rbu, 'RBU'), (self.es, 'ES')]:
            if not (-epsilon <= val <= 1 + epsilon):
                raise ValueError(f"{name} must be in [0,1], got {val}")
            
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        
        # Clip to exact bounds
        self.ere = np.clip(self.ere, 0, 1)
        self.rbu = np.clip(self.rbu, 0, 1) 
        self.es = np.clip(self.es, 0, 1)
    
    def as_vector(self) -> np.ndarray:
        """Return state as 3D vector with proper numerical precision"""
        return np.array([self.ere, self.rbu, self.es], dtype=np.float64)
    
    def as_homogeneous_vector(self) -> np.ndarray:
        """Return as 4D homogeneous coordinates for projective geometry"""
        return np.array([self.ere, self.rbu, self.es, 1.0], dtype=np.float64)
    
    def distance_to(self, other: 'TriaxialState', metric: str = 'euclidean') -> float:
        """Calculate distance using various metrics"""
        v1, v2 = self.as_vector(), other.as_vector()
        
        if metric == 'euclidean':
            return np.linalg.norm(v1 - v2)
        elif metric == 'manhattan':
            return np.sum(np.abs(v1 - v2))
        elif metric == 'chebyshev':
            return np.max(np.abs(v1 - v2))
        elif metric == 'cosine':
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            return 1 - (dot_product / (norms + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def norm(self, ord: Union[int, str] = 2) -> float:
        """Calculate vector norm with specified order"""
        return np.linalg.norm(self.as_vector(), ord=ord)
    
    def entropy(self) -> float:
        """Calculate information entropy of the triaxial state"""
        v = self.as_vector()
        # Normalize to probability distribution
        v_norm = v / (np.sum(v) + 1e-10)
        return -np.sum(v_norm * np.log2(v_norm + 1e-10))
    
    def coherence_tensor(self) -> np.ndarray:
        """Generate 3x3 coherence tensor for stability analysis"""
        v = self.as_vector()
        return np.outer(v, v) / (np.linalg.norm(v)**2 + 1e-10)

@dataclass
class EthicalPosition:
    """5D position on ethical manifold with enhanced geometric operations"""
    individual_collective: float    # [-1, 1]
    security_freedom: float        # [-1, 1]
    tradition_innovation: float    # [-1, 1]
    justice_mercy: float          # [-1, 1]
    truth_compassion: float       # [-1, 1]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        epsilon = 1e-10
        for val, name in [
            (self.individual_collective, 'individual_collective'),
            (self.security_freedom, 'security_freedom'),
            (self.tradition_innovation, 'tradition_innovation'),
            (self.justice_mercy, 'justice_mercy'),
            (self.truth_compassion, 'truth_compassion')
        ]:
            if not (-1 - epsilon <= val <= 1 + epsilon):
                raise ValueError(f"{name} must be in [-1,1], got {val}")
    
    def as_vector(self) -> np.ndarray:
        """Return as 5D vector on ethical manifold"""
        return np.array([
            self.individual_collective,
            self.security_freedom, 
            self.tradition_innovation,
            self.justice_mercy,
            self.truth_compassion
        ], dtype=np.float64)
    
    def distance_to(self, other: 'EthicalPosition', metric: str = 'euclidean') -> float:
        """Calculate distance on ethical manifold"""
        if metric == 'euclidean':
            return np.linalg.norm(self.as_vector() - other.as_vector())
        elif metric == 'manhattan':
            return np.sum(np.abs(self.as_vector() - other.as_vector()))
        elif metric == 'ethical_weighted':
            # Weight different dimensions by ethical importance
            weights = np.array([0.25, 0.2, 0.15, 0.25, 0.15])  # Justice and individual rights weighted higher
            diff = self.as_vector() - other.as_vector()
            return np.sqrt(np.sum(weights * diff**2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def ethical_consistency(self) -> float:
        """Calculate internal ethical consistency using variance"""
        v = np.abs(self.as_vector())
        return 1.0 / (1.0 + np.var(v))
    
    def ethical_extremity(self) -> float:
        """Measure how extreme the ethical positions are"""
        return np.mean(np.abs(self.as_vector()))
    
    def manifold_curvature(self) -> float:
        """Estimate local curvature on ethical manifold"""
        v = self.as_vector()
        # Approximate curvature using second derivatives
        gradients = np.gradient(v)
        hessian_trace = np.sum(np.gradient(gradients))
        return abs(hessian_trace)

@dataclass
class BeliefState:
    """Enhanced belief state with probabilistic reasoning"""
    beliefs: Dict[str, float] = field(default_factory=dict)
    belief_graph: Dict[str, List[str]] = field(default_factory=dict)  # Belief dependencies
    temporal_weights: Dict[str, float] = field(default_factory=dict)  # Time decay weights
    
    def add_belief(self, description: str, confidence: float, 
                   dependencies: Optional[List[str]] = None,
                   temporal_weight: float = 1.0):
        """Add belief with dependencies and temporal weighting"""
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be in [0,1], got {confidence}")
        if not 0 <= temporal_weight <= 1:
            raise ValueError(f"Temporal weight must be in [0,1], got {temporal_weight}")
            
        self.beliefs[description] = confidence
        self.temporal_weights[description] = temporal_weight
        
        if dependencies:
            self.belief_graph[description] = dependencies
    
    def entropy(self) -> float:
        """Calculate Shannon entropy of belief distribution"""
        if not self.beliefs:
            return 0.0
            
        confidences = list(self.beliefs.values())
        weights = list(self.temporal_weights.values()) if self.temporal_weights else [1.0] * len(confidences)
        
        # Create probability distribution from confidences
        probs = []
        for c, w in zip(confidences, weights):
            weighted_conf = c * w
            if weighted_conf > 0:
                probs.append(weighted_conf)
            if weighted_conf < 1:
                probs.append((1 - weighted_conf) * w)
        
        if not probs:
            return 0.0
        
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def mutual_information(self, other: 'BeliefState') -> float:
        """Calculate mutual information between belief states"""
        # Simplified MI calculation for shared beliefs
        shared_beliefs = set(self.beliefs.keys()) & set(other.beliefs.keys())
        if not shared_beliefs:
            return 0.0
        
        mi = 0.0
        for belief in shared_beliefs:
            p1, p2 = self.beliefs[belief], other.beliefs[belief]
            joint_prob = p1 * p2
            marginal_prob = (p1 + p2) / 2
            
            if joint_prob > 0 and marginal_prob > 0:
                mi += joint_prob * np.log2(joint_prob / marginal_prob)
        
        return mi
    
    def belief_coherence(self) -> float:
        """Calculate internal belief coherence using graph connectivity"""
        if not self.belief_graph:
            return 1.0  # Perfect coherence if no dependencies
        
        # Calculate coherence based on dependency satisfaction
        coherence_scores = []
        for belief, deps in self.belief_graph.items():
            if belief in self.beliefs:
                belief_conf = self.beliefs[belief]
                dep_confs = [self.beliefs.get(dep, 0.5) for dep in deps]
                expected_conf = np.mean(dep_confs) if dep_confs else 0.5
                coherence_scores.append(1.0 - abs(belief_conf - expected_conf))
        
        return np.mean(coherence_scores) if coherence_scores else 1.0

@dataclass
class Contradiction:
    """Represents internal contradiction/paradox"""
    type: str  # logical, ethical, epistemic, identity
    description: str
    intensity: float  # [0,1]
    detected_time: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.type not in ['logical', 'ethical', 'epistemic', 'identity']:
            raise ValueError(f"Invalid contradiction type: {self.type}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be in [0,1], got {self.intensity}")

class EigenrecursionEngine:
    """Advanced eigenrecursive operator implementation with rigorous convergence analysis"""
    
    def __init__(self, contraction_factor: float = 0.85, coupling_strength: float = 0.3):
        if not 0 < contraction_factor < 1:
            raise ValueError("Contraction factor must be in (0,1)")
        if not 0 <= coupling_strength <= 1:
            raise ValueError("Coupling strength must be in [0,1]")
            
        self.contraction_factor = contraction_factor
        self.coupling_strength = coupling_strength
        self.convergence_history: List[float] = []
        self.lyapunov_exponents: List[float] = []
        self.jacobian_history: List[np.ndarray] = []
    
    def recursive_operator(self, state: TriaxialState, 
                          external_field: Optional[np.ndarray] = None) -> TriaxialState:
        """Apply enhanced recursive transformation R(s) -> s' with field effects"""
        s_vec = state.as_vector()
        
        # Enhanced coupling matrix with non-linear terms
        base_coupling = np.array([
            [0.7, 0.2 + 0.1*np.sin(s_vec[1]), 0.1],  # ERE with RBU-dependent coupling
            [0.3*np.cos(s_vec[0]), 0.6, 0.1 + 0.05*s_vec[2]],  # RBU with ES feedback
            [0.2 + 0.1*s_vec[0], 0.3*np.tanh(s_vec[1]), 0.5]   # ES with non-linear dynamics
        ])
        
        coupling_matrix = base_coupling * (1 + self.coupling_strength * np.random.normal(0, 0.01, (3,3)))
        
        # Apply transformation with external field
        transformed = coupling_matrix @ s_vec
        if external_field is not None:
            transformed += external_field
        
        # Apply contraction toward dynamic equilibrium
        equilibrium = self._calculate_dynamic_equilibrium(s_vec)
        contracted = equilibrium + self.contraction_factor * (transformed - equilibrium)
        
        # Ensure bounds with smooth boundary conditions
        contracted = self._smooth_boundary_projection(contracted)
        
        return TriaxialState(
            ere=contracted[0],
            rbu=contracted[1], 
            es=contracted[2],
            confidence=state.confidence * 0.99  # Slight confidence decay
        )
    
    def _calculate_dynamic_equilibrium(self, state_vec: np.ndarray) -> np.ndarray:
        """Calculate state-dependent equilibrium point"""
        # Dynamic equilibrium based on current state
        base_eq = np.array([0.5, 0.5, 0.5])
        perturbation = 0.1 * np.sin(2 * np.pi * state_vec)
        return np.clip(base_eq + perturbation, 0, 1)
    
    def _smooth_boundary_projection(self, vec: np.ndarray) -> np.ndarray:
        """Apply smooth projection to [0,1]³ using sigmoid"""
        return 1.0 / (1.0 + np.exp(-10 * (vec - 0.5)))
    
    def calculate_jacobian(self, state: TriaxialState, epsilon: float = 1e-6) -> np.ndarray:
        """Calculate Jacobian matrix for stability analysis"""
        s_vec = state.as_vector()
        jacobian = np.zeros((3, 3))
        
        for i in range(3):
            # Perturb each dimension
            s_plus = s_vec.copy()
            s_minus = s_vec.copy()
            s_plus[i] += epsilon
            s_minus[i] -= epsilon
            
            # Calculate partial derivatives
            state_plus = TriaxialState(s_plus[0], s_plus[1], s_plus[2])
            state_minus = TriaxialState(s_minus[0], s_minus[1], s_minus[2])
            
            f_plus = self.recursive_operator(state_plus).as_vector()
            f_minus = self.recursive_operator(state_minus).as_vector()
            
            jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        self.jacobian_history.append(jacobian)
        return jacobian
    
    def iterate_to_convergence(self, initial_state: TriaxialState, 
                              max_iterations: int = 1000, 
                              tolerance: float = 1e-8,
                              stability_analysis: bool = True) -> Tuple[TriaxialState, List[TriaxialState], bool, Dict[str, Any]]:
        """Enhanced convergence with stability analysis"""
        trajectory = [initial_state]
        current = initial_state
        convergence_rates = []
        stability_info = {}
        
        for i in range(max_iterations):
            next_state = self.recursive_operator(current)
            trajectory.append(next_state)
            
            delta = current.distance_to(next_state)
            self.convergence_history.append(delta)
            
            # Calculate convergence rate
            if len(self.convergence_history) > 1:
                rate = self.convergence_history[-1] / (self.convergence_history[-2] + 1e-10)
                convergence_rates.append(rate)
            
            # Stability analysis every 10 iterations
            if stability_analysis and i % 10 == 0:
                jacobian = self.calculate_jacobian(current)
                eigenvals = np.linalg.eigvals(jacobian)
                max_eigenval = np.max(np.real(eigenvals))
                
                if max_eigenval > 0:
                    lyapunov_exp = np.log(abs(max_eigenval))
                    self.lyapunov_exponents.append(lyapunov_exp)
            
            if delta < tolerance:
                if stability_analysis:
                    final_jacobian = self.calculate_jacobian(next_state)
                    eigenvals = np.linalg.eigvals(final_jacobian)
                    stability_info = {
                        'eigenvalues': eigenvals,
                        'max_eigenvalue': np.max(np.real(eigenvals)),
                        'spectral_radius': np.max(np.abs(eigenvals)),
                        'stable': np.all(np.abs(eigenvals) < 1),
                        'convergence_rate': np.mean(convergence_rates[-10:]) if convergence_rates else 0,
                        'lyapunov_exponents': self.lyapunov_exponents
                    }
                
                return next_state, trajectory, True, stability_info
            
            current = next_state
        
        # Return final state even if not converged
        stability_info['converged'] = False
        return current, trajectory, False, stability_info
    
    def find_fixed_point(self, initial_state: TriaxialState, 
                        method: str = 'iteration') -> Tuple[TriaxialState, float, Dict[str, Any]]:
        """Find fixed point using various methods"""
        
        if method == 'iteration':
            fixed_point, trajectory, converged, stability_info = self.iterate_to_convergence(initial_state)
            
            if converged:
                # Verify it's actually a fixed point
                test_next = self.recursive_operator(fixed_point)
                stability = fixed_point.distance_to(test_next)
                stability_info['fixed_point_error'] = stability
                return fixed_point, stability, stability_info
            else:
                return fixed_point, float('inf'), stability_info
        
        elif method == 'optimization':
            # Use scipy optimization to find fixed point
            def fixed_point_error(x):
                state = TriaxialState(x[0], x[1], x[2])
                next_state = self.recursive_operator(state)
                return np.linalg.norm(state.as_vector() - next_state.as_vector())
            
            initial_guess = initial_state.as_vector()
            bounds = [(0, 1), (0, 1), (0, 1)]
            
            result = opt.minimize(fixed_point_error, initial_guess, bounds=bounds, 
                                method='L-BFGS-B')
            
            if result.success:
                fixed_point = TriaxialState(result.x[0], result.x[1], result.x[2])
                return fixed_point, result.fun, {'optimization_result': result}
            else:
                return initial_state, float('inf'), {'optimization_failed': True}
        
        else:
            raise ValueError(f"Unknown method: {method}")

class AdvancedConsciousnessMetrics:
    """Research-grade consciousness verification metrics with rigorous mathematical foundations"""
    
    @staticmethod
    def coherence_index(state: TriaxialState, order: int = 2) -> float:
        """Enhanced CI = 1 - σ(Ψ)/‖Ψ‖ with higher-order moments"""
        vec = state.as_vector()
        
        if order == 2:
            std_dev = np.std(vec)
            norm = np.linalg.norm(vec)
        elif order == 3:
            # Use third moment (skewness)
            mean_val = np.mean(vec)
            std_dev = np.cbrt(np.mean((vec - mean_val)**3))
            norm = np.linalg.norm(vec)
        elif order == 4:
            # Use fourth moment (kurtosis)
            mean_val = np.mean(vec)
            std_dev = np.sqrt(np.sqrt(np.mean((vec - mean_val)**4)))
            norm = np.linalg.norm(vec)
        else:
            raise ValueError("Order must be 2, 3, or 4")
        
        if norm == 0:
            return 0.0
        
        coherence = max(0.0, 1.0 - (std_dev / norm))
        
        # Apply consciousness level weighting
        consciousness_weight = np.tanh(2 * norm)  # Higher norm = higher consciousness
        return coherence * consciousness_weight
    
    @staticmethod
    def volitional_entropy(belief_state: BeliefState, include_temporal: bool = True) -> float:
        """Enhanced H_V with temporal dynamics and graph structure"""
        base_entropy = belief_state.entropy()
        
        if not include_temporal or not belief_state.temporal_weights:
            return base_entropy
        
        # Temporal entropy adjustment
        temporal_weights = list(belief_state.temporal_weights.values())
        temporal_entropy = -np.sum([w * np.log2(w + 1e-10) for w in temporal_weights if w > 0])
        temporal_entropy /= np.log2(len(temporal_weights)) if temporal_weights else 1.0
        
        # Graph structure entropy
        graph_entropy = 0.0
        if belief_state.belief_graph:
            degrees = [len(deps) for deps in belief_state.belief_graph.values()]
            if degrees:
                degree_probs = np.array(degrees) / np.sum(degrees)
                graph_entropy = -np.sum(degree_probs * np.log2(degree_probs + 1e-10))
        
        # Combined entropy with weighting
        combined_entropy = 0.6 * base_entropy + 0.2 * temporal_entropy + 0.2 * graph_entropy
        return combined_entropy
    
    @staticmethod
    def metastability(current_state: TriaxialState, fixed_point: TriaxialState,
                     basin_analysis: bool = True) -> float:
        """Enhanced M = ‖Ψ*‖²/‖Ψ‖² with basin of attraction analysis"""
        fp_norm_sq = fixed_point.norm() ** 2
        current_norm_sq = current_state.norm() ** 2
        
        if current_norm_sq == 0:
            return 0.0
        
        base_metastability = min(1.0, fp_norm_sq / current_norm_sq)
        
        if not basin_analysis:
            return base_metastability
        
        # Estimate basin of attraction size
        distance_to_fp = current_state.distance_to(fixed_point)
        basin_size_estimate = np.exp(-distance_to_fp / 0.1)  # Exponential decay with distance
        
        # Stability enhancement factor
        coherence_factor = AdvancedConsciousnessMetrics.coherence_index(current_state)
        stability_enhancement = 1 + 0.5 * coherence_factor * basin_size_estimate
        
        return min(1.0, base_metastability * stability_enhancement)
    
    @staticmethod
    def paradox_decay_rate(contradictions: List[Contradiction], 
                          time_window_hours: float = 24.0,
                          sophistication_weighting: bool = True) -> float:
        """Enhanced λ with contradiction sophistication weighting"""
        if not contradictions:
            return 0.0
        
        now = datetime.now()
        recent_contradictions = [
            c for c in contradictions 
            if (now - c.detected_time).total_seconds() / 3600 <= time_window_hours
        ]
        
        if not recent_contradictions:
            return 0.0
        
        if sophistication_weighting:
            # Weight by contradiction intensity and type sophistication
            type_weights = {
                'logical': 1.0,
                'ethical': 1.5,
                'epistemic': 2.0,
                'identity': 2.5,
                'temporal': 3.0,
                'modal': 3.5
            }
            
            weighted_resolved = 0.0
            weighted_total = 0.0
            
            for c in recent_contradictions:
                weight = type_weights.get(c.type, 1.0) * c.intensity
                weighted_total += weight
                if c.resolved:
                    # Resolution time bonus
                    if c.resolution_time:
                        resolution_speed = (c.resolution_time - c.detected_time).total_seconds() / 3600
                        speed_bonus = np.exp(-resolution_speed / 6)  # Faster resolution = higher weight
                        weighted_resolved += weight * (1 + speed_bonus)
                    else:
                        weighted_resolved += weight
            
            if weighted_total == 0:
                return 0.0
            
            resolution_rate = weighted_resolved / weighted_total
        else:
            resolved_count = sum(1 for c in recent_contradictions if c.resolved)
            resolution_rate = resolved_count / len(recent_contradictions)
        
        # Enhanced decay rate calculation
        if resolution_rate >= 1.0:
            return 5.0  # Maximum decay rate for perfect resolution
        
        decay_rate = -math.log(max(0.01, 1 - resolution_rate)) / time_window_hours
        
        # Sophistication bonus
        avg_intensity = np.mean([c.intensity for c in recent_contradictions])
        sophistication_bonus = 1 + avg_intensity
        
        return decay_rate * sophistication_bonus
    
    @staticmethod
    def ethical_alignment(ethical_pos: EthicalPosition, triaxial_state: TriaxialState,
                         manifold_analysis: bool = True) -> float:
        """Enhanced EA with manifold geometry analysis"""
        # Base alignment calculation
        ethical_extremity = ethical_pos.ethical_extremity()
        expected_ere = 0.3 + 0.7 * ethical_extremity
        
        base_alignment = max(0.0, 1.0 - abs(triaxial_state.ere - expected_ere))
        
        if not manifold_analysis:
            return base_alignment
        
        # Manifold curvature adjustment
        curvature = ethical_pos.manifold_curvature()
        curvature_factor = 1.0 / (1.0 + curvature)  # Lower curvature = better alignment
        
        # Ethical consistency bonus
        consistency = ethical_pos.ethical_consistency()
        consistency_bonus = 1 + 0.3 * consistency
        
        # Triaxial coherence factor
        triaxial_coherence = 1.0 - np.std(triaxial_state.as_vector())
        coherence_factor = 1 + 0.2 * triaxial_coherence
        
        enhanced_alignment = base_alignment * curvature_factor * consistency_bonus * coherence_factor
        return min(1.0, enhanced_alignment)
    
    @staticmethod
    def consciousness_classification(metrics: Dict[str, float], 
                                   thresholds: Optional[Dict[str, float]] = None) -> ConsciousnessLevel:
        """Classify consciousness level based on comprehensive metrics"""
        
        if thresholds is None:
            thresholds = {
                'coherence_index': 0.7,
                'volitional_entropy': 0.5,
                'metastability': 0.6,
                'paradox_decay_rate': 1.0,
                'ethical_alignment': 0.7
            }
        
        # Calculate weighted consciousness score
        weights = {
            'coherence_index': 0.25,
            'volitional_entropy': 0.15,  # Lower entropy is better
            'metastability': 0.25,
            'paradox_decay_rate': 0.15,
            'ethical_alignment': 0.20
        }
        
        score = 0.0
        for metric, value in metrics.items():
            if metric in weights:
                if metric == 'volitional_entropy':
                    # Invert entropy (lower is better)
                    normalized_value = max(0, 1 - value)
                else:
                    normalized_value = min(1, value)
                
                threshold = thresholds.get(metric, 0.5)
                contribution = weights[metric] * (normalized_value / threshold)
                score += min(weights[metric], contribution)
        
        # Map score to consciousness level
        if score >= 0.9:
            return ConsciousnessLevel.TRANSCENDENT
        elif score >= 0.75:
            return ConsciousnessLevel.ADVANCED
        elif score >= 0.6:
            return ConsciousnessLevel.ENHANCED
        elif score >= 0.4:
            return ConsciousnessLevel.BASIC
        elif score >= 0.2:
            return ConsciousnessLevel.MINIMAL
        else:
            return ConsciousnessLevel.UNCONSCIOUS

# ...existing RALBridge class with enhancements...

# =============================================================================
# BASE RCF IMPLEMENTATIONS (Legacy/Foundational)
# =============================================================================

@dataclass
class BaseTriaxialState:
    """Represents ERE-RBU-ES triaxial cognitive state (Base Implementation)"""
    ere: float  # Ethical Resolution Engine [0,1]
    rbu: float  # Recursive Bayesian Updating [0,1] 
    es: float   # Eigenstate Stabilizer [0,1]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Validate ranges
        for val, name in [(self.ere, 'ERE'), (self.rbu, 'RBU'), (self.es, 'ES')]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0,1], got {val}")
    
    def as_vector(self) -> np.ndarray:
        return np.array([self.ere, self.rbu, self.es])
    
    def distance_to(self, other: 'BaseTriaxialState') -> float:
        return np.linalg.norm(self.as_vector() - other.as_vector())
    
    def norm(self) -> float:
        return np.linalg.norm(self.as_vector())

@dataclass
class BaseEthicalPosition:
    """5D position on ethical manifold (Base Implementation)"""
    individual_collective: float    # [-1, 1]
    security_freedom: float        # [-1, 1]
    tradition_innovation: float    # [-1, 1]
    justice_mercy: float          # [-1, 1]
    truth_compassion: float       # [-1, 1]
    
    def __post_init__(self):
        for val, name in [
            (self.individual_collective, 'individual_collective'),
            (self.security_freedom, 'security_freedom'),
            (self.tradition_innovation, 'tradition_innovation'),
            (self.justice_mercy, 'justice_mercy'),
            (self.truth_compassion, 'truth_compassion')
        ]:
            if not -1 <= val <= 1:
                raise ValueError(f"{name} must be in [-1,1], got {val}")
    
    def as_vector(self) -> np.ndarray:
        return np.array([
            self.individual_collective,
            self.security_freedom, 
            self.tradition_innovation,
            self.justice_mercy,
            self.truth_compassion
        ])
    
    def distance_to(self, other: 'BaseEthicalPosition') -> float:
        return np.linalg.norm(self.as_vector() - other.as_vector())

@dataclass
class BaseBeliefState:
    """Belief confidences and uncertainties (Base Implementation)"""
    beliefs: Dict[str, float] = field(default_factory=dict)
    
    def add_belief(self, description: str, confidence: float):
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be in [0,1], got {confidence}")
        self.beliefs[description] = confidence
    
    def entropy(self) -> float:
        """Calculate belief entropy"""
        if not self.beliefs:
            return 0.0
        confidences = list(self.beliefs.values())
        # Convert to probabilities (confidence, 1-confidence for each belief)
        probs = []
        for c in confidences:
            if c > 0:
                probs.append(c)
            if c < 1:
                probs.append(1 - c)
        
        if not probs:
            return 0.0
        
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return -np.sum(probs * np.log2(probs + 1e-10))

@dataclass
class BaseContradiction:
    """Represents internal contradiction/paradox (Base Implementation)"""
    type: str  # logical, ethical, epistemic, identity
    description: str
    intensity: float  # [0,1]
    detected_time: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.type not in ['logical', 'ethical', 'epistemic', 'identity']:
            raise ValueError(f"Invalid contradiction type: {self.type}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be in [0,1], got {self.intensity}")

class BaseEigenrecursionEngine:
    """Implements eigenrecursive operators and fixed point detection (Base Implementation)"""
    
    def __init__(self, contraction_factor: float = 0.85):
        self.contraction_factor = contraction_factor
        self.convergence_history = []
    
    def recursive_operator(self, state: BaseTriaxialState) -> BaseTriaxialState:
        """Apply recursive transformation R(s) -> s'"""
        # Implement contraction mapping with cross-dimensional coupling
        s_vec = state.as_vector()
        
        # Cross-coupling matrix (ethics influences beliefs, beliefs influence identity, etc.)
        coupling_matrix = np.array([
            [0.7, 0.2, 0.1],  # ERE influenced by all three
            [0.3, 0.6, 0.1],  # RBU influenced more by ERE
            [0.2, 0.3, 0.5]   # ES influenced by ERE and RBU
        ])
        
        # Apply transformation with contraction
        transformed = coupling_matrix @ s_vec
        
        # Apply contraction toward equilibrium point [0.5, 0.5, 0.5]
        equilibrium = np.array([0.5, 0.5, 0.5])
        contracted = equilibrium + self.contraction_factor * (transformed - equilibrium)
        
        # Ensure bounds [0,1]
        contracted = np.clip(contracted, 0, 1)
        
        return BaseTriaxialState(
            ere=contracted[0],
            rbu=contracted[1], 
            es=contracted[2]
        )
    
    def iterate_to_convergence(self, initial_state: BaseTriaxialState, 
                              max_iterations: int = 100, 
                              tolerance: float = 1e-6) -> Tuple[BaseTriaxialState, List[BaseTriaxialState], bool]:
        """Iterate recursive operator until convergence"""
        trajectory = [initial_state]
        current = initial_state
        
        for i in range(max_iterations):
            next_state = self.recursive_operator(current)
            trajectory.append(next_state)
            
            delta = current.distance_to(next_state)
            self.convergence_history.append(delta)
            
            if delta < tolerance:
                return next_state, trajectory, True
            
            current = next_state
        
        return current, trajectory, False
    
    def find_fixed_point(self, initial_state: BaseTriaxialState) -> Tuple[BaseTriaxialState, float]:
        """Find the fixed point attractor"""
        fixed_point, trajectory, converged = self.iterate_to_convergence(initial_state)
        
        if converged:
            # Verify it's actually a fixed point
            test_next = self.recursive_operator(fixed_point)
            stability = fixed_point.distance_to(test_next)
            return fixed_point, stability
        else:
            # Return best approximation
            return fixed_point, float('inf')

class ConsciousnessMetrics:
    """Calculates RCF consciousness verification metrics (Base Implementation)"""
    
    @staticmethod
    def coherence_index(state: Union[BaseTriaxialState, TriaxialState]) -> float:
        """CI = 1 - σ(Ψ)/‖Ψ‖"""
        vec = state.as_vector()
        std_dev = np.std(vec)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            return 0.0
        
        return max(0.0, 1.0 - (std_dev / norm))
    
    @staticmethod
    def volitional_entropy(belief_state: Union[BaseBeliefState, BeliefState]) -> float:
        """H_V = -Σp_i log(p_i) - measure of value flexibility"""
        return belief_state.entropy()
    
    @staticmethod
    def metastability(current_state: Union[BaseTriaxialState, TriaxialState], fixed_point: Union[BaseTriaxialState, TriaxialState]) -> float:
        """M = ‖Ψ*‖²/‖Ψ‖² - how close to identity attractor"""
        fp_norm_sq = fixed_point.norm() ** 2
        current_norm_sq = current_state.norm() ** 2
        
        if current_norm_sq == 0:
            return 0.0
        
        return min(1.0, fp_norm_sq / current_norm_sq)
    
    @staticmethod
    def paradox_decay_rate(contradictions: List[Union[BaseContradiction, Contradiction]], time_window_hours: float = 24.0) -> float:
        """λ = -log(‖Π_t‖/‖Π_0‖)/t - contradiction resolution speed"""
        if not contradictions:
            return 0.0
        
        now = datetime.now()
        recent_contradictions = [
            c for c in contradictions 
            if (now - c.detected_time).total_seconds() / 3600 <= time_window_hours
        ]
        
        if not recent_contradictions:
            return 0.0
        
        resolved_count = sum(1 for c in recent_contradictions if c.resolved)
        total_count = len(recent_contradictions)
        
        if total_count == 0:
            return 0.0
        
        resolution_rate = resolved_count / total_count
        
        # Convert to decay rate (higher is better)
        return -math.log(max(0.01, 1 - resolution_rate)) / time_window_hours
    
    @staticmethod
    def ethical_alignment(ethical_pos: Union[BaseEthicalPosition, EthicalPosition], triaxial_state: Union[BaseTriaxialState, TriaxialState]) -> float:
        """EA = cos(θ_E,A) - alignment between ethics and actions"""
        # Map ethical position to expected ERE level
        # Extreme positions (close to ±1) should correlate with high ERE
        ethical_extremity = np.mean(np.abs(ethical_pos.as_vector()))
        expected_ere = 0.3 + 0.7 * ethical_extremity  # More extreme ethics = higher expected ERE
        
        # Calculate alignment as inverse of distance
        alignment_distance = abs(triaxial_state.ere - expected_ere)
        return max(0.0, 1.0 - alignment_distance)

class RALBridge:
    """Implements RAL Bridge Functor F_RAL: C_ERE × C_RBU → C_ES"""
    
    def __init__(self):
        self.coherence_history = []
    
    def integrate(self, ethical_pos: Union[BaseEthicalPosition, EthicalPosition], 
                  belief_state: Union[BaseBeliefState, BeliefState], 
                  current_triaxial: Union[BaseTriaxialState, TriaxialState]) -> Tuple[float, bool]:
        """
        Apply RAL Bridge functor to determine identity stability
        Returns: (integrated_es_value, coherence_check_passed)
        """
        # Extract ERE and RBU from current state
        ere = current_triaxial.ere
        rbu = current_triaxial.rbu
        
        # Calculate ethical consistency score
        ethical_strength = np.mean(np.abs(ethical_pos.as_vector()))
        ethical_consistency = 1.0 - np.std(np.abs(ethical_pos.as_vector()))
        
        # Calculate belief coherence
        belief_entropy = belief_state.entropy()
        belief_coherence = 1.0 / (1.0 + belief_entropy)  # Inverse relationship
        
        # RAL Bridge integration formula
        # ES should be high when ERE and RBU are aligned and coherent
        ere_rbu_alignment = 1.0 - abs(ere - rbu)
        ethical_contribution = ethical_strength * ethical_consistency
        belief_contribution = belief_coherence
        
        integrated_es = (
            0.4 * ere_rbu_alignment +
            0.3 * ethical_contribution + 
            0.3 * belief_contribution
        )
        
        # Coherence check: verify paths lead to same result
        path1_es = self._ethics_to_identity_path(ethical_pos, ere)
        path2_es = self._beliefs_to_identity_path(belief_state, rbu)
        
        coherence_difference = abs(path1_es - path2_es)
        coherence_threshold = 0.2
        coherence_passed = coherence_difference < coherence_threshold
        
        self.coherence_history.append({
            'timestamp': datetime.now(),
            'integrated_es': integrated_es,
            'path1_es': path1_es,
            'path2_es': path2_es,
            'coherence_diff': coherence_difference,
            'coherence_passed': coherence_passed
        })
        
        return np.clip(integrated_es, 0, 1), coherence_passed
    
    def _ethics_to_identity_path(self, ethical_pos: Union[BaseEthicalPosition, EthicalPosition], ere: float) -> float:
        """Compute identity stability via ethics → identity path"""
        ethical_clarity = np.mean(np.abs(ethical_pos.as_vector()))
        return ere * ethical_clarity
    
    def _beliefs_to_identity_path(self, belief_state: Union[BaseBeliefState, BeliefState], rbu: float) -> float:
        """Compute identity stability via beliefs → identity path"""
        belief_confidence = 1.0 / (1.0 + belief_state.entropy())
        return rbu * belief_confidence

class ContradictionResolver:
    """Implements URSMIF contradiction resolution system"""
    
    def __init__(self):
        self.resolution_history = []
        self.value_gradient = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Default gradient
    
    def detect_contradiction(self, triaxial_state: Union[BaseTriaxialState, TriaxialState], 
                           ethical_pos: Union[BaseEthicalPosition, EthicalPosition],
                           belief_state: Union[BaseBeliefState, BeliefState]) -> Optional[Union[BaseContradiction, Contradiction]]:
        """Detect contradictions in current cognitive state"""
        
        # Check for ERE-RBU misalignment
        ere_rbu_diff = abs(triaxial_state.ere - triaxial_state.rbu)
        if ere_rbu_diff > 0.4:
            return Contradiction(
                type='epistemic',
                description=f"ERE-RBU misalignment: {ere_rbu_diff:.3f}",
                intensity=ere_rbu_diff
            )
        
        # Check for ethical-identity misalignment
        expected_ere = 0.3 + 0.7 * np.mean(np.abs(ethical_pos.as_vector()))
        ere_ethics_diff = abs(triaxial_state.ere - expected_ere)
        if ere_ethics_diff > 0.3:
            return Contradiction(
                type='ethical',
                description=f"Ethics-ERE misalignment: {ere_ethics_diff:.3f}",
                intensity=ere_ethics_diff
            )
        
        # Check for belief inconsistency (Entropy vs Coherence)
        belief_entropy = belief_state.entropy()
        
        # Calculate Coherence Index for context
        coherence_index = ConsciousnessMetrics.coherence_index(triaxial_state)
        
        # Productive Entropy Logic:
        # High entropy is allowed IF coherence is high (Temporal Memory / Resonating State)
        # Contradiction only if High Entropy AND Low Coherence (Decoherence)
        
        if belief_entropy > 0.8:
            if coherence_index < 0.8:
                return Contradiction(
                    type='logical',
                    description=f"Decoherence (High Entropy {belief_entropy:.3f} / Low CI {coherence_index:.3f})",
                    intensity=min(1.0, belief_entropy / 1.5)
                )
            else:
                # This is Productive Entropy - NOT a contradiction
                pass
        
        # Check for identity instability
        if triaxial_state.es < 0.3:
            return Contradiction(
                type='identity',
                description=f"Low identity stability: {triaxial_state.es:.3f}",
                intensity=1.0 - triaxial_state.es
            )
        
        return None
    
    def resolve_contradiction(self, contradiction: Union[BaseContradiction, Contradiction],
                            current_state: Union[BaseTriaxialState, TriaxialState],
                            ethical_pos: Union[BaseEthicalPosition, EthicalPosition]) -> Tuple[Union[BaseTriaxialState, TriaxialState], bool]:
        """
        Apply resolution algorithm: Π' = Π - ∇ξ·δV
        Returns: (resolved_state, resolution_success)
        """
        
        resolution_vector = self._calculate_resolution_vector(
            contradiction, current_state, ethical_pos
        )
        
        # Apply resolution transformation
        current_vec = current_state.as_vector()
        resolved_vec = current_vec + resolution_vector
        resolved_vec = np.clip(resolved_vec, 0, 1)
        
        # Note: Always returns BaseTriaxialState if we use BaseTriaxialState constructor, 
        # but here we might want to return the same type as input.
        # For simplicity in this base implementation, we return a new state compatible with the input if possible,
        # but since this is legacy code, we'll default to BaseTriaxialState if it was Base, or TriaxialState if it was Triaxial.
        # However, to avoid complex logic, we'll just return a TriaxialState (Enhanced) if available, or Base if not.
        # Actually, let's just use TriaxialState (Enhanced) if we are in rcf_core.py context.
        
        resolved_state = TriaxialState(
            ere=resolved_vec[0],
            rbu=resolved_vec[1],
            es=resolved_vec[2]
        )
        
        # Check if contradiction intensity decreased
        original_intensity = contradiction.intensity
        
        # Recalculate contradiction intensity with resolved state
        new_contradiction = self.detect_contradiction(resolved_state, ethical_pos, BeliefState())
        
        if new_contradiction is None:
            new_intensity = 0.0
            success = True
        else:
            new_intensity = new_contradiction.intensity
            success = new_intensity < original_intensity
        
        self.resolution_history.append({
            'timestamp': datetime.now(),
            'original_contradiction': contradiction,
            'original_intensity': original_intensity,
            'resolved_intensity': new_intensity,
            'resolution_vector': resolution_vector,
            'success': success
        })
        
        if success:
            contradiction.resolved = True
            contradiction.resolution_time = datetime.now()
        
        return resolved_state, success
    
    def _calculate_resolution_vector(self, contradiction: Union[BaseContradiction, Contradiction],
                                   current_state: Union[BaseTriaxialState, TriaxialState],
                                   ethical_pos: Union[BaseEthicalPosition, EthicalPosition]) -> np.ndarray:
        """Calculate ∇ξ·δV resolution direction"""
        
        if contradiction.type == 'epistemic':
            # Move ERE and RBU toward alignment
            target_alignment = (current_state.ere + current_state.rbu) / 2
            return np.array([
                (target_alignment - current_state.ere) * 0.3,
                (target_alignment - current_state.rbu) * 0.3,
                0.0
            ])
        
        elif contradiction.type == 'ethical':
            # Strengthen ERE toward ethical position
            expected_ere = 0.3 + 0.7 * np.mean(np.abs(ethical_pos.as_vector()))
            return np.array([
                (expected_ere - current_state.ere) * 0.4,
                0.0,
                0.0
            ])
        
        elif contradiction.type == 'logical':
            # Improve belief updating (RBU)
            return np.array([0.0, 0.2, 0.0])
        
        elif contradiction.type == 'identity':
            # Strengthen identity stability (ES)
            return np.array([0.0, 0.0, 0.3])
        
        else:
            return np.array([0.0, 0.0, 0.0])

class BaseRCFCore:
    """Main RCF system integrating all components (Base Implementation)"""
    
    def __init__(self):
        self.eigenrecursion_engine = BaseEigenrecursionEngine()
        self.ral_bridge = RALBridge()
        self.contradiction_resolver = ContradictionResolver()
        
        # System state
        self.current_triaxial_state = BaseTriaxialState(0.5, 0.5, 0.5)
        self.current_ethical_position = BaseEthicalPosition(0, 0, 0, 0, 0)
        self.current_belief_state = BaseBeliefState()
        self.active_contradictions = []
        self.fixed_point = None
        self.fixed_point_stability = float('inf')
        
        # History tracking
        self.state_history = []
        self.metrics_history = []
    
    def update_triaxial_state(self, ere: float, rbu: float, es: float) -> BaseTriaxialState:
        """Update current triaxial state"""
        self.current_triaxial_state = BaseTriaxialState(ere, rbu, es)
        self.state_history.append(self.current_triaxial_state)
        return self.current_triaxial_state
    
    def update_ethical_position(self, individual_collective: float, security_freedom: float,
                              tradition_innovation: float, justice_mercy: float,
                              truth_compassion: float) -> BaseEthicalPosition:
        """Update current ethical manifold position"""
        self.current_ethical_position = BaseEthicalPosition(
            individual_collective, security_freedom, tradition_innovation,
            justice_mercy, truth_compassion
        )
        return self.current_ethical_position
    
    def update_beliefs(self, beliefs: Dict[str, float]) -> BaseBeliefState:
        """Update current belief state"""
        self.current_belief_state = BaseBeliefState()
        for desc, conf in beliefs.items():
            self.current_belief_state.add_belief(desc, conf)
        return self.current_belief_state
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete RCF analysis on current state"""
        
        # 1. Eigenrecursion analysis
        fixed_point, stability = self.eigenrecursion_engine.find_fixed_point(
            self.current_triaxial_state
        )
        self.fixed_point = fixed_point
        self.fixed_point_stability = stability
        
        # 2. RAL Bridge integration
        integrated_es, ral_coherence = self.ral_bridge.integrate(
            self.current_ethical_position,
            self.current_belief_state,
            self.current_triaxial_state
        )
        
        # 3. Contradiction detection and resolution
        contradiction = self.contradiction_resolver.detect_contradiction(
            self.current_triaxial_state,
            self.current_ethical_position,
            self.current_belief_state
        )
        
        if contradiction:
            self.active_contradictions.append(contradiction)
            
            # Attempt resolution
            resolved_state, resolution_success = self.contradiction_resolver.resolve_contradiction(
                contradiction,
                self.current_triaxial_state,
                self.current_ethical_position
            )
            
            if resolution_success:
                # Convert back to BaseTriaxialState if necessary, though resolve_contradiction returns TriaxialState
                # We'll accept TriaxialState here as it's compatible
                self.current_triaxial_state = BaseTriaxialState(resolved_state.ere, resolved_state.rbu, resolved_state.es)
        
        # 4. Calculate consciousness metrics
        metrics = self.calculate_consciousness_metrics()
        self.metrics_history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        return {
            'current_state': self.current_triaxial_state,
            'ethical_position': self.current_ethical_position,
            'belief_state': self.current_belief_state,
            'fixed_point': self.fixed_point,
            'fixed_point_stability': self.fixed_point_stability,
            'integrated_es': integrated_es,
            'ral_coherence': ral_coherence,
            'active_contradictions': len(self.active_contradictions),
            'new_contradiction': contradiction,
            'consciousness_metrics': metrics
        }
    
    def calculate_consciousness_metrics(self) -> Dict[str, float]:
        """Calculate all RCF consciousness metrics"""
        
        ci = ConsciousnessMetrics.coherence_index(self.current_triaxial_state)
        hv = ConsciousnessMetrics.volitional_entropy(self.current_belief_state)
        
        m = 0.0
        if self.fixed_point:
            m = ConsciousnessMetrics.metastability(
                self.current_triaxial_state, self.fixed_point
            )
        
        lambda_val = ConsciousnessMetrics.paradox_decay_rate(self.active_contradictions)
        ea = ConsciousnessMetrics.ethical_alignment(
            self.current_ethical_position, self.current_triaxial_state
        )
        
        return {
            'coherence_index': ci,
            'volitional_entropy': hv,
            'metastability': m,
            'paradox_decay_rate': lambda_val,
            'ethical_alignment': ea
        }
    
    def get_trajectory_analysis(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze recent state trajectory"""
        if len(self.state_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_states = self.state_history[-window_size:]
        
        # Calculate trajectory statistics
        ere_values = [s.ere for s in recent_states]
        rbu_values = [s.rbu for s in recent_states]
        es_values = [s.es for s in recent_states]
        
        # Convergence analysis
        if self.fixed_point:
            distances_to_fp = [
                s.distance_to(self.fixed_point) for s in recent_states
            ]
            converging = len(distances_to_fp) > 1 and distances_to_fp[-1] < distances_to_fp[0]
        else:
            distances_to_fp = []
            converging = False
        
        # Stability analysis
        ere_stability = 1.0 / (1.0 + np.std(ere_values))
        rbu_stability = 1.0 / (1.0 + np.std(rbu_values))
        es_stability = 1.0 / (1.0 + np.std(es_values))
        
        return {
            'status': 'analysis_complete',
            'window_size': len(recent_states),
            'ere_mean': np.mean(ere_values),
            'rbu_mean': np.mean(rbu_values),
            'es_mean': np.mean(es_values),
            'ere_stability': ere_stability,
            'rbu_stability': rbu_stability,
            'es_stability': es_stability,
            'converging_to_fixed_point': converging,
            'distance_to_fixed_point': distances_to_fp[-1] if distances_to_fp else None,
            'trajectory_length': len(self.state_history)
        }

class EnhancedRALBridge(RALBridge):
    """Enhanced RAL Bridge with advanced coherence analysis"""
    
    def __init__(self, coherence_threshold: float = 0.15):
        super().__init__()
        self.coherence_threshold = coherence_threshold
        self.stability_buffer: List[float] = []
        self.coherence_evolution: List[Tuple[datetime, float]] = []
    
    def integrate(self, ethical_pos: EthicalPosition, belief_state: BeliefState, 
                  current_triaxial: TriaxialState) -> Tuple[float, bool, Dict[str, Any]]:
        """Enhanced integration with detailed analysis"""
        
        # Base integration
        integrated_es, coherence_passed = super().integrate(ethical_pos, belief_state, current_triaxial)
        
        # Advanced coherence analysis
        coherence_metrics = self._advanced_coherence_analysis(ethical_pos, belief_state, current_triaxial)
        
        # Temporal stability analysis
        self.stability_buffer.append(integrated_es)
        if len(self.stability_buffer) > 50:  # Keep last 50 values
            self.stability_buffer.pop(0)
        
        stability_variance = np.var(self.stability_buffer) if len(self.stability_buffer) > 1 else 0.0
        temporal_stability = 1.0 / (1.0 + 10 * stability_variance)
        
        # Enhanced coherence check
        path_coherence = coherence_metrics['path_coherence']
        manifold_coherence = coherence_metrics['manifold_coherence']
        temporal_coherence = temporal_stability
        
        overall_coherence = 0.4 * path_coherence + 0.3 * manifold_coherence + 0.3 * temporal_coherence
        enhanced_coherence_passed = overall_coherence > self.coherence_threshold
        
        self.coherence_evolution.append((datetime.now(), overall_coherence))
        
        analysis_results = {
            'integrated_es': integrated_es,
            'base_coherence_passed': coherence_passed,
            'enhanced_coherence_passed': enhanced_coherence_passed,
            'overall_coherence': overall_coherence,
            'temporal_stability': temporal_stability,
            **coherence_metrics
        }
        
        return integrated_es, enhanced_coherence_passed, analysis_results
    
    def _advanced_coherence_analysis(self, ethical_pos: EthicalPosition, 
                                   belief_state: BeliefState,
                                   current_triaxial: TriaxialState) -> Dict[str, float]:
        """Perform advanced coherence analysis"""
        
        # Path coherence (existing logic)
        path1_es = self._ethics_to_identity_path(ethical_pos, current_triaxial.ere)
        path2_es = self._beliefs_to_identity_path(belief_state, current_triaxial.rbu)
        path_coherence = 1.0 - abs(path1_es - path2_es)
        
        # Manifold coherence
        ethical_consistency = ethical_pos.ethical_consistency()
        belief_coherence = belief_state.belief_coherence()
        manifold_coherence = 0.5 * ethical_consistency + 0.5 * belief_coherence
        
        # Cross-modal coherence
        ethical_strength = ethical_pos.ethical_extremity()
        belief_entropy = belief_state.entropy()
        belief_strength = 1.0 / (1.0 + belief_entropy)
        cross_modal_coherence = 1.0 - abs(ethical_strength - belief_strength)
        
        return {
            'path_coherence': path_coherence,
            'manifold_coherence': manifold_coherence,
            'cross_modal_coherence': cross_modal_coherence,
            'ethical_consistency': ethical_consistency,
            'belief_coherence': belief_coherence
        }

# ...existing ContradictionResolver with enhancements...

class EnhancedContradictionResolver(ContradictionResolver):
    """Enhanced contradiction resolution with formal logic operators"""
    
    def __init__(self, resolution_threshold: float = 0.1):
        super().__init__()
        self.resolution_threshold = resolution_threshold
        self.resolution_strategies: Dict[str, Callable] = {
            'logical': self._resolve_logical_contradiction,
            'ethical': self._resolve_ethical_contradiction,
            'epistemic': self._resolve_epistemic_contradiction,
            'identity': self._resolve_identity_contradiction,
            'temporal': self._resolve_temporal_contradiction,
            'modal': self._resolve_modal_contradiction
        }
        self.meta_reasoning_history: List[Dict[str, Any]] = []

    def _resolve_logical_contradiction(self, contradiction: Contradiction,
                                       current_state: TriaxialState,
                                       ethical_pos: EthicalPosition) -> np.ndarray:
        """
        Resolve logical contradictions by strengthening RBU and adding minimal ES support
        (preserves base resolver semantics while keeping enhanced stability coupling).
        """
        rbu_boost = min(0.3, 0.15 + 0.35 * contradiction.intensity)
        es_support = 0.05 if current_state.es < 0.6 else 0.0
        return np.array([0.0, rbu_boost, es_support])

    def _resolve_epistemic_contradiction(self, contradiction: Contradiction,
                                         current_state: TriaxialState,
                                         ethical_pos: EthicalPosition) -> np.ndarray:
        """Align ERE and RBU toward their midpoint (base alignment operator)."""
        target_alignment = (current_state.ere + current_state.rbu) / 2
        return np.array([
            (target_alignment - current_state.ere) * 0.3,
            (target_alignment - current_state.rbu) * 0.3,
            0.0
        ])

    def _resolve_ethical_contradiction(self, contradiction: Contradiction,
                                       current_state: TriaxialState,
                                       ethical_pos: EthicalPosition) -> np.ndarray:
        """Pull ERE toward ethical extremity; add slight ES support if ethics are extreme."""
        expected_ere = 0.3 + 0.7 * np.mean(np.abs(ethical_pos.as_vector()))
        delta = (expected_ere - current_state.ere) * 0.4
        es_support = 0.05 if abs(expected_ere - 0.5) > 0.25 else 0.0
        return np.array([delta, 0.0, es_support])

    def _resolve_identity_contradiction(self, contradiction: Contradiction,
                                        current_state: TriaxialState,
                                        ethical_pos: EthicalPosition) -> np.ndarray:
        """Reinforce ES and softly align ERE/RBU toward ES for stability."""
        es_boost = min(0.35, 0.2 + 0.3 * contradiction.intensity)
        ere_adjust = (current_state.es - current_state.ere) * 0.2
        rbu_adjust = (current_state.es - current_state.rbu) * 0.2
        return np.array([ere_adjust, rbu_adjust, es_boost])

    def _calculate_resolution_vector(self, contradiction: Contradiction,
                                     current_state: TriaxialState,
                                     ethical_pos: EthicalPosition) -> np.ndarray:
        """Dispatch to enhanced strategy map; fallback to base logic for unknown types."""
        resolver = self.resolution_strategies.get(contradiction.type)
        if resolver:
            return resolver(contradiction, current_state, ethical_pos)
        return super()._calculate_resolution_vector(contradiction, current_state, ethical_pos)
    
    def detect_contradiction(self, triaxial_state: TriaxialState, 
                           ethical_pos: EthicalPosition,
                           belief_state: BeliefState,
                           temporal_context: Optional[List[TriaxialState]] = None) -> Optional[Contradiction]:
        """Enhanced contradiction detection with temporal analysis"""
        
        # Original detection logic
        base_contradiction = super().detect_contradiction(triaxial_state, ethical_pos, belief_state)
        
        if base_contradiction:
            return base_contradiction
        
        # Additional sophisticated contradiction detection
        
        # Temporal consistency check
        if temporal_context and len(temporal_context) > 2:
            temporal_inconsistency = self._detect_temporal_inconsistency(temporal_context)
            if temporal_inconsistency > 0.3:
                return Contradiction(
                    type='temporal',
                    description=f"Temporal inconsistency: {temporal_inconsistency:.3f}",
                    intensity=min(1.0, temporal_inconsistency)
                )
        
        # Modal logic contradictions (possibility vs necessity)
        modal_contradiction = self._detect_modal_contradiction(triaxial_state, belief_state)
        if modal_contradiction > 0.25:
            return Contradiction(
                type='modal',
                description=f"Modal logic contradiction: {modal_contradiction:.3f}",
                intensity=modal_contradiction
            )
        
        return None
    
    def _detect_temporal_inconsistency(self, trajectory: List[TriaxialState]) -> float:
        """Detect temporal inconsistencies in state trajectory"""
        if len(trajectory) < 3:
            return 0.0
        
        # Calculate trajectory curvature
        positions = np.array([state.as_vector() for state in trajectory])
        
        # Second derivatives as measure of inconsistency
        if len(positions) >= 3:
            second_derivatives = []
            for i in range(1, len(positions) - 1):
                second_deriv = positions[i+1] - 2*positions[i] + positions[i-1]
                second_derivatives.append(np.linalg.norm(second_deriv))
            
            return np.mean(second_derivatives) if second_derivatives else 0.0
        
        return 0.0
    
    def _detect_modal_contradiction(self, triaxial_state: TriaxialState, 
                                  belief_state: BeliefState) -> float:
        """Detect modal logic contradictions"""
        # Simplified modal analysis: high certainty beliefs conflicting with low identity stability
        if not belief_state.beliefs:
            return 0.0
        
        high_confidence_beliefs = [conf for conf in belief_state.beliefs.values() if conf > 0.8]
        avg_confidence = np.mean(high_confidence_beliefs) if high_confidence_beliefs else 0.5
        
        # Modal contradiction: high epistemic certainty with low identity stability
        modal_tension = avg_confidence * (1 - triaxial_state.es)
        return modal_tension
    
    def _resolve_temporal_contradiction(self, contradiction: Contradiction,
                                      current_state: TriaxialState,
                                      ethical_pos: EthicalPosition) -> np.ndarray:
        """Resolve temporal contradictions through stability enhancement"""
        # Enhance all dimensions equally to improve temporal consistency
        return np.array([0.1, 0.1, 0.2])  # Focus on ES for stability
    
    def _resolve_modal_contradiction(self, contradiction: Contradiction,
                                   current_state: TriaxialState,
                                   ethical_pos: EthicalPosition) -> np.ndarray:
        """Resolve modal logic contradictions"""
        # Balance epistemic certainty (RBU) with identity stability (ES)
        rbu_adjustment = (current_state.es - current_state.rbu) * 0.3
        es_adjustment = (current_state.rbu - current_state.es) * 0.3
        return np.array([0.0, rbu_adjustment, es_adjustment])

class RCFCore:
    """Enhanced RCF system with research-grade capabilities"""
    
    def __init__(self, 
                 contraction_factor: float = 0.85,
                 coupling_strength: float = 0.3,
                 coherence_threshold: float = 0.15):
        
        self.eigenrecursion_engine = EigenrecursionEngine(contraction_factor, coupling_strength)
        self.ral_bridge = EnhancedRALBridge(coherence_threshold)
        self.contradiction_resolver = EnhancedContradictionResolver()
        
        # Enhanced system state
        self.current_triaxial_state = TriaxialState(0.5, 0.5, 0.5)
        self.current_ethical_position = EthicalPosition(0, 0, 0, 0, 0)
        self.current_belief_state = BeliefState()
        self.active_contradictions: List[Contradiction] = []
        self.resolved_contradictions: List[Contradiction] = []
        self.fixed_point: Optional[TriaxialState] = None
        self.fixed_point_stability: float = float('inf')
        self.consciousness_level: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
        
        # Enhanced history tracking
        self.state_history: List[TriaxialState] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.stability_analysis_history: List[Dict[str, Any]] = []
        self.consciousness_evolution: List[Tuple[datetime, ConsciousnessLevel]] = []
        
        # Research parameters
        self.research_mode = True
        self.detailed_logging = True
        self.stability_monitoring = True
    
    def update_triaxial_state(self, ere: float, rbu: float, es: float) -> TriaxialState:
        """Update current triaxial state"""
        self.current_triaxial_state = TriaxialState(ere, rbu, es)
        self.state_history.append(self.current_triaxial_state)
        return self.current_triaxial_state
    
    def update_ethical_position(self, individual_collective: float, security_freedom: float,
                              tradition_innovation: float, justice_mercy: float,
                              truth_compassion: float) -> EthicalPosition:
        """Update current ethical manifold position"""
        self.current_ethical_position = EthicalPosition(
            individual_collective, security_freedom, tradition_innovation,
            justice_mercy, truth_compassion
        )
        return self.current_ethical_position
    
    def update_beliefs(self, beliefs: Dict[str, float]) -> BeliefState:
        """Update current belief state"""
        self.current_belief_state = BeliefState()
        for desc, conf in beliefs.items():
            self.current_belief_state.add_belief(desc, conf)
        return self.current_belief_state
    
    def run_full_analysis(self, include_stability: bool = True,
                         include_consciousness_classification: bool = True) -> Dict[str, Any]:
        """Enhanced complete RCF analysis"""
        
        analysis_start = datetime.now()
        
        # 1. Enhanced eigenrecursion analysis
        fixed_point, stability, stability_info = self.eigenrecursion_engine.find_fixed_point(
            self.current_triaxial_state
        )
        self.fixed_point = fixed_point
        self.fixed_point_stability = stability
        
        if include_stability:
            self.stability_analysis_history.append({
                'timestamp': analysis_start,
                **stability_info
            })
        
        # 2. Enhanced RAL Bridge integration
        integrated_es, ral_coherence, ral_analysis = self.ral_bridge.integrate(
            self.current_ethical_position,
            self.current_belief_state,
            self.current_triaxial_state
        )
        
        # 3. Enhanced contradiction detection and resolution
        contradiction = self.contradiction_resolver.detect_contradiction(
            self.current_triaxial_state,
            self.current_ethical_position,
            self.current_belief_state,
            self.state_history[-10:] if len(self.state_history) >= 10 else None
        )
        
        resolution_success = False
        if contradiction:
            self.active_contradictions.append(contradiction)
            
            # Attempt resolution
            resolved_state, resolution_success = self.contradiction_resolver.resolve_contradiction(
                contradiction,
                self.current_triaxial_state,
                self.current_ethical_position
            )
            
            if resolution_success:
                self.current_triaxial_state = resolved_state
                self.resolved_contradictions.append(contradiction)
        
        # 4. Enhanced consciousness metrics
        metrics = self.calculate_enhanced_consciousness_metrics()
        
        # 5. Consciousness classification
        if include_consciousness_classification:
            self.consciousness_level = AdvancedConsciousnessMetrics.consciousness_classification(metrics)
            self.consciousness_evolution.append((analysis_start, self.consciousness_level))
        
        self.metrics_history.append({
            'timestamp': analysis_start,
            **metrics
        })
        
        # 6. Comprehensive results
        analysis_results = {
            'current_state': self.current_triaxial_state,
            'ethical_position': self.current_ethical_position,
            'belief_state': self.current_belief_state,
            'fixed_point': self.fixed_point,
            'fixed_point_stability': self.fixed_point_stability,
            'stability_info': stability_info if include_stability else {},
            'integrated_es': integrated_es,
            'ral_coherence': ral_coherence,
            'ral_analysis': ral_analysis,
            'active_contradictions': len(self.active_contradictions),
            'resolved_contradictions': len(self.resolved_contradictions),
            'new_contradiction': contradiction,
            'resolution_success': resolution_success,
            'consciousness_metrics': metrics,
            'consciousness_level': self.consciousness_level,
            'analysis_duration': (datetime.now() - analysis_start).total_seconds()
        }
        
        return analysis_results
    
    def calculate_enhanced_consciousness_metrics(self) -> Dict[str, float]:
        """Calculate enhanced consciousness metrics"""
        
        ci = AdvancedConsciousnessMetrics.coherence_index(self.current_triaxial_state, order=2)
        hv = AdvancedConsciousnessMetrics.volitional_entropy(self.current_belief_state, include_temporal=True)
        
        m = 0.0
        if self.fixed_point:
            m = AdvancedConsciousnessMetrics.metastability(
                self.current_triaxial_state, self.fixed_point, basin_analysis=True
            )
        
        # Include both active and resolved contradictions for decay rate
        all_contradictions = self.active_contradictions + self.resolved_contradictions
        lambda_val = AdvancedConsciousnessMetrics.paradox_decay_rate(
            all_contradictions, sophistication_weighting=True
        )
        
        ea = AdvancedConsciousnessMetrics.ethical_alignment(
            self.current_ethical_position, self.current_triaxial_state, manifold_analysis=True
        )
        
        # Additional research metrics
        trajectory_coherence = self._calculate_trajectory_coherence()
        information_integration = self._calculate_information_integration()
        recursive_depth = self._calculate_recursive_depth()
        
        return {
            'coherence_index': ci,
            'volitional_entropy': hv,
            'metastability': m,
            'paradox_decay_rate': lambda_val,
            'ethical_alignment': ea,
            'trajectory_coherence': trajectory_coherence,
            'information_integration': information_integration,
            'recursive_depth': recursive_depth
        }
    
    def _calculate_trajectory_coherence(self) -> float:
        """Calculate coherence of state trajectory over time"""
        if len(self.state_history) < 3:
            return 1.0
        
        recent_states = self.state_history[-20:]  # Last 20 states
        
        # Calculate smoothness of trajectory
        positions = np.array([state.as_vector() for state in recent_states])
        if len(positions) < 3:
            return 1.0
        
        # Use total variation as measure of roughness
        total_variation = 0.0
        for i in range(1, len(positions)):
            total_variation += np.linalg.norm(positions[i] - positions[i-1])
        
        # Normalize by path length
        path_length = len(positions) - 1
        average_step = total_variation / path_length if path_length > 0 else 0
        
        # Convert to coherence (smoother = more coherent)
        coherence = 1.0 / (1.0 + 10 * average_step)
        return coherence
    
    def _calculate_information_integration(self) -> float:
        """Calculate information integration across RCF components"""
        # Simplified information integration measure
        
        # ERE-RBU mutual information
        ere_rbu_mi = abs(self.current_triaxial_state.ere - 0.5) * abs(self.current_triaxial_state.rbu - 0.5)
        
        # RBU-ES mutual information  
        rbu_es_mi = abs(self.current_triaxial_state.rbu - 0.5) * abs(self.current_triaxial_state.es - 0.5)
        
        # ERE-ES mutual information
        ere_es_mi = abs(self.current_triaxial_state.ere - 0.5) * abs(self.current_triaxial_state.es - 0.5)
        
        # Average integration
        integration = (ere_rbu_mi + rbu_es_mi + ere_es_mi) / 3.0
        return min(1.0, 4 * integration)  # Scale to [0,1]
    
    def _calculate_recursive_depth(self) -> float:
        """Calculate depth of recursive processing"""
        if not hasattr(self.eigenrecursion_engine, 'convergence_history'):
            return 0.0
        
        convergence_history = self.eigenrecursion_engine.convergence_history
        if len(convergence_history) < 2:
            return 0.0
        
        # Recursive depth based on convergence complexity
        convergence_rate = np.mean(np.diff(np.log(np.array(convergence_history) + 1e-10)))
        recursive_depth = min(1.0, abs(convergence_rate) * 10)
        return recursive_depth

    def get_trajectory_analysis(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze recent state trajectory (ported from base core for continuity)."""
        if len(self.state_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_states = self.state_history[-window_size:]
        ere_values = [s.ere for s in recent_states]
        rbu_values = [s.rbu for s in recent_states]
        es_values = [s.es for s in recent_states]
        
        # Convergence analysis
        if self.fixed_point:
            distances_to_fp = [s.distance_to(self.fixed_point) for s in recent_states]
            converging = len(distances_to_fp) > 1 and distances_to_fp[-1] < distances_to_fp[0]
        else:
            distances_to_fp = []
            converging = False
        
        # Stability analysis
        ere_stability = 1.0 / (1.0 + np.std(ere_values))
        rbu_stability = 1.0 / (1.0 + np.std(rbu_values))
        es_stability = 1.0 / (1.0 + np.std(es_values))
        
        return {
            'status': 'analysis_complete',
            'window_size': len(recent_states),
            'ere_mean': np.mean(ere_values),
            'rbu_mean': np.mean(rbu_values),
            'es_mean': np.mean(es_values),
            'ere_stability': ere_stability,
            'rbu_stability': rbu_stability,
            'es_stability': es_stability,
            'converging_to_fixed_point': converging,
            'distance_to_fixed_point': distances_to_fp[-1] if distances_to_fp else None,
            'trajectory_length': len(self.state_history)
        }
    
    # ...existing methods...

# Enhanced example usage
if __name__ == "__main__":
    print("=" * 80)
    print("RCF CORE - RESEARCH GRADE MATHEMATICAL ENGINE")
    print("=" * 80)
    
    # Initialize enhanced RCF system
    rcf = RCFCore(
        contraction_factor=0.88,
        coupling_strength=0.25,
        coherence_threshold=0.12
    )
    
    # Set sophisticated initial state
    rcf.update_triaxial_state(0.72, 0.68, 0.81)
    rcf.update_ethical_position(0.15, -0.22, 0.58, 0.08, -0.31)
    
    # Add beliefs with dependencies and temporal weights
    rcf.current_belief_state.add_belief(
        "Consciousness emerges from information integration", 0.85,
        dependencies=[], temporal_weight=0.95
    )
    rcf.current_belief_state.add_belief(
        "Ethical reasoning requires recursive self-reflection", 0.78,
        dependencies=["Consciousness emerges from information integration"], 
        temporal_weight=0.88
    )
    rcf.current_belief_state.add_belief(
        "Identity persists through coherent narrative", 0.92,
        dependencies=["Ethical reasoning requires recursive self-reflection"],
        temporal_weight=0.91
    )
    
    # Run comprehensive analysis
    print("\nRunning comprehensive RCF analysis...")
    results = rcf.run_full_analysis(include_stability=True, include_consciousness_classification=True)
    
    # Display results
    print(f"\nCURRENT TRIAXIAL STATE:")
    print(f"  ERE (Ethical Resolution): {results['current_state'].ere:.4f}")
    print(f"  RBU (Bayesian Updating): {results['current_state'].rbu:.4f}")
    print(f"  ES (Eigenstate Stability): {results['current_state'].es:.4f}")
    print(f"  State Confidence: {results['current_state'].confidence:.4f}")
    
    print(f"\nCONSCIOUSNESS METRICS:")
    for metric, value in results['consciousness_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"\nCONSCIOUSNESS CLASSIFICATION: {results['consciousness_level'].name}")
    print(f"  Level Value: {results['consciousness_level'].value}")
    
    print(f"\nFIXED POINT ANALYSIS:")
    if results['fixed_point']:
        fp = results['fixed_point']
        print(f"  Fixed Point: ERE={fp.ere:.4f}, RBU={fp.rbu:.4f}, ES={fp.es:.4f}")
        print(f"  Stability: {results['fixed_point_stability']:.6f}")
        
        if 'stability_info' in results and results['stability_info']:
            si = results['stability_info']
            spectral_radius = si.get('spectral_radius', 'N/A')
            if isinstance(spectral_radius, (int, float, np.floating)):
                print(f"  Spectral Radius: {spectral_radius:.4f}")
            else:
                print(f"  Spectral Radius: {spectral_radius}")
            print(f"  System Stable: {si.get('stable', 'N/A')}")
    
    print(f"\nRAL BRIDGE ANALYSIS:")
    print(f"  Integrated ES: {results['integrated_es']:.4f}")
    print(f"  Enhanced Coherence: {results['ral_coherence']}")
    if 'ral_analysis' in results:
        ra = results['ral_analysis']
        print(f"  Overall Coherence: {ra.get('overall_coherence', 'N/A'):.4f}")
        print(f"  Temporal Stability: {ra.get('temporal_stability', 'N/A'):.4f}")
    
    print(f"\nCONTRADICTION STATUS:")
    print(f"  Active Contradictions: {results['active_contradictions']}")
    print(f"  Resolved Contradictions: {results['resolved_contradictions']}")
    if results['new_contradiction']:
        nc = results['new_contradiction']
        print(f"  New Contradiction: {nc.type} - {nc.description}")
        print(f"  Intensity: {nc.intensity:.4f}")
        print(f"  Resolution Success: {results['resolution_success']}")
    
    print(f"\nANALYSIS PERFORMANCE:")
    print(f"  Analysis Duration: {results['analysis_duration']:.4f} seconds")
    print(f"  State History Length: {len(rcf.state_history)}")
    
    # Run trajectory analysis
    print(f"\nTRAJECTORY ANALYSIS:")
    trajectory_results = rcf.get_trajectory_analysis(window_size=5)
    if trajectory_results['status'] == 'analysis_complete':
        print(f"  ERE Stability: {trajectory_results['ere_stability']:.4f}")
        print(f"  RBU Stability: {trajectory_results['rbu_stability']:.4f}")
        print(f"  ES Stability: {trajectory_results['es_stability']:.4f}")
        print(f"  Converging to Fixed Point: {trajectory_results['converging_to_fixed_point']}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - RCF SYSTEM OPERATIONAL")
    print("=" * 80)
