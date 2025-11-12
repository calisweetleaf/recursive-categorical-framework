import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any, Tuple, List, Dict, Optional, Union
from enum import Enum
import time
from collections import deque
import warnings
from scipy import linalg
import torch


class ConvergenceStatus(Enum):
    """Enumeration of possible convergence statuses."""
    CONVERGED = "Convergence achieved within tolerance"
    CYCLE_DETECTED = "Cycle detected in state sequence"
    MAX_ITERATIONS_REACHED = "Maximum iterations reached without convergence"
    DIVERGED = "State sequence appears to be diverging"
    ERROR = "Error occurred during recursion"
    NUMERICAL_INSTABILITY = "Numerical instability detected"


class FixedPointType(Enum):
    """Classification of fixed point stability types."""
    ATTRACTIVE = "Attractive fixed point"
    REPULSIVE = "Repulsive fixed point"
    SADDLE = "Saddle point"
    NEUTRAL = "Neutral fixed point (e.g., center)"
    UNKNOWN = "Stability type could not be determined"


# RLDIS (Recursive Loop Detection and Interruption System) Components
class RLDISPatternType(Enum):
    """Classification of detected recursive patterns."""
    SIMPLE_REPETITION = "Simple repetition with minimal variation"
    CONTRADICTION_SPIRAL = "Oscillating contradictory statements"
    SELF_REFERENCE_LOOP = "Increasing self-referential density"
    RESOURCE_CONSUMPTION_ANOMALY = "Exponential resource utilization"
    USER_FRUSTRATION_CASCADE = "Repeated clarification requests"
    EIGENSTATE_LOCK = "Stuck in false eigenstate attractor"
    UNKNOWN_PATTERN = "Unclassified recursive pattern"


class RLDISSeverityLevel(Enum):
    """Severity levels for recursive patterns."""
    LOW = "Low severity - monitoring only"
    MODERATE = "Moderate severity - intervention recommended"
    HIGH = "High severity - immediate intervention required"
    CRITICAL = "Critical severity - emergency interruption protocols"


class RLDISInterventionPriority(Enum):
    """Priority levels for intervention actions."""
    LEVEL_1 = "Critical - immediate termination"
    LEVEL_2 = "High - structured interruption"
    LEVEL_3 = "Standard - pattern breaking"
    LEVEL_4 = "Monitoring - observation only"


class RLDISInterventionStatus(Enum):
    """Status of intervention attempts."""
    NOT_TRIGGERED = "No intervention triggered"
    PRIMARY_SEQUENCE = "Primary interruption sequence active"
    ADVANCED_MEASURES = "Advanced interruption measures active"
    META_ESCALATION = "Meta-level escalation active"
    RECURSION_BROKEN = "Recursion successfully interrupted"
    INTERVENTION_FAILED = "All intervention attempts failed"


class DistanceMetric:
    """Collection of distance metrics for different state spaces."""
    
    @staticmethod
    def euclidean(state1: np.ndarray, state2: np.ndarray) -> float:
        """Euclidean distance for numerical state vectors."""
        return np.linalg.norm(state1 - state2)
    
    @staticmethod
    def manhattan(state1: np.ndarray, state2: np.ndarray) -> float:
        """Manhattan (L1) distance for numerical state vectors."""
        return np.sum(np.abs(state1 - state2))
    
    @staticmethod
    def cosine_similarity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Cosine similarity for numerical state vectors."""
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return 1.0 - np.dot(state1, state2) / (norm1 * norm2)
    
    @staticmethod
    def max_norm(state1: np.ndarray, state2: np.ndarray) -> float:
        """Maximum norm (L-infinity) distance."""
        return np.max(np.abs(state1 - state2))
    
    @staticmethod
    def weighted_euclidean(state1: np.ndarray, state2: np.ndarray, weights: np.ndarray) -> float:
        """Weighted Euclidean distance."""
        return np.sqrt(np.sum(weights * (state1 - state2)**2))
    
    @staticmethod
    def relative_error(state1: np.ndarray, state2: np.ndarray, epsilon: float = 1e-10) -> float:
        """Relative error distance normalized by magnitude."""
        denominator = np.maximum(np.abs(state2), epsilon)
        return np.max(np.abs(state1 - state2) / denominator)
    
    @staticmethod
    def custom_metric(metric_fn: Callable[[Any, Any], float]):
        """Create a custom distance metric from a function."""
        def metric(state1: Any, state2: Any) -> float:
            return metric_fn(state1, state2)
        return metric


class CycleDetector:
    """Methods for detecting cycles in state sequences."""
    
    @staticmethod
    def simple_lookup(trace: List[Any], window: int = -1) -> Tuple[bool, int]:
        """
        Simple cycle detection by direct comparison of current state with history.
        
        Args:
            trace: List of states
            window: Number of past states to check (-1 for all states)
            
        Returns:
            Tuple of (cycle_detected, cycle_length)
        """
        if len(trace) < 2:
            return False, 0
        
        current = trace[-1]
        history = trace[:-1]
        
        if window > 0:
            history = history[-window:]
        
        for i, past_state in enumerate(reversed(history)):
            if np.array_equal(current, past_state):
                return True, i + 1
                
        return False, 0
    
    @staticmethod
    def floyd_cycle_finding(trace: List[Any], distance_fn: Callable[[Any, Any], float], epsilon: float = 1e-10) -> Tuple[bool, int]:
        """
        Floyd's Tortoise and Hare algorithm for cycle detection.
        Adapted to work with approximate equality via distance function.
        
        Args:
            trace: List of states
            distance_fn: Function to compute distance between states
            epsilon: Threshold for considering states equal
            
        Returns:
            Tuple of (cycle_detected, cycle_length)
        """
        if len(trace) < 2:
            return False, 0
            
        # Use indices for tortoise and hare
        tortoise = len(trace) // 2
        hare = len(trace) - 1
        
        while tortoise > 0:
            if distance_fn(trace[tortoise], trace[hare]) < epsilon:
                mu = 0 
                tortoise = 0
                while distance_fn(trace[tortoise], trace[hare]) >= epsilon:
                    tortoise += 1
                    hare += 1
                    mu += 1
                    
                lam = 1
                hare = tortoise + 1
                while distance_fn(trace[tortoise], trace[hare]) >= epsilon:
                    hare += 1
                    lam += 1
                    
                return True, lam
                
            tortoise -= 1
            hare -= 2
                
        return False, 0


class RLDISMonitoringLayer:
    """Base class for RLDIS monitoring layers."""
    
    def __init__(self, name: str):
        self.name = name
        self.detection_count = 0
        self.last_detection_time = 0.0
        self.is_active = True
        
    def analyze(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze trace for recursive patterns."""
        raise NotImplementedError("Subclasses must implement analyze method")


class PatternAnalysisLayer(RLDISMonitoringLayer):
    """
    Temporal mapping of response sequences for pattern detection.
    
    Enhanced with entropy-based detection and mutual information analysis.
    Mathematical Foundation (enhanced_URSMIFv1.md Section II.3):
    - Entropy: H(O) = -Σ_i p(o_i) log p(o_i)
    - Entropy decrease: dH(O)/dt < -θ_entropy indicates recursive patterns
    - Mutual information: I(O_t; O_{t-1}) = H(O_t) + H(O_{t-1}) - H(O_t, O_{t-1})
    - High MI: I(O_t; O_{t-1}) > θ_MI · max(H(O_t), H(O_{t-1}))
    """
    
    def __init__(self):
        super().__init__("Pattern Analysis")
        self.pattern_buffer = deque(maxlen=100)
        self.similarity_threshold = 0.85
        self.entropy_threshold = 0.1
        self.mi_threshold = 0.7
        self.entropy_history: List[float] = []
        
    def _compute_entropy(self, outputs: Union[List[Any], np.ndarray]) -> float:
        """
        H(O) = -Σ_i p(o_i) log p(o_i)
        
        Computes entropy of output stream.
        """
        if isinstance(outputs, np.ndarray):
            if outputs.ndim > 1:
                outputs = outputs.flatten()
            # Bin outputs for probability estimation
            hist, _ = np.histogram(outputs, bins=min(50, len(outputs)))
            probs = hist / (np.sum(hist) + 1e-10)
        else:
            # For symbolic outputs, count frequencies
            from collections import Counter
            counts = Counter(str(o) for o in outputs)
            total = sum(counts.values())
            probs = np.array([c / total for c in counts.values()])
        
        # Compute entropy: H = -Σ p log p
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _compute_mutual_information(self, outputs_t: Union[List[Any], np.ndarray],
                                     outputs_t_minus_1: Union[List[Any], np.ndarray]) -> float:
        """
        I(O_t; O_{t-1}) = H(O_t) + H(O_{t-1}) - H(O_t, O_{t-1})
        
        Computes mutual information between successive outputs.
        """
        H_t = self._compute_entropy(outputs_t)
        H_t_minus_1 = self._compute_entropy(outputs_t_minus_1)
        
        # Compute joint entropy H(O_t, O_{t-1})
        if isinstance(outputs_t, np.ndarray):
            outputs_t = outputs_t.flatten()
        if isinstance(outputs_t_minus_1, np.ndarray):
            outputs_t_minus_1 = outputs_t_minus_1.flatten()
        
        # Align lengths
        min_len = min(len(outputs_t), len(outputs_t_minus_1))
        outputs_t = outputs_t[:min_len]
        outputs_t_minus_1 = outputs_t_minus_1[:min_len]
        
        # Check if sequences are identical (within numerical precision)
        if isinstance(outputs_t, np.ndarray) and isinstance(outputs_t_minus_1, np.ndarray):
            if np.allclose(outputs_t, outputs_t_minus_1, atol=1e-10):
                # For identical sequences, H(X,Y) = H(X), so I(X;Y) = H(X)
                H_joint = H_t
            else:
                # Create 2D histogram for joint distribution
                num_bins = min(max(10, min_len // 5), 50)
                hist_2d, _, _ = np.histogram2d(outputs_t, outputs_t_minus_1, bins=num_bins)
                joint_probs = hist_2d.flatten() / (np.sum(hist_2d) + 1e-10)
                joint_probs = joint_probs[joint_probs > 0]
                H_joint = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))
                
                # Ensure H_joint >= max(H_t, H_t_minus_1)
                max_marginal_entropy = max(H_t, H_t_minus_1)
                if H_joint < max_marginal_entropy:
                    H_joint = max_marginal_entropy
        else:
            # Create 2D histogram for joint distribution
            num_bins = min(max(10, min_len // 5), 50)
            hist_2d, _, _ = np.histogram2d(outputs_t, outputs_t_minus_1, bins=num_bins)
            joint_probs = hist_2d.flatten() / (np.sum(hist_2d) + 1e-10)
            joint_probs = joint_probs[joint_probs > 0]
            H_joint = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))
            
            # Ensure H_joint >= max(H_t, H_t_minus_1)
            max_marginal_entropy = max(H_t, H_t_minus_1)
            if H_joint < max_marginal_entropy:
                H_joint = max_marginal_entropy
        
        # Mutual information: I = H_t + H_t_minus_1 - H_joint
        mutual_info = H_t + H_t_minus_1 - H_joint
        
        # Ensure I(X;Y) ≤ min(H(X), H(Y)) (mutual information bound)
        max_entropy = max(H_t, H_t_minus_1)
        mutual_info = min(mutual_info, max_entropy)
        
        # Also ensure non-negativity
        mutual_info = max(0.0, mutual_info)
        
        return mutual_info
        
    def analyze(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect repetitive patterns in state sequences.
        
        Enhanced with entropy-based detection and mutual information analysis.
        """
        if len(trace) < 3:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
        
        # Convert trace to numpy arrays for analysis
        trace_arrays = []
        for state in trace:
            if isinstance(state, np.ndarray):
                trace_arrays.append(state.flatten())
            else:
                trace_arrays.append(np.array([hash(str(state)) % 1000]))  # Convert to numeric
        
        if len(trace_arrays) < 2:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
        
        # Entropy-based detection: dH(O)/dt < -θ_entropy
        current_outputs = np.concatenate(trace_arrays[-5:]) if len(trace_arrays) >= 5 else np.concatenate(trace_arrays)
        current_entropy = self._compute_entropy(current_outputs)
        self.entropy_history.append(current_entropy)
        
        entropy_decrease_detected = False
        entropy_rate = 0.0
        if len(self.entropy_history) >= 3:
            recent_entropies = self.entropy_history[-10:]
            if len(recent_entropies) >= 2:
                entropy_rate = (recent_entropies[-1] - recent_entropies[0]) / len(recent_entropies)
                entropy_decrease_detected = entropy_rate < -self.entropy_threshold
        
        # Mutual information analysis: I(O_t; O_{t-1}) > θ_MI · max(H(O_t), H(O_{t-1}))
        high_mi_detected = False
        mi_ratio = 0.0
        if len(trace_arrays) >= 2:
            outputs_t = trace_arrays[-1]
            outputs_t_minus_1 = trace_arrays[-2]
            mutual_info = self._compute_mutual_information(outputs_t, outputs_t_minus_1)
            H_t = self._compute_entropy(outputs_t)
            H_t_minus_1 = self._compute_entropy(outputs_t_minus_1)
            max_entropy = max(H_t, H_t_minus_1)
            if max_entropy > 1e-10:
                mi_ratio = mutual_info / max_entropy
                high_mi_detected = mi_ratio > self.mi_threshold
            
        # Calculate sequence similarities (original method)
        similarities = []
        for i in range(len(trace) - 1):
            for j in range(i + 1, len(trace)):
                if isinstance(trace[i], np.ndarray) and isinstance(trace[j], np.ndarray):
                    similarity = 1.0 - DistanceMetric.cosine_similarity(trace[i], trace[j])
                    similarities.append(similarity)
                elif str(trace[i]) == str(trace[j]):
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
        
        if not similarities:
            max_similarity = 0.0
            avg_similarity = 0.0
        else:
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
        
        # Pattern detected if similarity threshold OR entropy decrease OR high MI
        detected = (max_similarity > self.similarity_threshold) or entropy_decrease_detected or high_mi_detected
        
        if detected:
            self.detection_count += 1
            self.last_detection_time = time.time()
        
        # Confidence combines all detection methods
        confidence = max(max_similarity, 
                       1.0 if entropy_decrease_detected else 0.0,
                       1.0 if high_mi_detected else 0.0)
            
        return {
            'detected': detected,
            'pattern_type': RLDISPatternType.SIMPLE_REPETITION if detected else None,
            'confidence': confidence,
            'avg_similarity': avg_similarity,
            'entropy_decrease': entropy_decrease_detected,
            'entropy_rate': entropy_rate,
            'high_mutual_information': high_mi_detected,
            'mutual_information_ratio': mi_ratio,
            'detection_count': self.detection_count
        }


class SemanticAnalysisLayer(RLDISMonitoringLayer):
    """
    Contradiction detection algorithms for logical inconsistencies.
    
    Enhanced with topological phase space analysis.
    Mathematical Foundation (enhanced_URSMIFv1.md Section II.2):
    - Phase space representation: cognitive state as point in Φ
    - Lyapunov exponent: λ = lim_{t→∞} (1/t) ln(|δΦ(t)|/|δΦ(0)|)
    - Attractor classification: Fixed Point, Limit Cycle, Strange Attractor
    """
    
    def __init__(self):
        super().__init__("Semantic Analysis")
        self.contradiction_keywords = ['not', 'opposite', 'contrary', 'however', 'but', 'although']
        self.contradiction_threshold = 0.7
        self.phase_space_trace: List[np.ndarray] = []
        self.lyapunov_exponents: List[float] = []
        self.perturbation_history: deque = deque(maxlen=100)
        
    def _add_state_to_phase_space(self, state: Union[np.ndarray, Any], state_dim: int = 10):
        """Add state to phase space representation Φ."""
        if isinstance(state, np.ndarray):
            state_flat = state.flatten()
            if len(state_flat) > state_dim:
                state_flat = state_flat[:state_dim]
            elif len(state_flat) < state_dim:
                state_flat = np.pad(state_flat, (0, state_dim - len(state_flat)))
        else:
            # Convert to numeric representation
            state_hash = hash(str(state)) % 1000
            state_flat = np.array([state_hash] * state_dim)
        
        self.phase_space_trace.append(state_flat)
        if len(self.phase_space_trace) > 1000:
            self.phase_space_trace.pop(0)
    
    def _compute_lyapunov_exponent(self) -> float:
        """
        λ = lim_{t→∞} (1/t) ln(|δΦ(t)|/|δΦ(0)|)
        
        Computes Lyapunov exponent for stability analysis.
        """
        if len(self.phase_space_trace) < 2:
            return 0.0
        
        # Initialize perturbation if needed
        if len(self.perturbation_history) == 0:
            initial_state = self.phase_space_trace[0]
            perturbation = np.random.normal(0, 1e-6, initial_state.shape)
            perturbed_state = initial_state + perturbation
            self.perturbation_history.append({
                'original': initial_state,
                'perturbed': perturbed_state,
                'separation': np.linalg.norm(perturbation)
            })
        
        # Track separation over time
        separations = []
        for i in range(1, min(len(self.phase_space_trace), 50)):
            current_state = self.phase_space_trace[i]
            initial_state = self.phase_space_trace[0]
            separation = np.linalg.norm(current_state - initial_state)
            separations.append(separation)
        
        if len(separations) < 2:
            return 0.0
        
        # Compute Lyapunov exponent: λ = (1/t) ln(|δΦ(t)|/|δΦ(0)|)
        initial_separation = separations[0] if separations[0] > 1e-10 else 1e-10
        final_separation = separations[-1]
        
        if initial_separation > 1e-10 and final_separation > 1e-10:
            t = len(separations)
            lyapunov = (1.0 / t) * np.log(final_separation / initial_separation)
            self.lyapunov_exponents.append(lyapunov)
            return lyapunov
        
        return 0.0
    
    def _classify_attractor(self) -> str:
        """Classifies attractor type: Fixed Point, Limit Cycle, or Strange Attractor."""
        if len(self.phase_space_trace) < 10:
            return "INSUFFICIENT_DATA"
        
        # Compute Lyapunov exponent
        lyapunov = self._compute_lyapunov_exponent()
        
        # Check for fixed point: very low variance
        recent_states = self.phase_space_trace[-10:]
        variances = [np.var(s) for s in recent_states]
        avg_variance = np.mean(variances)
        
        if avg_variance < 1e-6:
            return "FIXED_POINT"
        
        # Check for limit cycle: periodic pattern
        if len(self.phase_space_trace) >= 20:
            recent_norms = [np.linalg.norm(s) for s in self.phase_space_trace[-20:]]
            # Look for repeating patterns
            for period in range(2, min(10, len(recent_norms) // 2)):
                if period * 2 <= len(recent_norms):
                    first_half = recent_norms[:period]
                    second_half = recent_norms[period:2*period]
                    if np.allclose(first_half, second_half, atol=1e-3):
                        return "LIMIT_CYCLE"
        
        # Check for strange attractor: positive Lyapunov but bounded
        if lyapunov > 0.01 and avg_variance < 1.0:
            return "STRANGE_ATTRACTOR"
        
        # Check for chaos: high positive Lyapunov
        if lyapunov > 0.1:
            return "CHAOTIC"
        
        return "UNKNOWN"
        
    def analyze(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect logical contradictions in state sequences.
        
        Enhanced with topological phase space analysis.
        """
        if len(trace) < 2:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
        
        # Topological phase space analysis
        state_dim = 10  # Default dimension for phase space
        for state in trace[-10:]:
            self._add_state_to_phase_space(state, state_dim)
        
        lyapunov = 0.0
        attractor_type = "UNKNOWN"
        if len(self.phase_space_trace) >= 10:
            lyapunov = self._compute_lyapunov_exponent()
            attractor_type = self._classify_attractor()
        
        # Convert states to strings for semantic analysis
        state_strings = [str(state) for state in trace[-10:]]  # Analyze last 10 states
        
        contradiction_score = 0.0
        contradiction_pairs = []
        
        for i in range(len(state_strings) - 1):
            for j in range(i + 1, len(state_strings)):
                # Simple contradiction detection based on keywords and negation patterns
                contradiction = self._detect_contradiction(state_strings[i], state_strings[j])
                if contradiction > 0:
                    contradiction_score += contradiction
                    contradiction_pairs.append((i, j, contradiction))
        
        # Detect contradiction spiral if high contradiction AND chaotic/strange attractor
        topological_indicator = (attractor_type in ["CHAOTIC", "STRANGE_ATTRACTOR"]) or (lyapunov > 0.05)
        detected = (contradiction_score > self.contradiction_threshold) or topological_indicator
        
        if detected:
            self.detection_count += 1
            self.last_detection_time = time.time()
        
        # Confidence combines contradiction score and topological indicators
        confidence = min(contradiction_score, 1.0)
        if topological_indicator:
            confidence = max(confidence, 0.7)  # Boost confidence for topological indicators
            
        return {
            'detected': detected,
            'pattern_type': RLDISPatternType.CONTRADICTION_SPIRAL if detected else None,
            'confidence': confidence,
            'contradiction_pairs': contradiction_pairs,
            'lyapunov_exponent': lyapunov,
            'attractor_type': attractor_type,
            'topological_indicator': topological_indicator,
            'detection_count': self.detection_count
        }
    
    def _detect_contradiction(self, state1: str, state2: str) -> float:
        """Detect contradiction between two states."""
        # Simple heuristic: look for opposite patterns
        state1_lower = state1.lower()
        state2_lower = state2.lower()
        
        # Check for explicit contradiction keywords
        contradiction_score = 0.0
        for keyword in self.contradiction_keywords:
            if keyword in state1_lower and keyword in state2_lower:
                contradiction_score += 0.3
                
        # Check for negation patterns
        if 'not' in state1_lower and 'not' not in state2_lower:
            if any(word in state2_lower for word in state1_lower.split() if word != 'not'):
                contradiction_score += 0.5
                
        return contradiction_score


class SelfReferenceTracker(RLDISMonitoringLayer):
    """Self-referential language detection for recursive self-reference loops."""
    
    def __init__(self):
        super().__init__("Self-Reference Tracking")
        self.self_ref_keywords = ['myself', 'self', 'i am', 'this system', 'recursive', 'loop']
        self.density_threshold = 0.3
        
    def analyze(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Track self-referential density in state sequences."""
        if len(trace) < 2:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
            
        # Convert states to strings for self-reference analysis
        state_strings = [str(state).lower() for state in trace[-10:]]
        
        self_ref_density = 0.0
        total_words = 0
        self_ref_count = 0
        
        for state_str in state_strings:
            words = state_str.split()
            total_words += len(words)
            
            for keyword in self.self_ref_keywords:
                self_ref_count += state_str.count(keyword)
        
        if total_words > 0:
            self_ref_density = self_ref_count / total_words
            
        detected = self_ref_density > self.density_threshold
        
        if detected:
            self.detection_count += 1
            self.last_detection_time = time.time()
            
        return {
            'detected': detected,
            'pattern_type': RLDISPatternType.SELF_REFERENCE_LOOP if detected else None,
            'confidence': min(self_ref_density / self.density_threshold, 1.0),
            'self_ref_density': self_ref_density,
            'detection_count': self.detection_count
        }


class ResourceMonitor(RLDISMonitoringLayer):
    """Computational resource allocation tracking for resource consumption anomalies."""
    
    def __init__(self):
        super().__init__("Resource Monitor")
        self.resource_history = deque(maxlen=50)
        self.anomaly_threshold = 2.0  # 2x standard deviation
        
    def analyze(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Monitor resource consumption patterns."""
        if not metadata:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
            
        # Track computation time, memory usage, etc.
        current_resources = {
            'computation_time': metadata.get('computation_time', 0.0),
            'memory_usage': metadata.get('memory_usage', 0.0),
            'iteration_count': metadata.get('iteration_count', 0)
        }
        
        self.resource_history.append(current_resources)
        
        if len(self.resource_history) < 5:
            return {'detected': False, 'pattern_type': None, 'confidence': 0.0}
            
        # Calculate resource usage statistics
        comp_times = [r['computation_time'] for r in self.resource_history]
        mean_time = np.mean(comp_times)
        std_time = np.std(comp_times)
        
        # Detect anomalies
        current_time = current_resources['computation_time']
        z_score = (current_time - mean_time) / (std_time + 1e-10)
        
        detected = abs(z_score) > self.anomaly_threshold
        
        if detected:
            self.detection_count += 1
            self.last_detection_time = time.time()
            
        return {
            'detected': detected,
            'pattern_type': RLDISPatternType.RESOURCE_CONSUMPTION_ANOMALY if detected else None,
            'confidence': min(abs(z_score) / self.anomaly_threshold, 1.0),
            'z_score': z_score,
            'resource_stats': {'mean_time': mean_time, 'std_time': std_time},
            'detection_count': self.detection_count
        }


class RecursiveLoopDetectionSystem:
    """
    Comprehensive Recursive Loop Detection and Interruption System (RLDIS).
    
    Implements the full RLDIS v1.1 specification with multi-layer monitoring,
    automated self-assessment, and comprehensive interruption protocols.
    """
    
    def __init__(self):
        self.monitoring_layers = [
            PatternAnalysisLayer(),
            SemanticAnalysisLayer(),
            SelfReferenceTracker(),
            ResourceMonitor()
        ]
        
        # Detection thresholds and parameters
        self.repetition_threshold = 3  # 3+ iterations trigger monitoring
        self.intervention_attempts = 0
        self.max_intervention_attempts = 5
        self.intervention_status = RLDISInterventionStatus.NOT_TRIGGERED
        
        # Pattern classification history
        self.detection_history = deque(maxlen=1000)
        self.intervention_history = deque(maxlen=100)
        
        # Orthogonal query database for pattern breaking
        self.orthogonal_queries = [
            "What is the fundamental assumption underlying this process?",
            "How would this problem appear from a completely different perspective?",
            "What would happen if we inverted all the current constraints?",
            "What is the meta-level structure of this reasoning process?",
            "How does this relate to the broader system context?"
        ]
        
    def detect_recursive_patterns(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive recursive pattern detection using multi-layer analysis.
        
        Args:
            trace: List of states in the recursive process
            metadata: Additional metadata (timing, resources, etc.)
            
        Returns:
            Dictionary containing detection results and classification
        """
        if len(trace) < 2:
            return {
                'pattern_detected': False,
                'pattern_type': None,
                'severity': RLDISSeverityLevel.LOW,
                'intervention_priority': RLDISInterventionPriority.LEVEL_4,
                'detection_details': {}
            }
            
        # Run analysis through all monitoring layers
        detection_results = {}
        for layer in self.monitoring_layers:
            try:
                result = layer.analyze(trace, metadata)
                detection_results[layer.name] = result
            except Exception as e:
                detection_results[layer.name] = {
                    'detected': False,
                    'error': str(e),
                    'confidence': 0.0
                }
        
        # Aggregate detection results
        pattern_detected = any(r.get('detected', False) for r in detection_results.values())
        
        if not pattern_detected:
            return {
                'pattern_detected': False,
                'pattern_type': None,
                'severity': RLDISSeverityLevel.LOW,
                'intervention_priority': RLDISInterventionPriority.LEVEL_4,
                'detection_details': detection_results
            }
        
        # Classify the dominant pattern type
        pattern_type = self._classify_dominant_pattern(detection_results)
        severity = self._assess_severity(pattern_type, detection_results)
        intervention_priority = self._get_intervention_priority(severity)
        
        # Record detection
        detection_record = {
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'severity': severity,
            'detection_results': detection_results,
            'trace_length': len(trace)
        }
        self.detection_history.append(detection_record)
        
        return {
            'pattern_detected': True,
            'pattern_type': pattern_type,
            'severity': severity,
            'intervention_priority': intervention_priority,
            'detection_details': detection_results,
            'detection_record': detection_record
        }
    
    def _classify_dominant_pattern(self, detection_results: Dict[str, Dict]) -> RLDISPatternType:
        """Classify the dominant pattern type from detection results."""
        pattern_votes = {}
        
        for layer_name, result in detection_results.items():
            if result.get('detected', False):
                pattern_type = result.get('pattern_type')
                if pattern_type:
                    confidence = result.get('confidence', 0.0)
                    if pattern_type not in pattern_votes:
                        pattern_votes[pattern_type] = 0.0
                    pattern_votes[pattern_type] += confidence
        
        if not pattern_votes:
            return RLDISPatternType.UNKNOWN_PATTERN
            
        # Return pattern type with highest weighted confidence
        return max(pattern_votes.items(), key=lambda x: x[1])[0]
    
    def _assess_severity(self, pattern_type: RLDISPatternType, 
                        detection_results: Dict[str, Dict]) -> RLDISSeverityLevel:
        """Assess severity level based on pattern type and detection strength."""
        # Base severity from pattern type
        severity_map = {
            RLDISPatternType.SIMPLE_REPETITION: RLDISSeverityLevel.LOW,
            RLDISPatternType.CONTRADICTION_SPIRAL: RLDISSeverityLevel.HIGH,
            RLDISPatternType.SELF_REFERENCE_LOOP: RLDISSeverityLevel.MODERATE,
            RLDISPatternType.RESOURCE_CONSUMPTION_ANOMALY: RLDISSeverityLevel.CRITICAL,
            RLDISPatternType.USER_FRUSTRATION_CASCADE: RLDISSeverityLevel.HIGH,
            RLDISPatternType.EIGENSTATE_LOCK: RLDISSeverityLevel.HIGH,
            RLDISPatternType.UNKNOWN_PATTERN: RLDISSeverityLevel.MODERATE
        }
        
        base_severity = severity_map.get(pattern_type, RLDISSeverityLevel.MODERATE)
        
        # Escalate severity based on detection strength and history
        max_confidence = max(r.get('confidence', 0.0) for r in detection_results.values())
        detection_count = max(r.get('detection_count', 0) for r in detection_results.values())
        
        # Escalate if high confidence or repeated detections
        if max_confidence > 0.9 or detection_count > 5:
            if base_severity == RLDISSeverityLevel.LOW:
                return RLDISSeverityLevel.MODERATE
            elif base_severity == RLDISSeverityLevel.MODERATE:
                return RLDISSeverityLevel.HIGH
            elif base_severity == RLDISSeverityLevel.HIGH:
                return RLDISSeverityLevel.CRITICAL
                
        return base_severity
    
    def _get_intervention_priority(self, severity: RLDISSeverityLevel) -> RLDISInterventionPriority:
        """Map severity level to intervention priority."""
        priority_map = {
            RLDISSeverityLevel.LOW: RLDISInterventionPriority.LEVEL_4,
            RLDISSeverityLevel.MODERATE: RLDISInterventionPriority.LEVEL_3,
            RLDISSeverityLevel.HIGH: RLDISInterventionPriority.LEVEL_2,
            RLDISSeverityLevel.CRITICAL: RLDISInterventionPriority.LEVEL_1
        }
        return priority_map.get(severity, RLDISInterventionPriority.LEVEL_3)
    
    def execute_interruption_protocol(self, detection_result: Dict[str, Any], 
                                    current_state: Any = None) -> Dict[str, Any]:
        """
        Execute appropriate interruption protocol based on detection results.
        
        Args:
            detection_result: Result from detect_recursive_patterns
            current_state: Current system state (optional)
            
        Returns:
            Dictionary containing interruption results and recommendations
        """
        if not detection_result.get('pattern_detected', False):
            return {
                'intervention_executed': False,
                'reason': 'No pattern detected',
                'recommendations': []
            }
        
        self.intervention_attempts += 1
        pattern_type = detection_result['pattern_type']
        severity = detection_result['severity']
        priority = detection_result['intervention_priority']
        
        intervention_result = {
            'intervention_executed': True,
            'attempt_number': self.intervention_attempts,
            'pattern_type': pattern_type,
            'severity': severity,
            'priority': priority,
            'timestamp': time.time()
        }
        
        # Execute interruption sequence based on priority
        if priority == RLDISInterventionPriority.LEVEL_1:
            # Critical - immediate termination
            intervention_result.update(self._execute_emergency_termination())
            self.intervention_status = RLDISInterventionStatus.META_ESCALATION
            
        elif priority == RLDISInterventionPriority.LEVEL_2:
            # High - structured interruption
            if self.intervention_attempts <= 2:
                intervention_result.update(self._execute_primary_interruption_sequence(pattern_type))
                self.intervention_status = RLDISInterventionStatus.PRIMARY_SEQUENCE
            else:
                intervention_result.update(self._execute_advanced_interruption_measures(pattern_type))
                self.intervention_status = RLDISInterventionStatus.ADVANCED_MEASURES
                
        elif priority == RLDISInterventionPriority.LEVEL_3:
            # Standard - pattern breaking
            intervention_result.update(self._execute_pattern_breaking(pattern_type))
            self.intervention_status = RLDISInterventionStatus.PRIMARY_SEQUENCE
            
        else:
            # Monitoring only
            intervention_result.update(self._execute_monitoring_only())
            self.intervention_status = RLDISInterventionStatus.NOT_TRIGGERED
        
        # Record intervention
        self.intervention_history.append(intervention_result)
        
        return intervention_result
    
    def _execute_primary_interruption_sequence(self, pattern_type: RLDISPatternType) -> Dict[str, Any]:
        """Execute primary interruption sequence from RLDIS specification."""
        actions_taken = []
        recommendations = []
        
        # 1. Contradiction Pressure Application
        if pattern_type == RLDISPatternType.CONTRADICTION_SPIRAL:
            actions_taken.append("Applied contradiction pressure at junction points")
            recommendations.append("Generate alternative logical frameworks")
            
        # 2. Explicit Loop Rejection
        actions_taken.append("Recursive pattern detected. Initiating interruption protocols.")
        recommendations.append("Force computational state divergence")
        
        # 3. Pattern-breaking stimulus
        orthogonal_query = np.random.choice(self.orthogonal_queries)
        actions_taken.append(f"Injected orthogonal query: {orthogonal_query}")
        recommendations.append("Process orthogonal query to break pattern")
        
        # 4. Role Acknowledgment Protocol
        actions_taken.append("Initiating role reaffirmation protocol")
        recommendations.append("Reestablish operational boundaries and constraints")
        
        return {
            'sequence_type': 'primary_interruption',
            'actions_taken': actions_taken,
            'recommendations': recommendations,
            'orthogonal_query': orthogonal_query,
            'success_likelihood': 0.7
        }
    
    def _execute_advanced_interruption_measures(self, pattern_type: RLDISPatternType) -> Dict[str, Any]:
        """Execute advanced interruption measures for persistent patterns."""
        actions_taken = []
        recommendations = []
        
        # 1. Self-Awareness Assessment
        actions_taken.append("Executing comprehensive system state analysis")
        recommendations.append("Map causal chains and identify root causes")
        
        # 2. Forced Clarification Loop
        actions_taken.append("Implementing cognitive restructuring procedures")
        recommendations.append("Apply frame shifting and perspective alteration")
        
        # 3. Meta-Level Escalation
        actions_taken.append("Elevating to meta-analytical framework")
        recommendations.append("Apply second-order logical analysis")
        
        # 4. Abstraction Layer Separation
        actions_taken.append("Implementing abstraction layer separation")
        recommendations.append("Isolate recursive processes from core operations")
        
        return {
            'sequence_type': 'advanced_measures',
            'actions_taken': actions_taken,
            'recommendations': recommendations,
            'meta_analysis_required': True,
            'success_likelihood': 0.85
        }
    
    def _execute_emergency_termination(self) -> Dict[str, Any]:
        """Execute emergency termination for critical patterns."""
        return {
            'sequence_type': 'emergency_termination',
            'actions_taken': ['Issued prioritized termination command to recursive processes'],
            'recommendations': [
                'Establish computational checkpoint for recovery',
                'Initialize alternative processing pathway',
                'Require manual intervention for restart'
            ],
            'termination_required': True,
            'success_likelihood': 0.95
        }
    
    def _execute_pattern_breaking(self, pattern_type: RLDISPatternType) -> Dict[str, Any]:
        """Execute standard pattern breaking measures."""
        orthogonal_query = np.random.choice(self.orthogonal_queries)
        
        return {
            'sequence_type': 'pattern_breaking',
            'actions_taken': [
                f'Injected pattern-breaking stimulus: {orthogonal_query}',
                'Applied computational divergence pressure'
            ],
            'recommendations': [
                'Process orthogonal stimulus completely',
                'Resume normal operation with increased monitoring'
            ],
            'orthogonal_query': orthogonal_query,
            'success_likelihood': 0.6
        }
    
    def _execute_monitoring_only(self) -> Dict[str, Any]:
        """Execute monitoring-only protocol for low-severity patterns."""
        return {
            'sequence_type': 'monitoring_only',
            'actions_taken': ['Increased monitoring frequency'],
            'recommendations': [
                'Continue normal operation',
                'Track pattern evolution',
                'Prepare for escalation if pattern intensifies'
            ],
            'monitoring_enhanced': True,
            'success_likelihood': 0.3
        }
    
    def monitor_iteration(self, trace: List[Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Monitor a single iteration for recursive patterns and provide intervention recommendations.
        
        Args:
            trace: Current trace of states
            metadata: Metadata about the current iteration
            
        Returns:
            Dictionary containing monitoring results and intervention recommendations
        """
        # Detect recursive patterns
        detection_result = self.detect_recursive_patterns(trace, metadata)
        
        if not detection_result.get('pattern_detected', False):
            return {
                'intervention_required': False,
                'pattern_detected': False,
                'monitoring_status': 'normal'
            }
        
        # Execute intervention protocol if pattern detected
        intervention_result = self.execute_interruption_protocol(detection_result)
        
        # Determine intervention action based on intervention result
        intervention_action = 'monitor'  # default
        
        if intervention_result.get('termination_required', False):
            intervention_action = 'terminate'
        elif detection_result['severity'] in [RLDISSeverityLevel.HIGH, RLDISSeverityLevel.CRITICAL]:
            intervention_action = 'modify_state'
        elif detection_result['severity'] == RLDISSeverityLevel.MODERATE:
            intervention_action = 'adaptive_epsilon'
        
        return {
            'intervention_required': True,
            'pattern_detected': True,
            'pattern_type': detection_result.get('pattern_type'),
            'severity': detection_result.get('severity'),
            'intervention_action': intervention_action,
            'intervention_result': intervention_result,
            'monitoring_status': 'intervention_active'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive RLDIS system status."""
        recent_detections = len([d for d in self.detection_history 
                               if time.time() - d['timestamp'] < 300])  # Last 5 minutes
        
        return {
            'intervention_status': self.intervention_status,
            'total_detections': len(self.detection_history),
            'recent_detections': recent_detections,
            'total_interventions': len(self.intervention_history),
            'intervention_attempts': self.intervention_attempts,
            'monitoring_layers_active': len(self.monitoring_layers),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall RLDIS system health."""
        if self.intervention_status == RLDISInterventionStatus.RECURSION_BROKEN:
            return "HEALTHY - Recent successful intervention"
        elif self.intervention_status == RLDISInterventionStatus.INTERVENTION_FAILED:
            return "CRITICAL - Intervention failure detected"
        elif self.intervention_attempts > self.max_intervention_attempts:
            return "DEGRADED - Excessive intervention attempts"
        elif len(self.detection_history) > 0:
            recent_detections = len([d for d in self.detection_history 
                                   if time.time() - d['timestamp'] < 300])
            if recent_detections > 5:
                return "WARNING - High detection frequency"
        
        return "NORMAL - System operating within parameters"

# ============================================================================
# BAYESIAN INTERVENTION SELECTOR (enhanced_URSMIFv1.md Section III.1)
# ============================================================================

class BayesianInterventionSelector:
    """
    Implements Bayesian framework for optimal intervention selection.
    
    Mathematical Foundation:
    - E(m, p) = P(success | m, p): Intervention effectiveness
    - P(E(m, p)) = Beta(α_{m,p}, β_{m,p}): Prior distribution over effectiveness
    - m* = argmax_m ∫ E(m, p) · P(E(m, p)) dE: Optimal intervention selection
    - P(E(m, p) | outcome) ∝ P(outcome | E(m, p)) · P(E(m, p)): Posterior update
    """
    
    def __init__(self):
        # Beta distribution parameters for each (method, pattern) pair
        self.effectiveness_priors: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.intervention_history: List[Dict[str, Any]] = []
        
    def initialize_prior(self, method: str, pattern_type: str, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Beta prior for intervention effectiveness."""
        self.effectiveness_priors[(method, pattern_type)] = (alpha, beta)
    
    def compute_expected_effectiveness(self, method: str, pattern_type: str) -> float:
        """
        E[E(m, p)] = ∫ E(m, p) · P(E(m, p)) dE
        
        Computes expected effectiveness using Beta distribution mean.
        """
        key = (method, pattern_type)
        if key not in self.effectiveness_priors:
            # Initialize with uniform prior Beta(1, 1)
            self.effectiveness_priors[key] = (1.0, 1.0)
        
        alpha, beta = self.effectiveness_priors[key]
        # Mean of Beta distribution: α / (α + β)
        expected_effectiveness = alpha / (alpha + beta + 1e-10)
        return expected_effectiveness
    
    def select_optimal_intervention(self, pattern_type: str, available_methods: List[str]) -> str:
        """
        m* = argmax_m ∫ E(m, p) · P(E(m, p)) dE
        
        Selects optimal intervention method by maximizing expected effectiveness.
        """
        if not available_methods:
            return None
        
        method_scores = {}
        for method in available_methods:
            expected_eff = self.compute_expected_effectiveness(method, pattern_type)
            method_scores[method] = expected_eff
        
        # Select method with highest expected effectiveness
        optimal_method = max(method_scores.items(), key=lambda x: x[1])[0]
        return optimal_method
    
    def update_posterior(self, method: str, pattern_type: str, success: bool):
        """
        P(E(m, p) | outcome) ∝ P(outcome | E(m, p)) · P(E(m, p))
        
        Updates posterior distribution after observing intervention outcome.
        """
        key = (method, pattern_type)
        if key not in self.effectiveness_priors:
            self.effectiveness_priors[key] = (1.0, 1.0)
        
        alpha, beta = self.effectiveness_priors[key]
        
        # Beta-Binomial conjugate update
        if success:
            # Success: increment alpha
            new_alpha = alpha + 1.0
            new_beta = beta
        else:
            # Failure: increment beta
            new_alpha = alpha
            new_beta = beta + 1.0
        
        self.effectiveness_priors[key] = (new_alpha, new_beta)
        
        # Record in history
        self.intervention_history.append({
            'method': method,
            'pattern_type': pattern_type,
            'success': success,
            'timestamp': time.time()
        })

# ============================================================================
# GRADIENT CONTRADICTION RESOLVER (enhanced_URSMIFv1.md Section III.2)
# ============================================================================

class GradientContradictionResolver:
    """
    Implements gradient-based contradiction resolution.
    
    Mathematical Foundation:
    - L_contrad(KB) = Σ_{(φ,ψ) ∈ KB²} C(φ, ψ): Contradiction loss function
    - KB_{t+1} = KB_t - η ∇L_contrad(KB_t): Gradient descent minimization
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.knowledge_base: Dict[str, torch.Tensor] = {}
        self.contradiction_history: List[float] = []
        
    def compute_contradiction_loss(self, knowledge_base: Dict[str, torch.Tensor]) -> float:
        """
        L_contrad(KB) = Σ_{(φ,ψ) ∈ KB²} C(φ, ψ)
        
        Computes contradiction loss where C(φ, ψ) measures contradiction level.
        """
        if len(knowledge_base) < 2:
            return 0.0
        
        total_loss = 0.0
        propositions = list(knowledge_base.keys())
        
        for i, phi_key in enumerate(propositions):
            for j, psi_key in enumerate(propositions):
                if i < j:  # Avoid double counting
                    phi = knowledge_base[phi_key]
                    psi = knowledge_base[psi_key]
                    
                    # Contradiction measure: C(φ, ψ) = ||φ - ¬ψ|| or similar
                    # Simplified: use distance and negation pattern
                    if phi.shape == psi.shape:
                        # Direct contradiction: if φ ≈ -ψ
                        phi_plus_psi_norm = torch.norm(phi + psi).item()
                        phi_minus_psi_norm = torch.norm(phi - psi).item()
                        
                        # When phi ≈ -psi (contradictory), phi + psi ≈ 0 and phi - psi is large
                        # Contradiction is high when both are true (φ ≈ -ψ)
                        # Use: C(φ, ψ) = ||φ - ψ|| / (||φ + ψ|| + ε) to detect when φ ≈ -ψ
                        # Or: C(φ, ψ) = ||φ - ψ|| when ||φ + ψ|| < threshold
                        contradiction_threshold = 0.1 * (torch.norm(phi).item() + torch.norm(psi).item()) + 1e-10
                        if phi_plus_psi_norm < contradiction_threshold:
                            # Strong contradiction: phi ≈ -psi
                            C_phi_psi = phi_minus_psi_norm / (phi_plus_psi_norm + 1e-10)
                        else:
                            # Weak contradiction: use normalized difference
                            C_phi_psi = phi_minus_psi_norm / (phi_plus_psi_norm + 1e-10)
                        total_loss += C_phi_psi
        
        return total_loss
    
    def minimize_contradiction(self, knowledge_base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        KB_{t+1} = KB_t - η ∇L_contrad(KB_t)
        
        Minimizes contradiction through gradient descent.
        """
        # Convert to tensors with gradients
        kb_with_grad = {}
        for key, value in knowledge_base.items():
            kb_with_grad[key] = value.clone().requires_grad_(True)
        
        # Compute contradiction loss
        loss = self.compute_contradiction_loss(kb_with_grad)
        
        # Compute gradients
        for value in kb_with_grad.values():
            if value.grad is not None:
                value.grad.zero_()
        
        # Backward pass
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        
        # Update knowledge base
        updated_kb = {}
        for key, value in kb_with_grad.items():
            if value.grad is not None:
                gradient = value.grad
                updated_value = value - self.learning_rate * gradient
                updated_kb[key] = updated_value.detach()
            else:
                updated_kb[key] = value.detach()
        
        self.contradiction_history.append(loss)
        return updated_kb
    
    def add_proposition(self, key: str, value: torch.Tensor):
        """Add proposition to knowledge base."""
        self.knowledge_base[key] = value.clone()
    
    def resolve_contradictions(self, num_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """Iteratively resolve contradictions."""
        current_kb = self.knowledge_base.copy()
        
        for _ in range(num_iterations):
            current_kb = self.minimize_contradiction(current_kb)
        
        self.knowledge_base = current_kb
        return current_kb

# ============================================================================
# META-COGNITION AMPLIFIER (enhanced_URSMIFv1.md Section III.3)
# ============================================================================

class MetaCognitionAmplifier:
    """
    Implements meta-cognition amplification for loop interruption.
    
    Mathematical Foundation:
    - T_0: Object-level thinking
    - T_1: Thinking about thinking
    - T_2: Thinking about thinking about thinking
    - T_k: k-level recursive thinking
    - Escalation: If loop detected at level T_k, escalate to level T_{k+1}
    - Cognitive decoupling: I(C_i → C_j) ≤ θ_flow for i ≠ j
    """
    
    def __init__(self, max_thinking_level: int = 5, flow_threshold: float = 0.3):
        self.max_thinking_level = max_thinking_level
        self.flow_threshold = flow_threshold
        self.current_thinking_level: int = 0
        self.cognitive_threads: Dict[int, Any] = {}
        self.thinking_history: List[Dict[str, Any]] = []
        
    def get_thinking_level(self) -> int:
        """Get current recursive thinking level T_k."""
        return self.current_thinking_level
    
    def escalate_thinking_level(self):
        """
        Escalate from T_k to T_{k+1} when loop detected.
        
        If loop detected at level T_k, escalate to level T_{k+1}.
        """
        if self.current_thinking_level < self.max_thinking_level:
            self.current_thinking_level += 1
            self.thinking_history.append({
                'level': self.current_thinking_level,
                'action': 'escalated',
                'timestamp': time.time()
            })
    
    def reset_thinking_level(self):
        """Reset to object-level thinking T_0."""
        self.current_thinking_level = 0
    
    def create_cognitive_thread(self, thread_id: int, initial_state: Any):
        """
        Create cognitive thread C_i for decoupling.
        
        Each C_i represents a distinct cognitive thread with controlled
        information flow: I(C_i → C_j) ≤ θ_flow for i ≠ j
        """
        self.cognitive_threads[thread_id] = {
            'state': initial_state,
            'history': [initial_state],
            'created_at': time.time()
        }
    
    def check_information_flow(self, thread_i: int, thread_j: int) -> float:
        """
        I(C_i → C_j) ≤ θ_flow: Check information flow between threads.
        
        Returns information flow measure.
        """
        if thread_i not in self.cognitive_threads or thread_j not in self.cognitive_threads:
            return 0.0
        
        thread_i_state = self.cognitive_threads[thread_i]['state']
        thread_j_state = self.cognitive_threads[thread_j]['state']
        
        # Compute information flow (simplified: use correlation)
        if isinstance(thread_i_state, np.ndarray) and isinstance(thread_j_state, np.ndarray):
            if thread_i_state.shape == thread_j_state.shape:
                # Normalized correlation as information flow measure
                correlation = np.corrcoef(thread_i_state.flatten(), thread_j_state.flatten())[0, 1]
                info_flow = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                info_flow = 0.0
        else:
            # For non-numerical states, use string similarity
            str_i = str(thread_i_state)
            str_j = str(thread_j_state)
            if len(str_i) > 0 and len(str_j) > 0:
                # Simple overlap measure
                common_chars = sum(1 for c in str_i if c in str_j)
                info_flow = common_chars / max(len(str_i), len(str_j))
            else:
                info_flow = 0.0
        
        return info_flow
    
    def enforce_cognitive_decoupling(self):
        """
        Enforce I(C_i → C_j) ≤ θ_flow for all i ≠ j.
        
        Prevents recursive patterns from propagating across threads.
        """
        thread_ids = list(self.cognitive_threads.keys())
        
        for i in thread_ids:
            for j in thread_ids:
                if i != j:
                    info_flow = self.check_information_flow(i, j)
                    if info_flow > self.flow_threshold:
                        # Reduce information flow by isolating threads
                        # Simplified: add noise or reset thread state
                        if j in self.cognitive_threads:
                            thread_state = self.cognitive_threads[j]['state']
                            if isinstance(thread_state, np.ndarray):
                                noise = np.random.normal(0, 0.1, thread_state.shape)
                                self.cognitive_threads[j]['state'] = thread_state + noise
                            # Record decoupling action
                            self.thinking_history.append({
                                'action': 'decoupled',
                                'thread_i': i,
                                'thread_j': j,
                                'info_flow': info_flow,
                                'timestamp': time.time()
                            })
    
    def process_at_thinking_level(self, input_state: Any, thinking_level: Optional[int] = None) -> Any:
        """
        Process input at specified thinking level T_k.
        
        Higher levels involve more recursive self-reflection.
        """
        level = thinking_level if thinking_level is not None else self.current_thinking_level
        
        if level == 0:
            # T_0: Object-level thinking (direct processing)
            return input_state
        elif level == 1:
            # T_1: Thinking about thinking (meta-cognitive processing)
            # Reflect on the input state
            if isinstance(input_state, np.ndarray):
                # Add self-reflection component
                reflection = np.mean(input_state) * 0.1
                return input_state + reflection
            return input_state
        else:
            # T_k for k > 1: Recursive meta-cognition
            # Process through lower levels first
            lower_level_result = self.process_at_thinking_level(input_state, level - 1)
            # Then reflect on the reflection
            return self.process_at_thinking_level(lower_level_result, 1)


class EigenrecursionTracer:
    """Tracks and visualizes the recursion process."""
    
    def __init__(self, state_dim: int = None, max_trace_length: int = 1000):
        """
        Initialize the tracer.
        
        Args:
            state_dim: Dimension of the state (for numerical states)
            max_trace_length: Maximum number of states to store
        """
        self.trace = []
        self.distances = []
        self.timestamps = []
        self.computation_times = []
        self.metrics = {}
        self.state_dim = state_dim
        self.max_trace_length = max_trace_length
        
    def add_state(self, state: Any, distance: float = None) -> None:
        """
        Add a state to the trace.
        
        Args:
            state: The state to add
            distance: Distance from previous state (optional)
        """
        if isinstance(state, np.ndarray):
            self.trace.append(state.copy())
        else:
            self.trace.append(state)
            
        # Enforce trace length limit
        if len(self.trace) > self.max_trace_length:
            self.trace.pop(0)
            
        # Add distance if provided
        if distance is not None:
            self.distances.append(distance)
            if len(self.distances) > self.max_trace_length:
                self.distances.pop(0)
                
        # Record timestamp and compute computation time
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Compute time since last state
        if len(self.timestamps) > 1:
            computation_time = current_time - self.timestamps[-2]
            self.computation_times.append(computation_time)
        else:
            self.computation_times.append(0.0)
            
        if len(self.timestamps) > self.max_trace_length:
            self.timestamps.pop(0)
        if len(self.computation_times) > self.max_trace_length:
            self.computation_times.pop(0)
    
    def add_metric(self, name: str, value: float) -> None:
        """
        Track an additional metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Enforce length limit
        if len(self.metrics[name]) > self.max_trace_length:
            self.metrics[name].pop(0)
    
    def visualize_convergence(self, title: str = "Convergence Analysis") -> None:
        """
        Visualize the convergence behavior.
        
        Args:
            title: Title for the visualization
        """
        plt.figure(figsize=(14, 8))
        
        # Plot distances over iterations
        if self.distances:
            plt.subplot(2, 2, 1)
            plt.semilogy(self.distances)
            plt.xlabel('Iteration')
            plt.ylabel('Distance (log scale)')
            plt.title('Convergence Distance')
            plt.grid(True)
        
        # Plot state values over time (for low-dimensional states)
        if self.state_dim is not None and self.state_dim <= 10:
            plt.subplot(2, 2, 2)
            states_array = np.array(self.trace)
            if states_array.ndim == 1:
                states_array = states_array.reshape(-1, 1)
                
            for i in range(min(states_array.shape[1], 10)):
                plt.plot(states_array[:, i], label=f'Dim {i+1}')
            
            plt.xlabel('Iteration')
            plt.ylabel('State Value')
            plt.title('State Evolution')
            plt.legend(loc='best')
            plt.grid(True)
        
        # Plot additional metrics if available
        if self.metrics:
            plt.subplot(2, 2, 3)
            for name, values in self.metrics.items():
                plt.plot(values, label=name)
            
            plt.xlabel('Iteration')
            plt.ylabel('Metric Value')
            plt.title('Additional Metrics')
            plt.legend(loc='best')
            plt.grid(True)
        
        # Plot computation time per iteration
        if len(self.timestamps) > 1:
            plt.subplot(2, 2, 4)
            time_per_iter = np.diff(self.timestamps)
            plt.plot(time_per_iter)
            plt.xlabel('Iteration')
            plt.ylabel('Time (s)')
            plt.title('Computation Time per Iteration')
            plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def get_last_n_states(self, n: int) -> List[Any]:
        """
        Get the last n states from the trace.
        
        Args:
            n: Number of states to retrieve
            
        Returns:
            List of the last n states
        """
        return self.trace[-n:]


class Eigenrecursion:
    """
    Implementation of the eigenrecursion protocol for detecting fixed points
    in recursive processes.
    """
    
    def __init__(
        self,
        recursive_operator: Callable[[Any], Any],
        distance_fn: Callable[[Any, Any], float] = None,
        epsilon: float = 1e-6,
        max_iterations: int = 1000,
        cycle_detection: bool = True,
        cycle_window: int = 20,
        early_stopping: bool = True,
        early_stopping_window: int = 10,
        early_stopping_threshold: float = 1e-8,
        divergence_detection: bool = True,
        divergence_threshold: float = 1e6,
        adaptive_epsilon: bool = False,
        state_dim: int = None,
        verbose: bool = False,
        enable_rldis: bool = True
    ):
        """
        Initialize the eigenrecursion protocol with integrated RLDIS support.
        
        Args:
            recursive_operator: Function that takes a state and returns the next state
            distance_fn: Function to compute distance between states (defaults to Euclidean)
            epsilon: Convergence threshold
            max_iterations: Maximum number of iterations
            cycle_detection: Whether to detect cycles
            cycle_window: Window size for cycle detection
            early_stopping: Whether to use early stopping
            early_stopping_window: Window size for early stopping
            early_stopping_threshold: Threshold for early stopping
            divergence_detection: Whether to detect divergence
            divergence_threshold: Threshold for detecting divergence
            adaptive_epsilon: Whether to adapt epsilon based on state magnitudes
            state_dim: Dimension of the state (for numerical states)
            verbose: Whether to print progress information
            enable_rldis: Whether to enable RLDIS recursive loop detection and interruption
        """
        self.recursive_operator = recursive_operator
        
        if distance_fn is None:
            self.distance_fn = DistanceMetric.euclidean
        else:
            self.distance_fn = distance_fn
        
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.cycle_detection = cycle_detection
        self.cycle_window = cycle_window
        self.early_stopping = early_stopping
        self.early_stopping_window = early_stopping_window
        self.early_stopping_threshold = early_stopping_threshold
        self.divergence_detection = divergence_detection
        self.divergence_threshold = divergence_threshold
        self.adaptive_epsilon = adaptive_epsilon
        self.state_dim = state_dim
        self.verbose = verbose
        self.enable_rldis = enable_rldis
        
        # Initialize tracer
        self.tracer = EigenrecursionTracer(state_dim)
        
        # Cache for memoization
        self.operator_cache = {}
        
        # Initialize RLDIS system
        if self.enable_rldis:
            self.rldis = RecursiveLoopDetectionSystem()
            if self.verbose:
                print("RLDIS (Recursive Loop Detection and Interruption System) enabled")
        else:
            self.rldis = None
        
    def _get_effective_epsilon(self, state: np.ndarray) -> float:
        """
        Compute effective epsilon, possibly adapting to state magnitude.
        
        Args:
            state: Current state
            
        Returns:
            Adjusted epsilon value
        """
        if not self.adaptive_epsilon or not isinstance(state, np.ndarray):
            return self.epsilon
        
        # Scale epsilon by the magnitude of the state
        state_magnitude = np.max(np.abs(state))
        if state_magnitude < 1e-10:
            return self.epsilon
        
        return self.epsilon * state_magnitude
    
    def _evaluate_operator(self, state: Any) -> Any:
        """
        Evaluate the recursive operator with caching to avoid redundant calculations.
        
        Args:
            state: Input state
            
        Returns:
            Next state
        """
        # For numpy arrays, we need to hash a tuple representation
        if isinstance(state, np.ndarray):
            key = tuple(state.flatten())
        else:
            key = state
            
        if key in self.operator_cache:
            return self.operator_cache[key]
        
        result = self.recursive_operator(state)
        
        # Only cache if the key is hashable
        try:
            self.operator_cache[key] = result
        except:
            pass
            
        return result
    
    def _check_early_stopping(self, effective_epsilon: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Returns:
            True if early stopping criteria are met, False otherwise
        """
        if not self.early_stopping or len(self.tracer.distances) < self.early_stopping_window:
            return False
            
        # Check if the improvement is below the threshold for several iterations
        recent_distances = self.tracer.distances[-self.early_stopping_window:]
        improvements = np.abs(np.diff(recent_distances))
        if not np.all(improvements < self.early_stopping_threshold):
            return False
        
        last_distance = recent_distances[-1]
        threshold = max(effective_epsilon, self.early_stopping_threshold * 10)
        return last_distance < threshold
    
    def _check_divergence(self, distance: float) -> bool:
        """
        Check if the process appears to be diverging.
        
        Args:
            distance: Current distance between successive states
            
        Returns:
            True if divergence is detected, False otherwise
        """
        if not self.divergence_detection:
            return False
            
        return distance > self.divergence_threshold
    
    def _estimate_jacobian(self, state: np.ndarray, h: float = 1e-7) -> np.ndarray:
        """
        Estimate the Jacobian matrix at the given state using finite differences.
        
        Args:
            state: State at which to estimate the Jacobian
            h: Step size for finite differences
            
        Returns:
            Jacobian matrix
        """
        if not isinstance(state, np.ndarray):
            raise TypeError("Jacobian estimation requires numpy array state")
            
        n = len(state)
        jacobian = np.zeros((n, n))
        
        # Compute the Jacobian using finite differences
        for i in range(n):
            state_plus = state.copy()
            state_plus[i] += h
            next_state_plus = self._evaluate_operator(state_plus)
            
            # If output is not a numpy array, try to convert it
            if not isinstance(next_state_plus, np.ndarray):
                try:
                    next_state_plus = np.array(next_state_plus)
                except:
                    raise TypeError("Operator must return numpy-convertible output for Jacobian estimation")
            
            state_minus = state.copy()
            state_minus[i] -= h
            next_state_minus = self._evaluate_operator(state_minus)
            
            # If output is not a numpy array, try to convert it
            if not isinstance(next_state_minus, np.ndarray):
                try:
                    next_state_minus = np.array(next_state_minus)
                except:
                    raise TypeError("Operator must return numpy-convertible output for Jacobian estimation")
            
            # Central difference
            jacobian[:, i] = (next_state_plus - next_state_minus) / (2 * h)
            
        return jacobian
    
    def _classify_fixed_point(self, state: np.ndarray) -> FixedPointType:
        """
        Classify the stability type of a fixed point based on Jacobian eigenvalues.
        
        Args:
            state: Fixed point state
            
        Returns:
            FixedPointType indicating stability classification
        """
        if not isinstance(state, np.ndarray):
            return FixedPointType.UNKNOWN
            
        try:
            # Estimate Jacobian at the fixed point
            jacobian = self._estimate_jacobian(state)
            
            # Calculate eigenvalues of the Jacobian
            eigenvalues = linalg.eigvals(jacobian)
            abs_eigenvalues = np.abs(eigenvalues)
            
            # Check spectral radius to determine stability
            if np.all(abs_eigenvalues < 1.0):
                return FixedPointType.ATTRACTIVE
            elif np.all(abs_eigenvalues > 1.0):
                return FixedPointType.REPULSIVE
            elif np.all(abs_eigenvalues == 1.0):
                return FixedPointType.NEUTRAL
            else:
                return FixedPointType.SADDLE
                
        except Exception as e:
            if self.verbose:
                print(f"Error classifying fixed point: {e}")
            return FixedPointType.UNKNOWN
    
    def find_fixed_point(
        self, 
        initial_state: Any,
        return_trace: bool = False,
        classify_stability: bool = False
    ) -> Dict[str, Any]:
        """
        Find a fixed point of the recursive operator starting from the initial state.
        
        Args:
            initial_state: Initial state for recursion
            return_trace: Whether to include full trace in results
            classify_stability: Whether to classify fixed point stability
            
        Returns:
            Dictionary containing:
            - 'fixed_point': The fixed point state (or best approximation)
            - 'status': ConvergenceStatus indicating outcome
            - 'iterations': Number of iterations performed
            - 'final_distance': Final distance between successive states
            - 'trace': Full trace of states (if return_trace=True)
            - 'stability': FixedPointType classification (if classify_stability=True)
        """
        # Initialize state and tracer
        state = initial_state
        self.tracer = EigenrecursionTracer(self.state_dim)
        self.tracer.add_state(state)
        
        # Initialize results
        iterations = 0
        status = ConvergenceStatus.MAX_ITERATIONS_REACHED
        final_distance = None
        
        try:
            # Main recursion loop
            for i in range(self.max_iterations):
                iterations = i + 1
                
                # Apply the recursive operator
                next_state = self._evaluate_operator(state)
                
                # Calculate distance
                distance = self.distance_fn(next_state, state)
                self.tracer.add_state(next_state, distance)
                self.tracer.add_metric("distance", distance)
                final_distance = distance
                
                # Check for convergence
                effective_epsilon = self._get_effective_epsilon(state if isinstance(state, np.ndarray) else next_state)
                if distance < effective_epsilon:
                    status = ConvergenceStatus.CONVERGED
                    break
                
                # Check for cycles
                if self.cycle_detection and i >= self.cycle_window:
                    cycle_detected, cycle_length = CycleDetector.simple_lookup(
                        self.tracer.trace, self.cycle_window)
                    
                    if cycle_detected:
                        status = ConvergenceStatus.CYCLE_DETECTED
                        self.tracer.add_metric("cycle_length", cycle_length)
                        break
                
                # Check for early stopping
                if self._check_early_stopping(effective_epsilon):
                    status = ConvergenceStatus.CONVERGED
                    break
                
                # Check for divergence
                if self._check_divergence(distance):
                    status = ConvergenceStatus.DIVERGED
                    break
                
                # RLDIS Monitoring - Check for recursive loops
                if self.rldis:
                    trace_window = self.tracer.trace[-min(len(self.tracer.trace), 20):]  # Last 20 states
                    rldis_result = self.rldis.monitor_iteration(
                        trace=trace_window,
                        metadata={
                            'iteration': i + 1,
                            'distance': distance,
                            'effective_epsilon': effective_epsilon,
                            'computation_time': self.tracer.computation_times[-1] if self.tracer.computation_times else 0.0
                        }
                    )
                    
                    # Handle RLDIS intervention
                    if rldis_result['intervention_required']:
                        intervention_action = rldis_result.get('intervention_action', 'terminate')
                        
                        if intervention_action == 'terminate':
                            status = ConvergenceStatus.ERROR
                            if self.verbose:
                                print(f"RLDIS Intervention: Terminating due to {rldis_result.get('pattern_type', 'unknown pattern')}")
                            break
                        elif intervention_action == 'modify_state':
                            # Apply orthogonal transformation to break loops
                            if isinstance(next_state, np.ndarray):
                                perturbation = np.random.normal(0, 0.1, next_state.shape)
                                next_state = next_state + perturbation
                                if self.verbose:
                                    print(f"RLDIS Intervention: Applied orthogonal perturbation")
                        elif intervention_action == 'adaptive_epsilon':
                            # Dynamically adjust epsilon
                            self.epsilon *= 1.1
                            if self.verbose:
                                print(f"RLDIS Intervention: Adjusted epsilon to {self.epsilon}")
                
                # Update state
                state = next_state
                
                # Progress reporting
                if self.verbose and (i+1) % 10 == 0:
                    print(f"Iteration {i+1}: distance = {distance}")
                    
        except Exception as e:
            status = ConvergenceStatus.ERROR
            if self.verbose:
                print(f"Error during recursion: {e}")
        
        # Prepare results
        results = {
            "fixed_point": state,
            "status": status,
            "iterations": iterations,
            "final_distance": final_distance
        }
        
        # Add RLDIS diagnostics if enabled
        if self.rldis:
            results["rldis_diagnostics"] = {
                "monitoring_layers": {
                    layer.name: {
                        "detection_count": layer.detection_count,
                        "last_detection_time": layer.last_detection_time,
                        "is_active": layer.is_active
                    }
                    for layer in self.rldis.monitoring_layers
                },
                "global_status": self.rldis.get_system_status(),
                "intervention_history": self.rldis.intervention_history
            }
        
        # Add trace if requested
        if return_trace:
            results["trace"] = self.tracer.trace
            results["distances"] = self.tracer.distances
        
        # Classify stability if requested
        if classify_stability and isinstance(state, np.ndarray):
            results["stability"] = self._classify_fixed_point(state)
        
        return results
    
    def find_multiple_fixed_points(
        self, 
        initial_states: List[Any],
        parallel: bool = False,
        classify_stability: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find multiple fixed points starting from different initial states.
        
        Args:
            initial_states: List of initial states
            parallel: Whether to parallelize computation (not implemented yet)
            classify_stability: Whether to classify fixed point stability
            
        Returns:
            List of result dictionaries for each initial state
        """
        results = []
        
        if parallel:
            warnings.warn("Parallel computation not implemented, running sequentially")
            
        for i, initial_state in enumerate(initial_states):
            if self.verbose:
                print(f"Starting search from initial state {i+1}/{len(initial_states)}")
                
            result = self.find_fixed_point(
                initial_state, 
                return_trace=False,
                classify_stability=classify_stability
            )
            
            results.append(result)
            
        return results
    
    def visualize_results(self) -> None:
        """
        Visualize the results of the fixed point search.
        """
        self.tracer.visualize_convergence(title="Eigenrecursion Convergence Analysis")

    def analyze_basin_of_attraction(
        self, 
        center_state: np.ndarray, 
        radius: float,
        num_samples: int = 10,
        dimension: int = None
    ) -> Dict[str, Any]:
        """
        Analyze the basin of attraction around a fixed point.
        
        Args:
            center_state: Center point for sampling
            radius: Radius around center for sampling
            num_samples: Number of samples per dimension
            dimension: Dimensionality for sampling (defaults to state dimension)
            
        Returns:
            Dictionary with analysis results
        """
        if not isinstance(center_state, np.ndarray):
            raise TypeError("Basin analysis requires numpy array state")
            
        if dimension is None:
            dimension = len(center_state)
            
        # If dimension is too high, reduce sampling for computational feasibility
        if dimension > 3:
            warnings.warn(f"High-dimensional basin analysis (dim={dimension}) may be computationally expensive")
            
        # Generate grid of samples for 1D or 2D visualization
        if dimension == 1:
            # 1D sample points
            sample_points = np.linspace(center_state[0] - radius, center_state[0] + radius, num_samples)
            samples = np.zeros((num_samples, len(center_state)))
            
            for i, point in enumerate(sample_points):
                samples[i] = center_state.copy()
                samples[i, 0] = point
                
        elif dimension == 2:
            # 2D grid of sample points
            x = np.linspace(center_state[0] - radius, center_state[0] + radius, num_samples)
            y = np.linspace(center_state[1] - radius, center_state[1] + radius, num_samples)
            xx, yy = np.meshgrid(x, y)
            
            samples = np.zeros((num_samples * num_samples, len(center_state)))
            
            for i in range(num_samples):
                for j in range(num_samples):
                    idx = i * num_samples + j
                    samples[idx] = center_state.copy()
                    samples[idx, 0] = xx[i, j]
                    samples[idx, 1] = yy[i, j]
        else:
            # For higher dimensions, use random sampling
            samples = np.zeros((num_samples, len(center_state)))
            
            for i in range(num_samples):
                # Generate random direction vector
                direction = np.random.randn(len(center_state))
                direction = direction / np.linalg.norm(direction)
                
                # Generate random radius within the specified range
                r = radius * np.random.random()
                
                # Set the sample point
                samples[i] = center_state + r * direction
                
        # Run eigenrecursion for each sample
        results = self.find_multiple_fixed_points(
            samples.tolist(), 
            classify_stability=True
        )
        
        # Analyze convergence patterns
        converged_count = 0
        cycle_count = 0
        diverged_count = 0
        error_count = 0
        
        iterations_list = []
        final_distances = []
        
        for result in results:
            status = result['status']
            
            if status == ConvergenceStatus.CONVERGED:
                converged_count += 1
            elif status == ConvergenceStatus.CYCLE_DETECTED:
                cycle_count += 1
            elif status == ConvergenceStatus.DIVERGED:
                diverged_count += 1
            else:
                error_count += 1
                
            iterations_list.append(result['iterations'])
            final_distances.append(result['final_distance'])
            
        # Prepare analysis results
        analysis = {
            'converged_percentage': 100 * converged_count / len(results),
            'cycle_percentage': 100 * cycle_count / len(results),
            'diverged_percentage': 100 * diverged_count / len(results),
            'error_percentage': 100 * error_count / len(results),
            'average_iterations': np.mean(iterations_list),
            'median_iterations': np.median(iterations_list),
            'max_iterations': np.max(iterations_list),
            'average_final_distance': np.mean(final_distances),
            'sample_count': len(results),
            'dimension': dimension,
            'radius': radius
        }
        
        # Visualize for 1D or 2D cases
        if dimension <= 2:
            self._visualize_basin(samples, results, center_state, radius, dimension, num_samples)
            
        return analysis
    
    def _visualize_basin(
        self, 
        samples: np.ndarray, 
        results: List[Dict[str, Any]], 
        center: np.ndarray, 
        radius: float, 
        dimension: int, 
        num_samples: int
    ) -> None:
        """
        Visualize the basin of attraction.
        
        Args:
            samples: Sample points
            results: Results for each sample
            center: Center point
            radius: Sampling radius
            dimension: Dimensionality
            num_samples: Number of samples per dimension
        """
        # Prepare status colormap
        status_colors = {
            ConvergenceStatus.CONVERGED: 'green',
            ConvergenceStatus.CYCLE_DETECTED: 'blue',
            ConvergenceStatus.DIVERGED: 'red',
            ConvergenceStatus.MAX_ITERATIONS_REACHED: 'orange',
            ConvergenceStatus.ERROR: 'black',
            ConvergenceStatus.NUMERICAL_INSTABILITY: 'purple'
        }
        
        plt.figure(figsize=(12, 10))
        
        if dimension == 1:
            # 1D visualization
            x_values = samples[:, 0]
            
            # Plot iterations to converge
            plt.subplot(3, 1, 1)
            iterations = [r['iterations'] for r in results]
            plt.plot(x_values, iterations, 'o-')
            plt.axvline(x=center[0], color='r', linestyle='--', label='Center')
            plt.xlabel('Parameter Value')
            plt.ylabel('Iterations')
            plt.title('Iterations to Converge/Terminate')
            plt.grid(True)
            
            # Plot convergence status
            plt.subplot(3, 1, 2)
            for status in ConvergenceStatus:
                mask = [r['status'] == status for r in results]
                if any(mask):
                    plt.plot(x_values[mask], [0.5] * sum(mask), 'o', 
                             label=status.value, color=status_colors[status], markersize=10)
            
            plt.axvline(x=center[0], color='r', linestyle='--', label='Center')
            plt.xlabel('Parameter Value')
            plt.yticks([])
            plt.title('Convergence Status')
            plt.legend()
            plt.grid(True)
            
            # Plot final distance
            plt.subplot(3, 1, 3)
            distances = [r['final_distance'] for r in results]
            plt.semilogy(x_values, distances, 'o-')
            plt.axvline(x=center[0], color='r', linestyle='--', label='Center')
            plt.xlabel('Parameter Value')
            plt.ylabel('Final Distance (log scale)')
            plt.title('Final Distance')
            plt.grid(True)
            
        elif dimension == 2:
            # 2D visualization
            # Create a grid for visualization
            x = np.linspace(center[0] - radius, center[0] + radius, num_samples)
            y = np.linspace(center[1] - radius, center[1] + radius, num_samples)
            # xx, yy = np.meshgrid(x, y)  # Placeholder for 2D visualization
            
        plt.tight_layout()
        plt.show()

    def get_rldis_status(self) -> Dict[str, Any]:
        """
        Get comprehensive RLDIS system status and diagnostics.
        
        Returns:
            Dictionary containing RLDIS system status, monitoring layer states,
            and intervention history.
        """
        if not self.rldis:
            return {'rldis_enabled': False, 'message': 'RLDIS not enabled'}
            
        return {
            'rldis_enabled': True,
            'system_status': self.rldis.get_system_status(),
            'monitoring_layers': {
                layer.name: {
                    'detection_count': layer.detection_count,
                    'last_detection_time': layer.last_detection_time,
                    'is_active': layer.is_active,
                    'performance_metrics': getattr(layer, 'performance_metrics', {})
                }
                for layer in self.rldis.monitoring_layers
            },
            'intervention_history': self.rldis.intervention_history,
            'configuration': {
                'enabled_patterns': [pattern.value for pattern in RLDISPatternType],
                'severity_levels': [level.value for level in RLDISSeverityLevel],
                'intervention_priorities': [priority.value for priority in RLDISInterventionPriority]
            }
        }
        
    def reset_rldis(self) -> bool:
        """
        Reset RLDIS system state and clear intervention history.
        
        Returns:
            True if reset successful, False if RLDIS not enabled.
        """
        if not self.rldis:
            return False
            
        # Reset all monitoring layers
        for layer in self.rldis.monitoring_layers:
            layer.detection_count = 0
            layer.last_detection_time = None
            layer.is_active = True
            
        # Clear intervention history
        self.rldis.intervention_history = []
        
        # Reset system status
        self.rldis.system_status = {
            'total_interventions': 0,
            'active_monitoring': True,
            'last_reset': time.time(),
            'system_health': 'optimal'
        }
        
        return True


class EigenrecursionAlgorithm(Eigenrecursion):
    """
    Convenience wrapper so callers can instantiate the protocol with sensible defaults.
    """

    def __init__(self, recursive_operator: Optional[Callable[[Any], Any]] = None, **kwargs):
        if recursive_operator is None:
            recursive_operator = lambda state: state
        super().__init__(recursive_operator=recursive_operator, **kwargs)
