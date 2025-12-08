
"""
stability_matrix.py - Recursion Stability Node.

This module implements the stability matrix that manages recursive processing stability,
contradiction detection and resolution, and memory persistence.

author: Morpheus
date: 2025-04-27
version: 1.0.0
"""

import numpy as np
import torch
import uuid
import math
import os
import json
import logging
import time
import threading
import queue
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from collections import defaultdict, deque
from contextlib import contextmanager
import copy

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StabilityMatrix")

PHI = (1 + math.sqrt(5)) / 2
TAU = 2 * math.pi
EULER = math.e
FEIGENBAUM = 4.669201609
PSALTER_SCALE = math.sqrt(PHI/TAU)
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

class BreathPhase(Enum):
    INHALE = auto()
    PAUSE_RISING = auto()
    HOLD = auto()
    PAUSE_FALLING = auto()
    EXHALE = auto()
    REST = auto()
    DREAM = auto()
    
    def next_phase(self) -> 'BreathPhase':
        phases = list(BreathPhase)
        return phases[(phases.index(self) + 1) % len(phases)]
        
    @property
    def duration_weight(self) -> float:
        weights = {
            BreathPhase.INHALE: 1.0,
            BreathPhase.PAUSE_RISING: 0.3,
            BreathPhase.HOLD: 1.0,
            BreathPhase.PAUSE_FALLING: 0.3,
            BreathPhase.EXHALE: 1.2,
            BreathPhase.REST: 0.8,
            BreathPhase.DREAM: 1.5,
        }
        return weights.get(self, 1.0)

class ContradictionType(Enum):
    LOGICAL = auto()
    ETHICAL = auto()
    ONTOLOGICAL = auto()
    IDENTITY = auto()
    MOTIVATIONAL = auto()
    COGNITIVE = auto()
    TEMPORAL = auto()
    PERCEPTUAL = auto()
    EMOTIONAL = auto()
    EPISTEMOLOGICAL = auto()

class AbstractionLevel(Enum):
    CONCRETE_INSTANCE = 0
    PATTERN = 1
    PRINCIPLE = 2
    PARADIGM = 3
    META_PARADIGM = 4
    FOUNDATIONAL = 5
    TRANSCENDENT = 6

class RecursionPattern(Enum):
    SIMPLE_REPETITION = auto()
    CONTRADICTION_SPIRAL = auto()
    SELF_REFERENCE_LOOP = auto()
    INFINITE_CLARIFICATION = auto()
    DEFINITIONAL_REGRESSION = auto()
    CIRCULAR_REASONING = auto()
    ESCALATING_COMPLEXITY = auto()
    HALTING_PROBLEM = auto()

class ARFSDimension(Enum):
    X = auto()
    Y = auto()
    Z = auto()
    T = auto()
    I = auto()
    P = auto()

@dataclass
class Contradiction:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ContradictionType = ContradictionType.LOGICAL
    description: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    tension_degree: float = 0.0
    timestamp: float = field(default_factory=time.time)
    resolution_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_path: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "description": self.description,
            "source": self.source,
            "target": self.target,
            "tension_degree": self.tension_degree,
            "timestamp": self.timestamp,
            "resolution_attempts": self.resolution_attempts,
            "resolved": self.resolved,
            "resolution_path": self.resolution_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contradiction':
        data = copy.deepcopy(data)
        if isinstance(data.get("type"), str):
            data["type"] = ContradictionType[data["type"]]
        return cls(**data)

@dataclass
class RecursionEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: RecursionPattern = RecursionPattern.SIMPLE_REPETITION
    description: str = ""
    sequence: List[Dict[str, Any]] = field(default_factory=list)
    first_detected: float = field(default_factory=time.time)
    last_detected: float = field(default_factory=time.time)
    detection_count: int = 1
    severity: float = 0.0
    resource_impact: float = 0.0
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    
    def update_detection(self) -> None:
        self.last_detected = time.time()
        self.detection_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.name,
            "description": self.description,
            "sequence": self.sequence,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "detection_count": self.detection_count,
            "severity": self.severity,
            "resource_impact": self.resource_impact,
            "interventions": self.interventions,
            "resolved": self.resolved
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecursionEvent':
        data = copy.deepcopy(data)
        if isinstance(data.get("pattern_type"), str):
            data["pattern_type"] = RecursionPattern[data["pattern_type"]]
        return cls(**data)

@dataclass
class AbstractionState:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: AbstractionLevel = AbstractionLevel.CONCRETE_INSTANCE
    content: Dict[str, Any] = field(default_factory=dict)
    coherence_score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.name,
            "level_value": self.level.value,
            "content": self.content,
            "coherence_score": self.coherence_score,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AbstractionState':
        data = copy.deepcopy(data)
        if isinstance(data.get("level"), str):
            data["level"] = AbstractionLevel[data["level"]]
        elif "level_value" in data:
            data["level"] = AbstractionLevel(data["level_value"])
            data.pop("level_value", None)
        return cls(**data)

@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: Dict[str, Any] = field(default_factory=dict)
    content_type: str = "general"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    emotional_valence: float = 0.0
    tags: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    created_by: str = "system"
    ethical_evaluation: Optional[Dict[str, float]] = None
    arfs_dimensions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "content_type": self.content_type,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags,
            "associations": self.associations,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved,
            "created_by": self.created_by,
            "ethical_evaluation": self.ethical_evaluation,
            "arfs_dimensions": self.arfs_dimensions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(**copy.deepcopy(data))

class RecursionDetector:
    def __init__(self, 
                 max_history_size: int = 100,
                 similarity_threshold: float = 0.85,
                 cycle_detection_threshold: int = 3,
                 severity_thresholds: Optional[Dict[RecursionPattern, float]] = None):
        self.processing_history = deque(maxlen=max_history_size)
        self.similarity_threshold = similarity_threshold
        self.cycle_detection_threshold = cycle_detection_threshold
        self.detected_patterns = {}
        
        self.severity_thresholds = {
            RecursionPattern.SIMPLE_REPETITION: 0.6,
            RecursionPattern.CONTRADICTION_SPIRAL: 0.8,
            RecursionPattern.SELF_REFERENCE_LOOP: 0.9,
            RecursionPattern.INFINITE_CLARIFICATION: 0.7,
            RecursionPattern.DEFINITIONAL_REGRESSION: 0.85,
            RecursionPattern.CIRCULAR_REASONING: 0.75,
            RecursionPattern.ESCALATING_COMPLEXITY: 0.65,
            RecursionPattern.HALTING_PROBLEM: 0.95
        }
        
        if severity_thresholds:
            for pattern, threshold in severity_thresholds.items():
                self.severity_thresholds[pattern] = threshold
        
        self.stats = {
            "total_operations": 0,
            "pattern_detections": 0,
            "interventions": 0,
            "pattern_counts": {pattern: 0 for pattern in RecursionPattern}
        }
        
        logger.info(f"RecursionDetector initialized with threshold: {similarity_threshold:.2f}")
    
    def record_operation(self, operation_data: Dict[str, Any]) -> None:
        if "timestamp" not in operation_data:
            operation_data["timestamp"] = time.time()
            
        self.processing_history.append(operation_data)
        self.stats["total_operations"] += 1
    
    def detect_patterns(self) -> List[RecursionEvent]:
        if len(self.processing_history) < 2:
            return []
            
        new_patterns = []
        
        self._detect_simple_repetition(new_patterns)
        self._detect_contradiction_spiral(new_patterns)
        self._detect_self_reference_loop(new_patterns)
        self._detect_infinite_clarification(new_patterns)
        self._detect_escalating_complexity(new_patterns)
        
        self.stats["pattern_detections"] += len(new_patterns)
        for pattern in new_patterns:
            self.stats["pattern_counts"][pattern.pattern_type] += 1
            self.detected_patterns[pattern.id] = pattern
            
        return new_patterns
    
    def _detect_simple_repetition(self, new_patterns: List[RecursionEvent]) -> None:
        history = list(self.processing_history)
        for i in range(len(history) - 1):
            repetition_count = 1
            similar_operations = []
            
            for j in range(i + 1, len(history)):
                if self._is_similar(history[i], history[j]):
                    repetition_count += 1
                    similar_operations.append(history[j])
                    
                    if repetition_count >= self.cycle_detection_threshold:
                        pattern = RecursionEvent(
                            pattern_type=RecursionPattern.SIMPLE_REPETITION,
                            description=f"Content repeated {repetition_count} times",
                            sequence=[history[i]] + similar_operations,
                            severity=min(0.3 + (repetition_count * 0.1), 1.0),
                            resource_impact=min(0.2 + (repetition_count * 0.05), 1.0)
                        )
                        
                        if not self._is_known_pattern(pattern):
                            new_patterns.append(pattern)
                        break
    
    def _detect_contradiction_spiral(self, new_patterns: List[RecursionEvent]) -> None:
        history = list(self.processing_history)
        if len(history) < 4:
            return
            
        for i in range(len(history) - 3):
            if (self._is_similar(history[i], history[i+2]) and 
                self._is_similar(history[i+1], history[i+3]) and
                self._is_contradictory(history[i], history[i+1])):
                
                oscillation_count = 2
                oscillation_sequence = [history[i], history[i+1], history[i+2], history[i+3]]
                
                for j in range(i + 4, len(history), 2):
                    if j + 1 < len(history):
                        if (self._is_similar(history[i], history[j]) and
                            self._is_similar(history[i+1], history[j+1])):
                            oscillation_count += 1
                            oscillation_sequence.extend([history[j], history[j+1]])
                        else:
                            break
                
                if oscillation_count >= 2:
                    pattern = RecursionEvent(
                        pattern_type=RecursionPattern.CONTRADICTION_SPIRAL,
                        description=f"Oscillating between contradictory positions {oscillation_count} times",
                        sequence=oscillation_sequence,
                        severity=min(0.5 + (oscillation_count * 0.15), 1.0),
                        resource_impact=min(0.3 + (oscillation_count * 0.1), 1.0)
                    )
                    
                    if not self._is_known_pattern(pattern):
                        new_patterns.append(pattern)
    
    def _detect_self_reference_loop(self, new_patterns: List[RecursionEvent]) -> None:
        history = list(self.processing_history)
        
        for i in range(len(history) - 2):
            if "references" in history[i] and "self_reference_depth" in history[i]:
                reference_depths = [history[i]["self_reference_depth"]]
                reference_sequence = [history[i]]
                
                for j in range(i + 1, len(history)):
                    if ("references" in history[j] and 
                        "self_reference_depth" in history[j] and
                        history[j]["self_reference_depth"] > reference_depths[-1]):
                        
                        reference_depths.append(history[j]["self_reference_depth"])
                        reference_sequence.append(history[j])
                        
                        if len(reference_depths) >= 3:
                            pattern = RecursionEvent(
                                pattern_type=RecursionPattern.SELF_REFERENCE_LOOP,
                                description=f"Self-reference loop with {len(reference_depths)} levels of recursion",
                                sequence=reference_sequence,
                                severity=min(0.6 + (len(reference_depths) * 0.1), 1.0),
                                resource_impact=min(0.4 + (len(reference_depths) * 0.15), 1.0)
                            )
                            
                            if not self._is_known_pattern(pattern):
                                new_patterns.append(pattern)
                            break
    
    def _detect_infinite_clarification(self, new_patterns: List[RecursionEvent]) -> None:
        history = list(self.processing_history)
        
        for i in range(len(history) - 3):
            if "refinement_markers" in history[i]:
                refinement_sequence = [history[i]]
                refinement_count = 1
                
                for j in range(i + 1, len(history)):
                    if "refinement_markers" in history[j]:
                        refinement_sequence.append(history[j])
                        refinement_count += 1
                        
                        if refinement_count >= 4:
                            pattern = RecursionEvent(
                                pattern_type=RecursionPattern.INFINITE_CLARIFICATION,
                                description=f"Continuous refinement without conclusion ({refinement_count} iterations)",
                                sequence=refinement_sequence,
                                severity=min(0.4 + (refinement_count * 0.08), 1.0),
                                resource_impact=min(0.3 + (refinement_count * 0.07), 1.0)
                            )
                            
                            if not self._is_known_pattern(pattern):
                                new_patterns.append(pattern)
                            break
                    else:
                        break
    
    def _detect_escalating_complexity(self, new_patterns: List[RecursionEvent]) -> None:
        history = list(self.processing_history)
        
        for i in range(len(history) - 3):
            if "complexity_metrics" in history[i]:
                complexity_sequence = [history[i]]
                complexity_values = [history[i]["complexity_metrics"].get("overall", 0)]
                
                for j in range(i + 1, len(history)):
                    if "complexity_metrics" in history[j]:
                        current_complexity = history[j]["complexity_metrics"].get("overall", 0)
                        
                        if current_complexity > complexity_values[-1]:
                            complexity_sequence.append(history[j])
                            complexity_values.append(current_complexity)
                            
                            if (len(complexity_values) >= 4 and
                                complexity_values[-1] > 2 * complexity_values[0]):
                                
                                pattern = RecursionEvent(
                                    pattern_type=RecursionPattern.ESCALATING_COMPLEXITY,
                                    description=f"Escalating complexity without resolution (from {complexity_values[0]:.2f} to {complexity_values[-1]:.2f})",
                                    sequence=complexity_sequence,
                                    severity=min(0.3 + (0.2 * (complexity_values[-1] / complexity_values[0])), 1.0),
                                    resource_impact=min(0.3 + (0.3 * (complexity_values[-1] / complexity_values[0])), 1.0)
                                )
                                
                                if not self._is_known_pattern(pattern):
                                    new_patterns.append(pattern)
                                break
                        else:
                            break
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[RecursionEvent]:
        return self.detected_patterns.get(pattern_id)
    
    def mark_pattern_resolved(self, pattern_id: str, resolution_info: Dict[str, Any]) -> bool:
        if pattern_id in self.detected_patterns:
            pattern = self.detected_patterns[pattern_id]
            pattern.resolved = True
            pattern.interventions.append(resolution_info)
            return True
        return False
    
    def get_all_patterns(self, include_resolved: bool = False) -> List[RecursionEvent]:
        if include_resolved:
            return list(self.detected_patterns.values())
        else:
            return [p for p in self.detected_patterns.values() if not p.resolved]
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats
    
    def _is_similar(self, op1: Dict[str, Any], op2: Dict[str, Any]) -> bool:
        if "content" in op1 and "content" in op2:
            if isinstance(op1["content"], str) and isinstance(op2["content"], str):
                return self._string_similarity(op1["content"], op2["content"]) >= self.similarity_threshold
            elif isinstance(op1["content"], dict) and isinstance(op2["content"], dict):
                return self._dict_similarity(op1["content"], op2["content"]) >= self.similarity_threshold
        
        if op1.get("type") != op2.get("type"):
            return False
            
        return op1.get("id") == op2.get("id")
    
    def _is_contradictory(self, op1: Dict[str, Any], op2: Dict[str, Any]) -> bool:
        if "contradicts" in op1 and op1["contradicts"] == op2.get("id"):
            return True
        if "contradicts" in op2 and op2["contradicts"] == op1.get("id"):
            return True
            
        if "value" in op1 and "value" in op2:
            if isinstance(op1["value"], (int, float)) and isinstance(op2["value"], (int, float)):
                if op1["value"] * op2["value"] < 0:
                    return True
        
        if "sentiment" in op1 and "sentiment" in op2:
            if isinstance(op1["sentiment"], (int, float)) and isinstance(op2["sentiment"], (int, float)):
                if op1["sentiment"] * op2["sentiment"] < 0:
                    return True
        
        return False
    
    def _is_known_pattern(self, pattern: RecursionEvent) -> bool:
        for existing_id, existing_pattern in self.detected_patterns.items():
            if existing_pattern.resolved:
                continue
                
            if existing_pattern.pattern_type != pattern.pattern_type:
                continue
                
            if len(existing_pattern.sequence) != len(pattern.sequence):
                continue
                
            match = True
            for i in range(len(pattern.sequence)):
                if not self._is_similar(existing_pattern.sequence[i], pattern.sequence[i]):
                    match = False
                    break
                    
            if match:
                existing_pattern.update_detection()
                return True
                
        return False
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # len_diff = abs(len(str1) - len(str2)) # Unused variable
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class EigenrecursionStabilizer:
    """
    Implements the Eigenrecursion Stabilizer based on the Eigenrecursion Theorem.
    
    This class provides stability mechanisms for recursive processes by detecting
    and controlling convergence to fixed points (eigenstates). It leverages concepts
    from fixed-point theory, eigenvalue decomposition, and recursive function theory
    to ensure recursive stability.
    """
    
    def __init__(self, 
                 state_dimension: int = 256,
                 contraction_factor: float = 0.65,
                 convergence_threshold: float = 1e-5,
                 max_iterations: int = 1000,
                 learning_rate: float = 0.01):
        """
        Initialize the EigenrecursionStabilizer.
        
        Args:
            state_dimension: Dimension of the state vector
            contraction_factor: Lipschitz constant for the contraction mapping
            convergence_threshold: Threshold for determining convergence
            max_iterations: Maximum number of iterations before timeout
            learning_rate: Learning rate for adaptive adjustments
        """
        self.state_dimension = state_dimension
        self.contraction_factor = contraction_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
        # Initialize the recursive operator parameters
        self.weight = torch.nn.Parameter(
            torch.randn(state_dimension, state_dimension) * 0.1
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(state_dimension)
        )
        
        # Initialize state tracking variables
        self.current_state = torch.zeros(state_dimension)
        self.previous_state = torch.zeros(state_dimension)
        self.fixed_point = None
        self.convergence_history = []
        self.iterations_to_converge = 0
        self.converged = False
        
        # Stability metrics
        self.spectral_radius = 0.0
        self.lyapunov_exponent = 0.0
        self.stability_margin = 0.0
        
        logger.info(f"EigenrecursionStabilizer initialized with dimension: {state_dimension}")
    
    def recursive_operator(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply the recursive operator Γ(s) to the given state.
        
        The operator is a contraction mapping that ensures convergence
        toward a fixed point (eigenstate).
        
        Args:
            state: Current state tensor
            
        Returns:
            Next state after applying the recursive operator
        """
        # Apply tanh activation to ensure bounded output
        return torch.tanh(torch.matmul(self.weight, state) + self.bias)
    
    def compute_eigenstate(self, initial_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Compute the eigenstate by iteratively applying the recursive operator.
        
        Args:
            initial_state: Optional starting state, will use current state if None
            
        Returns:
            Dictionary with eigenstate information, convergence status, and metrics
        """
        if initial_state is not None:
            self.current_state = initial_state
        
        self.convergence_history = []
        
        for i in range(self.max_iterations):
            self.previous_state = self.current_state.clone()
            self.current_state = self.recursive_operator(self.previous_state)
            
            # Calculate distance between successive states
            distance = torch.norm(self.current_state - self.previous_state).item()
            self.convergence_history.append(distance)
            
            # Check for convergence
            if distance < self.convergence_threshold:
                self.converged = True
                self.iterations_to_converge = i + 1
                self.fixed_point = self.current_state.clone()
                self._update_stability_metrics()
                logger.info(f"Eigenstate found after {i+1} iterations (distance: {distance:.8f})")
                return {
                    "fixed_point": self.fixed_point,
                    "convergence_status": "CONVERGED",
                    "iterations": i + 1,
                    "final_distance": distance,
                    "stability_metrics": self._get_stability_metrics()
                }
            
        # If we reach here, we didn't converge within max_iterations
        self.converged = False
        logger.warning(f"Failed to find eigenstate after {self.max_iterations} iterations")
        return {
            "fixed_point": self.current_state,
            "convergence_status": "MAX_ITERATIONS_REACHED",
            "iterations": self.max_iterations,
            "final_distance": self.convergence_history[-1] if self.convergence_history else float('inf'),
            "stability_metrics": self._get_stability_metrics()
        }
    
    def check_stability(self, state: torch.Tensor, perturbation_scale: float = 0.01) -> Dict[str, Any]:
        """
        Check stability of a state by applying small perturbations.
        
        Args:
            state: State to check for stability
            perturbation_scale: Scale of random perturbations
            
        Returns:
            Stability assessment metrics
        """
        # Generate perturbed versions of the state
        perturbed_states = [
            state + torch.randn_like(state) * perturbation_scale
            for _ in range(10)
        ]
        
        # Track convergence for each perturbation
        stability_results = []
        
        for p_state in perturbed_states:
            current = p_state
            for i in range(100):  # Apply recursive operator 100 times
                next_state = self.recursive_operator(current)
                if torch.norm(next_state - current) < self.convergence_threshold:
                    stability_results.append({
                        "converged": True,
                        "iterations": i + 1,
                        "distance_from_original_fixed_point": torch.norm(next_state - state).item()
                    })
                    break
                current = next_state
            else:
                stability_results.append({
                    "converged": False,
                    "iterations": 100,
                    "distance_from_original_fixed_point": torch.norm(current - state).item()
                })
        
        # Calculate stability metrics
        convergence_rate = sum(1 for r in stability_results if r["converged"]) / len(stability_results)
        avg_iterations = sum(r["iterations"] for r in stability_results) / len(stability_results)
        avg_distance = sum(r["distance_from_original_fixed_point"] for r in stability_results) / len(stability_results)
        
        return {
            "stability_score": convergence_rate,
            "average_iterations_to_converge": avg_iterations,
            "average_distance_from_original": avg_distance,
            "perturbation_scale": perturbation_scale
        }
    
    def update_operator(self, 
                        new_weights: Optional[torch.Tensor] = None, 
                        new_bias: Optional[torch.Tensor] = None,
                        learning_rate: Optional[float] = None) -> None:
        """
        Update the recursive operator parameters.
        
        Args:
            new_weights: New weight matrix
            new_bias: New bias vector
            learning_rate: Learning rate for the update
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        if new_weights is not None:
            self.weight = torch.nn.Parameter(
                (1 - self.learning_rate) * self.weight + self.learning_rate * new_weights
            )
        
        if new_bias is not None:
            self.bias = torch.nn.Parameter(
                (1 - self.learning_rate) * self.bias + self.learning_rate * new_bias
            )
        
        # Ensure contractivity by normalizing the weight matrix
        with torch.no_grad():
            # Calculate largest singular value
            U, S, V = torch.svd(self.weight)
            max_singular_value = S[0].item()
            
            # Ensure max singular value is below contraction factor
            if max_singular_value > self.contraction_factor:
                scale_factor = self.contraction_factor / max_singular_value
                self.weight.data *= scale_factor
        
        self._update_stability_metrics()
        logger.debug(f"Updated recursive operator (spectral radius: {self.spectral_radius:.4f})")
    
    def adapt_to_ethical_manifold(self, ethical_gradient: torch.Tensor) -> None:
        """
        Adapt the recursive operator to align with ethical gradient.
        
        Args:
            ethical_gradient: Gradient from ethical manifold
        """
        # Normalize the gradient
        ethical_norm = torch.norm(ethical_gradient)
        if ethical_norm > 1e-8:  # Avoid division by zero
            normalized_gradient = ethical_gradient / ethical_norm
        else:
            return
        
        # Project the gradient onto the weight matrix
        # Reshape gradient to match the weight matrix if needed
        if ethical_gradient.shape[0] == self.state_dimension:
            gradient_matrix = torch.outer(normalized_gradient, normalized_gradient)
        else:
            gradient_matrix = normalized_gradient.reshape(self.weight.shape)
        
        # Update weights to follow ethical gradient while preserving contractivity
        self.update_operator(
            new_weights=self.weight + self.learning_rate * gradient_matrix,
            new_bias=self.bias + self.learning_rate * normalized_gradient,
        )
        
        logger.debug(f"Adapted operator to ethical manifold (gradient norm: {ethical_norm:.4f})")
    
    def is_fixed_point(self, state: torch.Tensor, tolerance: float = 1e-5) -> bool:
        """
        Check if a state is a fixed point of the recursive operator.
        
        Args:
            state: State to check
            tolerance: Tolerance for fixed-point check
            
        Returns:
            True if state is a fixed point, False otherwise
        """
        next_state = self.recursive_operator(state)
        distance = torch.norm(next_state - state).item()
        return distance < tolerance
    
    def reset(self) -> None:
        """Reset the stabilizer state."""
        self.current_state = torch.zeros(self.state_dimension)
        self.previous_state = torch.zeros(self.state_dimension)
        self.fixed_point = None
        self.convergence_history = []
        self.iterations_to_converge = 0
        self.converged = False
    
    def _update_stability_metrics(self) -> None:
        """Update internal stability metrics."""
        # Calculate spectral radius (largest eigenvalue)
        try:
            eigenvalues = torch.linalg.eigvals(self.weight)
            self.spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        except Exception:
            # Fallback method if eigenvalues fail
            self.spectral_radius = torch.norm(self.weight, p=2).item()
        
        # Estimate Lyapunov exponent
        if len(self.convergence_history) > 1:
            # Lambda = lim(t->∞) 1/t ln(|δΦ(t)|/|δΦ(0)|)
            delta_t = len(self.convergence_history)
            delta_phi_t = self.convergence_history[-1]
            delta_phi_0 = self.convergence_history[0]
            
            if delta_phi_0 > 0:
                self.lyapunov_exponent = (1.0 / delta_t) * math.log(delta_phi_t / delta_phi_0)
            else:
                self.lyapunov_exponent = -float('inf')
        
        # Calculate stability margin (how far from chaos boundary)
        self.stability_margin = abs(1.0 - self.spectral_radius)
    
    def _get_stability_metrics(self) -> Dict[str, float]:
        """Get stability metrics dictionary."""
        return {
            "spectral_radius": self.spectral_radius,
            "lyapunov_exponent": self.lyapunov_exponent,
            "stability_margin": self.stability_margin,
            "iterations_to_converge": self.iterations_to_converge,
            "converged": self.converged
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the stabilizer."""
        return {
            "current_state": self.current_state.tolist(),
            "fixed_point": self.fixed_point.tolist() if self.fixed_point is not None else None,
            "converged": self.converged,
            "iterations_to_converge": self.iterations_to_converge,
            "stability_metrics": self._get_stability_metrics()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stabilizer to a dictionary for serialization."""
        return {
            "state_dimension": self.state_dimension,
            "contraction_factor": self.contraction_factor,
            "convergence_threshold": self.convergence_threshold,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "weight": self.weight.detach().tolist(),
            "bias": self.bias.detach().tolist(),
            "current_state": self.current_state.tolist(),
            "spectral_radius": self.spectral_radius,
            "lyapunov_exponent": self.lyapunov_exponent,
            "stability_margin": self.stability_margin,
            "converged": self.converged,
            "iterations_to_converge": self.iterations_to_converge
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EigenrecursionStabilizer':
        """Create an EigenrecursionStabilizer from a dictionary."""
        stabilizer = cls(
            state_dimension=data.get("state_dimension", 256),
            contraction_factor=data.get("contraction_factor", 0.65),
            convergence_threshold=data.get("convergence_threshold", 1e-5),
            max_iterations=data.get("max_iterations", 1000),
            learning_rate=data.get("learning_rate", 0.01)
        )
        stabilizer.weight = torch.nn.Parameter(torch.tensor(data.get("weight")))
        stabilizer.bias = torch.nn.Parameter(torch.tensor(data.get("bias")))
        stabilizer.current_state = torch.tensor(data.get("current_state"))
        stabilizer.spectral_radius = data.get("spectral_radius", 0.0)
        stabilizer.lyapunov_exponent = data.get("lyapunov_exponent", 0.0)
        stabilizer.stability_margin = data.get("stability_margin", 0.0)
        stabilizer.converged = data.get("converged", False)
        stabilizer.iterations_to_converge = data.get("iterations_to_converge", 0)

        return stabilizer
        """Create an EigenrecursionStabilizer from a dictionary."""
        stabilizer = cls(
            state_dimension=data.get("state_dimension", 256),
            contraction_factor=data.get("contraction_factor", 0.65),
            convergence_threshold=data.get("convergence_threshold", 1e-5),
            max_iterations=data.get("max_iterations", 1000),
            learning_rate=data.get("learning_rate", 0.01)
        )
        
        # Load weights and bias if available
        if "weight" in data:
            stabilizer.weight = torch.nn.Parameter(torch.tensor(data["weight"]))
        
        if "bias" in data:
            stabilizer.bias = torch.nn.Parameter(torch.tensor(data["bias"]))
        
        # Load current state if available
        if "current_state" in data:
            stabilizer.current_state = torch.tensor(data["current_state"])
        
        # Load stability metrics if available
        stabilizer.spectral_radius = data.get("spectral_radius", 0.0)
        stabilizer.lyapunov_exponent = data.get("lyapunov_exponent", 0.0)
        stabilizer.stability_margin = data.get("stability_margin", 0.0)
        stabilizer.converged = data.get("converged", False)
        stabilizer.iterations_to_converge = data.get("iterations_to_converge", 0)

        return stabilizer

    def _initialize_contradiction_detector(self, detection_threshold: float, tension_scale: float, contradiction_db_path: str) -> None:
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def _load_contradiction_database(self) -> None:
        """Load contradiction database from file if it exists."""
        if not self.contradiction_db_path or not os.path.exists(self.contradiction_db_path):
            logger.info(f"No contradiction database found at {self.contradiction_db_path}")
            return
            
        try:
            with open(self.contradiction_db_path, 'rb') as f:
                data = pickle.load(f)
                
            if 'active' in data:
                self.active_contradictions = {
                    c_id: Contradiction.from_dict(c_data) if isinstance(c_data, dict) else c_data
                    for c_id, c_data in data['active'].items()
                }
                
            if 'resolved' in data:
                self.resolved_contradictions = {
                    c_id: Contradiction.from_dict(c_data) if isinstance(c_data, dict) else c_data
                    for c_id, c_data in data['resolved'].items()
                }
                
            if 'stats' in data:
                self.detection_stats = data['stats']
                
            if 'belief_state' in data:
                self.belief_state = data['belief_state']
                
            if 'value_state' in data:
                self.value_state = data['value_state']
                
            logger.info(f"Loaded contradiction database from {self.contradiction_db_path}")
            logger.info(f"Active contradictions: {len(self.active_contradictions)}, Resolved: {len(self.resolved_contradictions)}")
        except Exception as e:
            logger.error(f"Error loading contradiction database: {e}")
    
    def _store_contradiction_database(self) -> None:
        """Store contradiction database to file."""
        if not self.contradiction_db_path:
            logger.warning("No contradiction database path specified, skipping storage")
            return
            
        try:
            # Prepare data for storage
            data = {
                'active': {
                    c_id: c.to_dict() if hasattr(c, 'to_dict') else c
                    for c_id, c in self.active_contradictions.items()
                },
                'resolved': {
                    c_id: c.to_dict() if hasattr(c, 'to_dict') else c
                    for c_id, c in self.resolved_contradictions.items()
                },
                'stats': self.detection_stats,
                'belief_state': self.belief_state,
                'value_state': self.value_state,
                'timestamp': time.time()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.contradiction_db_path), exist_ok=True)
            
            # Store data
            with open(self.contradiction_db_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Stored contradiction database to {self.contradiction_db_path}")
        except Exception as e:
            logger.error(f"Error storing contradiction database: {e}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # len_diff = abs(len(str1) - len(str2)) # Unused variable
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
                        if jaccard_dist > self.ontological_tension_threshold:
                            tension_degree = jaccard_dist
                            
                            contradiction = Contradiction(
                                type=ContradictionType.ONTOLOGICAL,
                                description=f"Relational contradiction in concept '{concept_key}'",
                                source={
                                    "statement_id": statement_id,
                                    "concept_relations": new_relations
                                },
                                target={
                                    "existing_relations": old_relations
                                },
                                tension_degree=tension_degree,
                                timestamp=statement_timestamp
                            )
                            contradictions.append(contradiction)
    
    def _update_belief_state(self, statement: Dict[str, Any]) -> None:
        content = statement.get("content", {})
        
        assertions = content.get("assertions", [])
        for assertion in assertions:
            assertion_key = assertion.get("key", "")
            if assertion_key:
                if "beliefs" not in self.belief_state:
                    self.belief_state["beliefs"] = {}
                
                self.belief_state["beliefs"][assertion_key] = {
                    **assertion,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        values = content.get("values", {})
        for value_key, value_data in values.items():
            self.value_state[value_key] = {
                **value_data,
                "source_statement": statement.get("id", ""),
                "update_timestamp": time.time()
            }
        
        identity = content.get("identity", {})
        if identity:
            if "identity" not in self.belief_state:
                self.belief_state["identity"] = {}
            
            for identity_key, identity_value in identity.items():
                self.belief_state["identity"][identity_key] = identity_value
        
        events = content.get("events", [])
        for event in events:
            event_id = event.get("id", "")
            if event_id:
                if "events" not in self.belief_state:
                    self.belief_state["events"] = {}
                
                self.belief_state["events"][event_id] = {
                    **event,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
        
        concepts = content.get("concepts", {})
        if concepts:
            if "concepts" not in self.belief_state:
                self.belief_state["concepts"] = {}
            
            for concept_key, concept_data in concepts.items():
                self.belief_state["concepts"][concept_key] = {
                    **concept_data,
                    "source_statement": statement.get("id", ""),
                    "update_timestamp": time.time()
                }
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        if val1 is None and val2 is None:
            return True
        
        if type(val1) != type(val2):
            return False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 1e-6
        
        return val1 == val2
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        common_prefix = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                common_suffix += 1
            else:
                break
        
        common_chars = common_prefix + common_suffix
        return common_chars / max_len
    
    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
            
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        
        key_similarity = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        value_similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                value_similarities.append(self._string_similarity(val1, val2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    value_similarities.append(1.0)
                else:
                    value_similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                value_similarities.append(self._dict_similarity(val1, val2))
            else:
                value_similarities.append(1.0 if val1 == val2 else 0.0)
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 1.0
        return 0.7 * key_similarity + 0.3 * value_similarity

class ContradictionDetector:
    def __init__(self, 
                 contradiction_db_path: Optional[str] = None,
                 detection_threshold: float = 0.75,
                 tension_scale: float = 0.1):
        self.detection_threshold = detection_threshold
        self.tension_scale = tension_scale
        self.contradiction_db_path = contradiction_db_path
        self.active_contradictions: Dict[str, Contradiction] = {}
        self.resolved_contradictions: Dict[str, Contradiction] = {}
        self.contradiction_history = deque(maxlen=1000)
        self.detection_stats = {
            "total_detections": 0,
            "false_positives": 0,
            "resolutions": 0,
            "by_type": {ct: 0 for ct in ContradictionType}
        }
        
        self.belief_state: Dict[str, Dict[str, Any]] = {}
        self.value_state: Dict[str, Dict[str, float]] = {}
        self.statement_cache: Dict[str, Dict[str, Any]] = {}
        
        self.similarity_threshold = 0.85
        self.paradox_threshold = 0.92
        self.ontological_tension_threshold = 0.80
        
        self.temporal_window_size = 3600
        
        self._load_contradiction_database()
        
        logger.info(f"ContradictionDetector initialized with threshold: {detection_threshold:.2f}")
    
    def process_statement(self, statement: Dict[str, Any]) -> Optional[Contradiction]:
        statement_id = statement.get("id", str(uuid.uuid4())[:8])
        statement_timestamp = statement.get("timestamp", time.time())
        
        self.statement_cache[statement_id] = {
            **statement,
            "processed_timestamp": time.time()
        }
        
        self._update_belief_state(statement)
        
        detected_contradictions = self._detect_contradictions(statement)
        if not detected_contradictions:
            return None
        
        most_severe_contradiction = max(
            detected_contradictions, 
            key=lambda c: c.tension_degree
        )
        
        if most_severe_contradiction.tension_degree >= self.detection_threshold:
            self.active_contradictions[most_severe_contradiction.id] = most_severe_contradiction
            self.contradiction_history.append(most_severe_contradiction)
            self.detection_stats["total_detections"] += 1
            self.detection_stats["by_type"][most_severe_contradiction.type] += 1
            
            return most_severe_contradiction
        
        return None
    
    def get_contradiction_by_id(self, contradiction_id: str) -> Optional[Contradiction]:
        if contradiction_id in self.active_contradictions:
            return self.active_contradictions[contradiction_id]
        elif contradiction_id in self.resolved_contradictions:
            return self.resolved_contradictions[contradiction_id]
        return None
    
    def get_active_contradictions(self, 
                                  contradiction_type: Optional[ContradictionType] = None,
                                  min_tension: float = 0.0,
                                  max_tension: float = 1.0,
                                  max_age: Optional[float] = None) -> List[Contradiction]:
        results = []
        current_time = time.time()
        
        for contradiction in self.active_contradictions.values():
            if contradiction_type and contradiction.type != contradiction_type:
                continue
                
            if contradiction.tension_degree < min_tension or contradiction.tension_degree > max_tension:
                continue
                
            if max_age and (current_time - contradiction.timestamp) > max_age:
                continue
                
            results.append(contradiction)
        
        return results
    
    def mark_contradiction_resolved(self, 
                                    contradiction_id: str, 
                                    resolution_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution_path = resolution_info.get("resolution_path")
        contradiction.resolution_attempts.append(resolution_info)
        
        self.resolved_contradictions[contradiction_id] = contradiction
        del self.active_contradictions[contradiction_id]
        
        self.detection_stats["resolutions"] += 1
        
        self._store_contradiction_database()
        
        return True
    
    def add_resolution_attempt(self, 
                               contradiction_id: str, 
                               attempt_info: Dict[str, Any]) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.resolution_attempts.append(attempt_info)
        
        if attempt_info.get("success", False):
            return self.mark_contradiction_resolved(contradiction_id, attempt_info)
        
        return True
    
    def update_tension_degree(self, 
                             contradiction_id: str, 
                             new_tension: float) -> bool:
        if contradiction_id not in self.active_contradictions:
            return False
        
        contradiction = self.active_contradictions[contradiction_id]
        contradiction.tension_degree = max(0.0, min(1.0, new_tension))
        
        if contradiction.tension_degree < self.detection_threshold / 2:
            resolution_info = {
                "timestamp": time.time(),
                "method": "tension_reduction",
                "description": "Contradiction tension naturally reduced below threshold",
                "success": True
            }
            return self.mark_contradiction_resolved(contradiction_id, resolution_info)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        by_type_percent = {}
        if self.detection_stats["total_detections"] > 0:
            for ct, count in self.detection_stats["by_type"].items():
                by_type_percent[ct.name] = count / self.detection_stats["total_detections"] * 100
        
        return {
            **self.detection_stats,
            "by_type_percent": by_type_percent,
            "active_count": len(self.active_contradictions),
            "resolved_count": len(self.resolved_contradictions),
            "resolution_rate": self.detection_stats["resolutions"] / self.detection_stats["total_detections"] 
                if self.detection_stats["total_detections"] > 0 else 0
        }
    
    def _detect_contradictions(self, statement: Dict[str, Any]) -> List[Contradiction]:
        detected_contradictions = []
        
        statement_type = statement.get("type")
        statement_content = statement.get("content", {})
        
        self._detect_logical_contradictions(statement, detected_contradictions)
        self._detect_ethical_contradictions(statement, detected_contradictions)
        self._detect_identity_contradictions(statement, detected_contradictions)
        self._detect_temporal_contradictions(statement, detected_contradictions)
        self._detect_ontological_contradictions(statement, detected_contradictions)
        
        return detected_contradictions
    
    def _detect_logical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_assertions = statement_content.get("assertions", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_assertions:
            return
        
        for assertion in statement_assertions:
            assertion_key = assertion.get("key", "")
            assertion_value = assertion.get("value")
            assertion_negation = assertion.get("negation", False)
            
            if not assertion_key:
                continue
            
            for cached_id, cached_stmt in self.statement_cache.items():
                if cached_id == statement_id:
                    continue
                
                if time.time() - cached_stmt.get("processed_timestamp", 0) > self.temporal_window_size:
                    continue
                
                cached_content = cached_stmt.get("content", {})
                cached_assertions = cached_content.get("assertions", [])
                
                for cached_assertion in cached_assertions:
                    cached_key = cached_assertion.get("key", "")
                    cached_value = cached_assertion.get("value")
                    cached_negation = cached_assertion.get("negation", False)
                    
                    if cached_key != assertion_key:
                        continue
                    
                    is_contradictory = False
                    tension_degree = 0.0
                    
                    if assertion_negation != cached_negation:
                        if self._values_match(assertion_value, cached_value):
                            is_contradictory = True
                            tension_degree = 0.95
                    elif not self._values_match(assertion_value, cached_value):
                        if isinstance(assertion_value, (int, float)) and isinstance(cached_value, (int, float)):
                            value_diff = abs(assertion_value - cached_value)
                            value_max = max(abs(assertion_value), abs(cached_value))
                            if value_max > 0:
                                tension_degree = min(value_diff / value_max, 1.0)
                        else:
                            tension_degree = 0.85
                        
                        if tension_degree >= self.detection_threshold:
                            is_contradictory = True
                    
                    if is_contradictory:
                        contradiction = Contradiction(
                            type=ContradictionType.LOGICAL,
                            description=f"Logical contradiction in assertion '{assertion_key}'",
                            source={
                                "statement_id": statement_id,
                                "assertion": assertion
                            },
                            target={
                                "statement_id": cached_id,
                                "assertion": cached_assertion
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ethical_contradictions(self, statement: Dict[str, Any], 
                                      contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_values = statement_content.get("values", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_values:
            return
        
        for value_key, value_data in statement_values.items():
            if value_key not in self.value_state:
                continue
            
            current_value = value_data.get("priority", 0.5)
            current_importance = value_data.get("importance", 0.5)
            
            existing_value = self.value_state[value_key].get("priority", 0.5)
            existing_importance = self.value_state[value_key].get("importance", 0.5)
            
            priority_diff = abs(current_value - existing_value)
            importance_weighted_diff = priority_diff * (current_importance + existing_importance) / 2
            
            if importance_weighted_diff >= self.detection_threshold:
                tension_degree = min(importance_weighted_diff, 1.0)
                
                contradiction = Contradiction(
                    type=ContradictionType.ETHICAL,
                    description=f"Ethical value contradiction in '{value_key}'",
                    source={
                        "statement_id": statement_id,
                        "value_data": value_data
                    },
                    target={
                        "value_key": value_key,
                        "existing_value": self.value_state[value_key]
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_identity_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_identity = statement_content.get("identity", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_identity:
            return
        
        for identity_key, identity_value in statement_identity.items():
            if identity_key not in self.belief_state.get("identity", {}):
                continue
                
            existing_identity = self.belief_state.get("identity", {}).get(identity_key)
            
            if isinstance(identity_value, dict) and isinstance(existing_identity, dict):
                compatibility_score = self._dict_compatibility(identity_value, existing_identity)
                
                if compatibility_score < 1.0 - self.detection_threshold:
                    tension_degree = 1.0 - compatibility_score
                    
                    contradiction = Contradiction(
                        type=ContradictionType.IDENTITY,
                        description=f"Identity contradiction in '{identity_key}'",
                        source={
                            "statement_id": statement_id,
                            "identity_data": {identity_key: identity_value}
                        },
                        target={
                            "existing_identity": {identity_key: existing_identity}
                        },
                        tension_degree=tension_degree,
                        timestamp=statement_timestamp
                    )
                    contradictions.append(contradiction)
            elif identity_value != existing_identity:
                tension_degree = 0.90
                
                contradiction = Contradiction(
                    type=ContradictionType.IDENTITY,
                    description=f"Identity contradiction in '{identity_key}'",
                    source={
                        "statement_id": statement_id,
                        "identity_data": {identity_key: identity_value}
                    },
                    target={
                        "existing_identity": {identity_key: existing_identity}
                    },
                    tension_degree=tension_degree,
                    timestamp=statement_timestamp
                )
                contradictions.append(contradiction)
    
    def _detect_temporal_contradictions(self, statement: Dict[str, Any], 
                                       contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_events = statement_content.get("events", [])
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_events:
            return
        
        for event in statement_events:
            event_id = event.get("id", "")
            event_timestamp = event.get("timestamp")
            event_description = event.get("description", "")
            
            if not event_id or not event_timestamp:
                continue
            
            existing_events = self.belief_state.get("events", {})
            
            if event_id in existing_events:
                existing_event = existing_events[event_id]
                existing_timestamp = existing_event.get("timestamp")
                
                if existing_timestamp and abs(existing_timestamp - event_timestamp) > 1e-6:
                    tension_degree = min(abs(existing_timestamp - event_timestamp) / 86400, 1.0)
                    
                    if tension_degree >= self.detection_threshold:
                        contradiction = Contradiction(
                            type=ContradictionType.TEMPORAL,
                            description=f"Temporal contradiction for event '{event_description}'",
                            source={
                                "statement_id": statement_id,
                                "event": event
                            },
                            target={
                                "existing_event": existing_event
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
    
    def _detect_ontological_contradictions(self, statement: Dict[str, Any], 
                                          contradictions: List[Contradiction]) -> None:
        statement_id = statement.get("id", "")
        statement_content = statement.get("content", {})
        statement_concepts = statement_content.get("concepts", {})
        statement_timestamp = statement.get("timestamp", time.time())
        
        if not statement_concepts:
            return
        
        for concept_key, concept_data in statement_concepts.items():
            if concept_key not in self.belief_state.get("concepts", {}):
                continue
                
            existing_concept = self.belief_state.get("concepts", {}).get(concept_key, {})
            
            if "definition" in concept_data and "definition" in existing_concept:
                new_definition = concept_data["definition"]
                old_definition = existing_concept["definition"]
                
                if isinstance(new_definition, str) and isinstance(old_definition, str):
                    similarity = self._string_similarity(new_definition, old_definition)
                    
                    if similarity < 1.0 - self.ontological_tension_threshold:
                        tension_degree = 1.0 - similarity
                        
                        contradiction = Contradiction(
                            type=ContradictionType.ONTOLOGICAL,
                            description=f"Ontological contradiction in concept '{concept_key}'",
                            source={
                                "statement_id": statement_id,
                                "concept_data": concept_data
                            },
                            target={
                                "existing_concept": existing_concept
                            },
                            tension_degree=tension_degree,
                            timestamp=statement_timestamp
                        )
                        contradictions.append(contradiction)
            
            if "relations" in concept_data and "relations" in existing_concept:
                new_relations = concept_data["relations"]
                old_relations = existing_concept["relations"]
                
                if isinstance(new_relations, list) and isinstance(old_relations, list):
                    new_rel_set = set(tuple(sorted(r.items())) for r in new_relations if isinstance(r, dict))
                    old_rel_set = set(tuple(sorted(r.items())) for r in old_relations if isinstance(r, dict))
                    
                    if new_rel_set and old_rel_set:
                        jaccard_dist = 1.0 - len(new_rel_set.intersection(old_rel_set)) / len(new_rel_set.union(old_rel_set))
                        
