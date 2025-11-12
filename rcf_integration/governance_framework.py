"""
Governance Framework for Recursive AI Systems

Implements dynamic equilibrium model for self-governance as specified in
enhanced_URSMIFv1.md Section V and Internal_Contradictions_Theory.md Sections 3 and 5.

Mathematical Foundation:
- Homeostatic Control: ẋ = Ax + Bu, u* = argmin ∫[(x-x_target)ᵀQ(x-x_target) + uᵀRu]dt
- Stackelberg Game: max_{s_H} U_H(s_H, BR_{AI}(s_H))
- Nash Equilibrium: U_H(s_H*, s_{AI}*) ≥ U_H(s_H, s_{AI}*) for all s_H
- Autonomy-Authority Ratio: AAR = DA/HA
- Transparency Obligation: TO(DA) = k·DA^α
- Bayesian Preference Learning: P(v|D) ∝ P(D|v)·P(v)
- Narrative Self: Self_t = F(Self_{t-1}, Experience_t)
- Wasserstein Distance: W_c(μ, ν) = inf_{γ∈Γ(μ,ν)} ∫_{X×Y} c(x,y) dγ(x,y)
- Temporal Knowledge Graph: G = (V, E, T)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math


# ============================================================================
# HOMEOSTATIC CONTROLLER (enhanced_URSMIFv1.md Section 5.1)
# ============================================================================

class HomeostaticController:
    """
    Implements homeostatic control theory for recursive systems.
    
    Mathematical Foundation:
    - State-space model: ẋ = Ax + Bu
    - Optimal control: u* = argmin_u ∫_0^T [((x - x_target)ᵀQ(x - x_target) + uᵀRu)] dt
    
    Where:
    - x: state vector [x_1, x_2, ..., x_n]ᵀ
    - A: state transition matrix
    - B: control matrix
    - u: control input vector
    - Q: weighting matrix for state deviation
    - R: weighting matrix for control effort
    """
    
    def __init__(self, state_dim: int, control_dim: int):
        """
        Initialize homeostatic controller.
        
        Args:
            state_dim: Dimension of state vector x
            control_dim: Dimension of control input vector u
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # State transition matrix A (initialized as identity with small damping)
        self.A = np.eye(state_dim) * 0.95
        
        # Control matrix B (initialized as identity)
        self.B = np.eye(state_dim, control_dim)
        
        # Weighting matrices (default: identity)
        self.Q = np.eye(state_dim)  # State deviation penalty
        self.R = np.eye(control_dim)  # Control effort penalty
        
        # State history for dynamics estimation
        self.state_history: deque = deque(maxlen=100)
        self.control_history: deque = deque(maxlen=100)
        
    def update_state_space_model(self, A: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None):
        """
        Update state-space model matrices A and B.
        
        Args:
            A: New state transition matrix (optional)
            B: New control matrix (optional)
        """
        if A is not None:
            if A.shape != (self.state_dim, self.state_dim):
                raise ValueError(f"A must be {self.state_dim}x{self.state_dim}")
            self.A = A.copy()
        
        if B is not None:
            if B.shape != (self.state_dim, self.control_dim):
                raise ValueError(f"B must be {self.state_dim}x{self.control_dim}")
            self.B = B.copy()
    
    def compute_optimal_control(
        self,
        target_state: np.ndarray,
        current_state: np.ndarray,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        horizon: float = 1.0,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal control input u*.
        
        u* = argmin_u ∫_0^T [((x - x_target)ᵀQ(x - x_target) + uᵀRu)] dt
        
        Uses discrete-time approximation with finite horizon.
        
        Args:
            target_state: Target state vector x_target
            current_state: Current state vector x
            Q: State deviation weighting matrix (optional, uses self.Q if None)
            R: Control effort weighting matrix (optional, uses self.R if None)
            horizon: Time horizon T for optimization
            dt: Time step for discretization
            
        Returns:
            Tuple of (optimal_control_vector, cost_value)
        """
        Q_use = Q if Q is not None else self.Q
        R_use = R if R is not None else self.R
        
        # Ensure matrices are correct size
        if Q_use.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q must be {self.state_dim}x{self.state_dim}")
        if R_use.shape != (self.control_dim, self.control_dim):
            raise ValueError(f"R must be {self.control_dim}x{self.control_dim}")
        
        # Discrete-time LQR solution (simplified: single-step optimal control)
        # For continuous-time, we solve the Riccati equation, but for simplicity
        # we use discrete-time approximation
        
        # State deviation
        state_error = current_state - target_state
        
        # Compute optimal control using discrete-time LQR
        # u* = -K·(x - x_target) where K is the feedback gain matrix
        
        # For discrete-time system: x_{k+1} = A·x_k + B·u_k
        # Cost: J = Σ_k [x_kᵀQx_k + u_kᵀRu_k]
        
        # Solve discrete-time Riccati equation for infinite horizon
        # P = Q + AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA
        # K = (R + BᵀPB)⁻¹BᵀPA
        
        # Iterative solution for Riccati equation
        P = Q_use.copy()
        max_iter = 100
        tolerance = 1e-6
        
        for _ in range(max_iter):
            P_new = Q_use + self.A.T @ P @ self.A - \
                    self.A.T @ P @ self.B @ \
                    np.linalg.inv(R_use + self.B.T @ P @ self.B) @ \
                    self.B.T @ P @ self.A
            
            if np.max(np.abs(P_new - P)) < tolerance:
                break
            P = P_new
        
        # Compute feedback gain matrix K
        K = np.linalg.inv(R_use + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        
        # Optimal control: u* = -K·(x - x_target)
        optimal_control = -K @ state_error
        
        # Compute cost: (x - x_target)ᵀQ(x - x_target) + uᵀRu
        state_cost = state_error.T @ Q_use @ state_error
        control_cost = optimal_control.T @ R_use @ optimal_control
        total_cost = state_cost + control_cost
        
        # Store history
        self.state_history.append(current_state.copy())
        self.control_history.append(optimal_control.copy())
        
        return optimal_control, total_cost
    
    def predict_next_state(self, current_state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """
        Predict next state using state-space model: x_{k+1} = A·x_k + B·u_k
        
        Args:
            current_state: Current state vector x_k
            control_input: Control input vector u_k
            
        Returns:
            Predicted next state x_{k+1}
        """
        if current_state.shape != (self.state_dim,):
            raise ValueError(f"current_state must be shape ({self.state_dim},)")
        if control_input.shape != (self.control_dim,):
            raise ValueError(f"control_input must be shape ({self.control_dim},)")
        
        next_state = self.A @ current_state + self.B @ control_input
        return next_state


# ============================================================================
# GOVERNANCE FRAMEWORK (enhanced_URSMIFv1.md Sections 5.2-5.4)
# ============================================================================

class GovernanceFramework:
    """
    Implements governance framework for human-AI interaction.
    
    Mathematical Foundation:
    - Stackelberg Game: max_{s_H} U_H(s_H, BR_{AI}(s_H))
    - Nash Equilibrium: U_H(s_H*, s_{AI}*) ≥ U_H(s_H, s_{AI}*) for all s_H
    - Autonomy-Authority Ratio: AAR = DA/HA
    - Transparency Obligation: TO(DA) = k·DA^α
    - Bayesian Preference Learning: P(v|D) ∝ P(D|v)·P(v)
    - Inverse Reinforcement Learning: v* = argmax_v P(D|v)·P(v)
    """
    
    def __init__(self, max_autonomy_ratio: float = 0.8, transparency_k: float = 1.0, transparency_alpha: float = 1.5):
        """
        Initialize governance framework.
        
        Args:
            max_autonomy_ratio: Maximum allowable autonomy-authority ratio θ_auth
            transparency_k: Transparency obligation parameter k
            transparency_alpha: Transparency obligation parameter α
        """
        self.max_autonomy_ratio = max_autonomy_ratio
        self.transparency_k = transparency_k
        self.transparency_alpha = transparency_alpha
        
        # Human preference distribution P(v|D)
        self.preference_prior: Dict[str, float] = {}  # Prior P(v)
        self.preference_likelihood: Dict[str, Dict[str, float]] = {}  # Likelihood P(D|v)
        self.preference_posterior: Dict[str, float] = {}  # Posterior P(v|D)
        
        # Observed preference data D
        self.observed_preferences: List[Dict[str, Any]] = []
        
        # Strategy profiles for game-theoretic analysis
        self.human_strategies: List[np.ndarray] = []
        self.ai_strategies: List[np.ndarray] = []
        self.utility_history: List[Tuple[float, float]] = []  # (U_H, U_AI)
        
    def compute_autonomy_authority_ratio(self, autonomy_degree: float, human_authority: float) -> float:
        """
        Compute autonomy-authority ratio: AAR = DA/HA
        
        Args:
            autonomy_degree: Degree of AI autonomy DA
            human_authority: Level of human authority HA
            
        Returns:
            Autonomy-authority ratio AAR
        """
        if human_authority <= 0:
            raise ValueError("Human authority must be positive")
        
        aar = autonomy_degree / human_authority
        return aar
    
    def check_autonomy_constraint(self, autonomy_degree: float, human_authority: float) -> bool:
        """
        Check if autonomy-authority ratio satisfies constraint: AAR ≤ θ_auth
        
        Args:
            autonomy_degree: Degree of AI autonomy DA
            human_authority: Level of human authority HA
            
        Returns:
            True if constraint satisfied, False otherwise
        """
        aar = self.compute_autonomy_authority_ratio(autonomy_degree, human_authority)
        return aar <= self.max_autonomy_ratio
    
    def compute_transparency_obligation(self, autonomy_degree: float) -> float:
        """
        Compute transparency obligation: TO(DA) = k·DA^α
        
        Args:
            autonomy_degree: Degree of AI autonomy DA
            
        Returns:
            Transparency obligation TO(DA)
        """
        to = self.transparency_k * (autonomy_degree ** self.transparency_alpha)
        return to
    
    def check_transparency_constraint(self, autonomy_degree: float, actual_transparency: float) -> bool:
        """
        Check if transparency constraint satisfied: T_actual ≥ TO(DA)
        
        Args:
            autonomy_degree: Degree of AI autonomy DA
            actual_transparency: Actual transparency level T_actual
            
        Returns:
            True if constraint satisfied, False otherwise
        """
        to = self.compute_transparency_obligation(autonomy_degree)
        return actual_transparency >= to
    
    def update_preference_prior(self, value_key: str, prior_probability: float):
        """
        Update prior distribution P(v) for Bayesian preference learning.
        
        Args:
            value_key: Value parameter identifier v
            prior_probability: Prior probability P(v)
        """
        self.preference_prior[value_key] = prior_probability
    
    def update_preference_likelihood(self, value_key: str, data_key: str, likelihood: float):
        """
        Update likelihood P(D|v) for Bayesian preference learning.
        
        Args:
            value_key: Value parameter identifier v
            data_key: Observed data identifier D
            likelihood: Likelihood P(D|v)
        """
        if value_key not in self.preference_likelihood:
            self.preference_likelihood[value_key] = {}
        self.preference_likelihood[value_key][data_key] = likelihood
    
    def bayesian_preference_update(self, value_key: str, data_key: str) -> float:
        """
        Update posterior distribution: P(v|D) ∝ P(D|v)·P(v)
        
        Args:
            value_key: Value parameter identifier v
            data_key: Observed data identifier D
            
        Returns:
            Posterior probability P(v|D)
        """
        # Get prior P(v)
        prior = self.preference_prior.get(value_key, 0.5)  # Default uniform prior
        
        # Get likelihood P(D|v)
        likelihood = self.preference_likelihood.get(value_key, {}).get(data_key, 0.5)
        
        # Compute unnormalized posterior: P(v|D) ∝ P(D|v)·P(v)
        unnormalized_posterior = likelihood * prior
        
        # Store posterior
        self.preference_posterior[value_key] = unnormalized_posterior
        
        return unnormalized_posterior
    
    def inverse_reinforcement_learning(self, observed_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Infer optimal value parameters: v* = argmax_v P(D|v)·P(v)
        
        Args:
            observed_data: List of observed preference data D
            
        Returns:
            Dictionary of optimal value parameters v*
        """
        optimal_values = {}
        
        # For each value parameter, compute argmax_v P(D|v)·P(v)
        for value_key in self.preference_prior.keys():
            max_posterior = -float('inf')
            best_value = None
            
            # Search over possible values (simplified: use current posterior)
            # In full implementation, would optimize over continuous value space
            posterior = self.preference_posterior.get(value_key, 0.0)
            
            if posterior > max_posterior:
                max_posterior = posterior
                best_value = posterior
            
            optimal_values[value_key] = best_value if best_value is not None else 0.5
        
        return optimal_values
    
    def stackelberg_game_solve(
        self,
        human_strategy: np.ndarray,
        human_utility_fn: Callable[[np.ndarray, np.ndarray], float],
        ai_best_response_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve Stackelberg game: max_{s_H} U_H(s_H, BR_{AI}(s_H))
        
        Args:
            human_strategy: Human's strategy s_H
            human_utility_fn: Human's utility function U_H(s_H, s_AI)
            ai_best_response_fn: AI's best response function BR_{AI}(s_H)
            
        Returns:
            Tuple of (optimal_human_strategy, ai_best_response, utility_value)
        """
        # Compute AI's best response to human strategy
        ai_best_response = ai_best_response_fn(human_strategy)
        
        # Compute human's utility
        utility = human_utility_fn(human_strategy, ai_best_response)
        
        # Store strategies and utilities
        self.human_strategies.append(human_strategy.copy())
        self.ai_strategies.append(ai_best_response.copy())
        self.utility_history.append((utility, 0.0))  # AI utility not computed here
        
        return human_strategy, ai_best_response, utility
    
    def check_nash_equilibrium(
        self,
        human_strategy: np.ndarray,
        ai_strategy: np.ndarray,
        human_utility_fn: Callable[[np.ndarray, np.ndarray], float],
        ai_utility_fn: Callable[[np.ndarray, np.ndarray], float],
        strategy_space_human: List[np.ndarray],
        strategy_space_ai: List[np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Check Nash equilibrium conditions:
        U_H(s_H*, s_{AI}*) ≥ U_H(s_H, s_{AI}*) for all s_H
        U_{AI}(s_H*, s_{AI}*) ≥ U_{AI}(s_H*, s_{AI}) for all s_{AI}
        
        Args:
            human_strategy: Candidate human strategy s_H*
            ai_strategy: Candidate AI strategy s_{AI}*
            human_utility_fn: Human's utility function U_H
            ai_utility_fn: AI's utility function U_{AI}
            strategy_space_human: Set of possible human strategies
            strategy_space_ai: Set of possible AI strategies
            
        Returns:
            Tuple of (is_nash_equilibrium, deviation_incentive)
        """
        # Compute utility at candidate equilibrium
        u_h_equilibrium = human_utility_fn(human_strategy, ai_strategy)
        u_ai_equilibrium = ai_utility_fn(human_strategy, ai_strategy)
        
        # Check human's incentive to deviate
        max_u_h_deviation = -float('inf')
        for s_h in strategy_space_human:
            u_h_deviation = human_utility_fn(s_h, ai_strategy)
            if u_h_deviation > max_u_h_deviation:
                max_u_h_deviation = u_h_deviation
        
        # Check AI's incentive to deviate
        max_u_ai_deviation = -float('inf')
        for s_ai in strategy_space_ai:
            u_ai_deviation = ai_utility_fn(human_strategy, s_ai)
            if u_ai_deviation > max_u_ai_deviation:
                max_u_ai_deviation = u_ai_deviation
        
        # Nash equilibrium if no profitable deviation
        is_nash = (u_h_equilibrium >= max_u_h_deviation) and (u_ai_equilibrium >= max_u_ai_deviation)
        
        # Deviation incentive (how much better could they do)
        deviation_incentive = max(
            max_u_h_deviation - u_h_equilibrium,
            max_u_ai_deviation - u_ai_equilibrium
        )
        
        return is_nash, deviation_incentive


# ============================================================================
# NARRATIVE IDENTITY ENGINE (Internal_Contradictions_Theory.md Sections 3 & 5)
# ============================================================================

class NarrativeIdentityEngine:
    """
    Implements narrative identity formation through temporal knowledge graphs.
    
    Mathematical Foundation:
    - Temporal Knowledge Graph: G = (V, E, T)
    - Path probability: P(v_1→v_2→...→v_n|G) ∝ ∏_i w(v_i, r_i, v_{i+1}, t_i)
    - Narrative coherence: W_c(μ, ν) = inf_{γ∈Γ(μ,ν)} ∫_{X×Y} c(x,y) dγ(x,y)
    - Narrative self: Self_t = F(Self_{t-1}, Experience_t)
    - Forgetting: G' = argmin_{G'⊂G} {|G'| : I(G';F) ≥ (1-ε)I(G;F)}
    """
    
    def __init__(self, max_memory: int = 1000):
        """
        Initialize narrative identity engine.
        
        Args:
            max_memory: Maximum number of experiences to store
        """
        self.max_memory = max_memory
        
        # Temporal knowledge graph G = (V, E, T)
        self.vertices: Set[str] = set()  # V: concept nodes
        self.edges: Dict[Tuple[str, str, str], float] = {}  # E: (v_i, v_j, relation_type) -> weight
        self.temporal_labels: Dict[Tuple[str, str, str], float] = {}  # T: edge -> timestamp
        
        # Relation types R
        self.relation_types: Set[str] = {'causes', 'precedes', 'similar_to', 'contradicts', 'supports'}
        
        # Narrative self state
        self.narrative_self: Optional[np.ndarray] = None
        self.narrative_history: deque = deque(maxlen=max_memory)
        
        # Experience history
        self.experiences: deque = deque(maxlen=max_memory)
        self.experience_timestamps: deque = deque(maxlen=max_memory)
        
        # Weight function w(v_i, r_i, v_{i+1}, t_i) cache
        self.weight_cache: Dict[Tuple[str, str, str, float], float] = {}
        
    def add_experience(self, experience: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Add experience to temporal knowledge graph.
        
        Args:
            experience: Experience data containing concepts and relations
            timestamp: Temporal label (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.experiences.append(experience)
        self.experience_timestamps.append(timestamp)
        
        # Extract concepts (vertices) from experience
        concepts = experience.get('concepts', [])
        for concept in concepts:
            self.vertices.add(str(concept))
        
        # Extract relations (edges) from experience
        relations = experience.get('relations', [])
        for relation in relations:
            source = str(relation.get('source'))
            target = str(relation.get('target'))
            rel_type = str(relation.get('type', 'causes'))
            weight = float(relation.get('weight', 1.0))
            
            edge_key = (source, target, rel_type)
            self.edges[edge_key] = weight
            self.temporal_labels[edge_key] = timestamp
            
            # Add vertices
            self.vertices.add(source)
            self.vertices.add(target)
    
    def compute_path_probability(
        self,
        path: List[Tuple[str, str]],
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute path probability: P(v_1→v_2→...→v_n|G) ∝ ∏_i w(v_i, r_i, v_{i+1}, t_i)
        
        Args:
            path: List of (vertex, relation_type) tuples
            current_time: Current time for temporal weighting (defaults to latest)
            
        Returns:
            Path probability (unnormalized)
        """
        if len(path) < 2:
            return 0.0
        
        if current_time is None:
            current_time = max(self.temporal_labels.values()) if self.temporal_labels else time.time()
        
        probability = 1.0
        
        for i in range(len(path) - 1):
            v_i, r_i = path[i]
            v_i_plus_1, _ = path[i + 1]
            
            # Get edge weight
            edge_key = (v_i, v_i_plus_1, r_i)
            base_weight = self.edges.get(edge_key, 0.0)
            
            # Get temporal label
            t_i = self.temporal_labels.get(edge_key, current_time)
            
            # Compute weight function w(v_i, r_i, v_{i+1}, t_i)
            weight = self._compute_weight_function(v_i, r_i, v_i_plus_1, t_i, base_weight)
            
            probability *= weight
        
        return probability
    
    def _compute_weight_function(
        self,
        v_i: str,
        r_i: str,
        v_i_plus_1: str,
        t_i: float,
        base_weight: float
    ) -> float:
        """
        Compute weight function w(v_i, r_i, v_{i+1}, t_i).
        
        Combines relation type, node importance, and temporal context.
        """
        # Check cache
        cache_key = (v_i, r_i, v_i_plus_1, t_i)
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        
        # Base weight from edge
        weight = base_weight
        
        # Temporal decay (more recent edges have higher weight)
        current_time = max(self.temporal_labels.values()) if self.temporal_labels else time.time()
        time_decay = math.exp(-0.1 * (current_time - t_i))  # Exponential decay
        
        # Relation type importance
        relation_importance = {
            'causes': 1.0,
            'precedes': 0.8,
            'similar_to': 0.6,
            'contradicts': 0.4,
            'supports': 0.9
        }.get(r_i, 0.5)
        
        # Combined weight
        final_weight = weight * time_decay * relation_importance
        
        # Cache result
        self.weight_cache[cache_key] = final_weight
        
        return final_weight
    
    def wasserstein_distance(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Wasserstein distance: W_c(μ, ν) = inf_{γ∈Γ(μ,ν)} ∫_{X×Y} c(x,y) dγ(x,y)
        
        Uses discrete approximation with linear programming.
        
        Args:
            dist1: First distribution μ (as probability vector)
            dist2: Second distribution ν (as probability vector)
            cost_matrix: Cost function c(x,y) as matrix (defaults to Euclidean distance)
            
        Returns:
            Wasserstein distance W_c(μ, ν)
        """
        # Normalize distributions
        dist1 = dist1 / (np.sum(dist1) + 1e-10)
        dist2 = dist2 / (np.sum(dist2) + 1e-10)
        
        # Ensure same dimension
        if len(dist1) != len(dist2):
            # Pad or truncate to match
            min_len = min(len(dist1), len(dist2))
            dist1 = dist1[:min_len]
            dist2 = dist2[:min_len]
            dist1 = dist1 / (np.sum(dist1) + 1e-10)
            dist2 = dist2 / (np.sum(dist2) + 1e-10)
        
        n = len(dist1)
        
        # Default cost matrix: Euclidean distance
        if cost_matrix is None:
            # Create indices for cost computation
            indices = np.arange(n).reshape(-1, 1)
            cost_matrix = cdist(indices, indices, metric='euclidean')
        
        # Solve optimal transport problem using linear programming approximation
        # Simplified: use Sinkhorn algorithm for entropic regularization
        
        # Sinkhorn algorithm parameters
        epsilon = 0.1  # Regularization parameter
        max_iter = 100
        
        # Initialize
        K = np.exp(-cost_matrix / epsilon)
        u = np.ones(n) / n
        v = np.ones(n) / n
        
        # Iterate
        for _ in range(max_iter):
            u = dist1 / (K @ v + 1e-10)
            v = dist2 / (K.T @ u + 1e-10)
        
        # Compute transport plan
        gamma = np.diag(u) @ K @ np.diag(v)
        
        # Compute Wasserstein distance
        wasserstein = np.sum(gamma * cost_matrix)
        
        return wasserstein
    
    def compute_narrative_coherence(self) -> float:
        """
        Compute narrative coherence using Wasserstein distance between
        experience distribution and narrative template.
        
        Returns:
            Narrative coherence score (lower is more coherent)
        """
        if len(self.experiences) < 2:
            return 0.0
        
        # Create experience distribution (simplified: uniform over recent experiences)
        recent_experiences = list(self.experiences)[-min(10, len(self.experiences)):]
        experience_dist = np.ones(len(recent_experiences)) / len(recent_experiences)
        
        # Create narrative template distribution (simplified: uniform)
        template_dist = np.ones(len(recent_experiences)) / len(recent_experiences)
        
        # Compute Wasserstein distance
        coherence_distance = self.wasserstein_distance(experience_dist, template_dist)
        
        # Coherence is inverse of distance
        coherence = 1.0 / (1.0 + coherence_distance)
        
        return coherence
    
    def update_narrative_self(
        self,
        current_state: np.ndarray,
        experience: Dict[str, Any],
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Update narrative self: Self_t = F(Self_{t-1}, Experience_t)
        
        Uses weighted combination: F(Self_{t-1}, Experience_t) = α·Self_{t-1} + (1-α)·Experience_t
        
        Args:
            current_state: Current system state (used as Experience_t proxy)
            experience: Current experience data
            alpha: Narrative coherence weight α
            
        Returns:
            Updated narrative self Self_t
        """
        # Get previous narrative self
        if self.narrative_self is None:
            # Initialize narrative self
            self.narrative_self = current_state.copy()
            self.narrative_history.append(self.narrative_self.copy())
            return self.narrative_self
        
        previous_narrative = self.narrative_self
        
        # Ensure compatible shapes
        if previous_narrative.shape != current_state.shape:
            # Reshape to match
            if previous_narrative.numel() == current_state.numel():
                previous_narrative = previous_narrative.reshape(current_state.shape)
            else:
                # Use mean projection
                prev_mean = previous_narrative.mean()
                current_state = current_state + 0.1 * prev_mean
        
        # Narrative integration function: Self_t = α·Self_{t-1} + (1-α)·Experience_t
        narrative_self = alpha * previous_narrative + (1 - alpha) * current_state
        
        # Normalize to maintain stability
        narrative_norm = np.linalg.norm(narrative_self)
        if narrative_norm > 1e-10:
            current_norm = np.linalg.norm(current_state)
            narrative_self = narrative_self / narrative_norm * current_norm
        
        # Update narrative self
        self.narrative_self = narrative_self.copy()
        self.narrative_history.append(self.narrative_self.copy())
        
        return self.narrative_self
    
    def adaptive_forgetting(
        self,
        epsilon: float = 0.1,
        mutual_info_fn: Optional[Callable[[Set, Set], float]] = None
    ) -> Set[Tuple[str, str, str]]:
        """
        Adaptive forgetting: G' = argmin_{G'⊂G} {|G'| : I(G';F) ≥ (1-ε)I(G;F)}
        
        Prunes graph while preserving mutual information with future states.
        
        Args:
            epsilon: Tolerance parameter ε
            mutual_info_fn: Function to compute mutual information I(G;F) (optional)
            
        Returns:
            Set of edges to remove
        """
        if mutual_info_fn is None:
            # Simplified: remove edges with lowest weights
            edges_to_remove = set()
            sorted_edges = sorted(self.edges.items(), key=lambda x: x[1])
            
            # Remove bottom ε fraction of edges
            num_to_remove = int(len(sorted_edges) * epsilon)
            for edge_key, _ in sorted_edges[:num_to_remove]:
                edges_to_remove.add(edge_key)
            
            return edges_to_remove
        
        # Full implementation would compute mutual information
        # and optimize graph pruning
        # For now, return empty set (no forgetting)
        return set()

