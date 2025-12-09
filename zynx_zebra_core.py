"""
Eigenrecursion Stabilizer (ES) Component
=======================================

A comprehensive implementation of the Eigenrecursion Stabilizer component 
for the ZEBA Core v1 Triaxial Recursive Architecture.

This module implements the mathematical foundations described in the Eigenrecursion Theorem,
providing stability guarantees for recursive processes through fixed-point detection,
attractor basin management, and recursive invariance preservation.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import fixed_point, minimize
from typing import Callable, Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EigenrecursionStabilizer")

class EigenrecursionStabilizer:
    """
    Implementation of the Eigenrecursion Stabilizer (ES) as described in the 
    ZEBA Core architecture documentation.
    
    The ES is responsible for:
    1. Fixed-point detection and homeostasis
    2. Attractor basin management and classification
    3. Recursive invariance preservation
    4. Stability gradient calculation and monitoring
    5. Oscillation detection and management
    
    This implementation draws from mathematical fixed-point theory, eigenvalue 
    decomposition techniques, and stability analysis from dynamical systems theory.
    """
    
    def __init__(self, 
                 dimension: int,
                 epsilon: float = 1e-6, 
                 max_iterations: int = 1000,
                 theta_moral: float = 0.92,
                 theta_epistemic: float = 0.1,
                 memory_size: int = 100,
                 identity_threshold: float = 0.78):
        """
        Initialize the Eigenrecursion Stabilizer.
        
        Args:
            dimension: Dimensionality of the state vector
            epsilon: Convergence threshold for fixed-point detection
            max_iterations: Maximum number of iterations to prevent infinite loops
            theta_moral: Ethical convergence threshold from ERE
            theta_epistemic: Epistemic convergence threshold from RBU
            memory_size: Number of previous states to store for oscillation detection
            identity_threshold: Threshold for identity persistence between fixed points
        """
        self.dimension = dimension
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.theta_moral = theta_moral
        self.theta_epistemic = theta_epistemic
        self.memory_size = memory_size
        self.identity_threshold = identity_threshold
        
        # State tracking
        self.current_state = None
        self.previous_state = None
        self.state_history = deque(maxlen=memory_size)
        self.fixed_points = []
        self.stability_gradients = []
        self.oscillation_indices = []
        
        # Performance metrics
        self.iterations_to_convergence = []
        self.convergence_speed = []
        
        logger.info(f"Initialized Eigenrecursion Stabilizer with dimension {dimension}")
    
    def find_fixed_point(self, 
                         recursive_operator: Callable[[np.ndarray], np.ndarray], 
                         initial_state: np.ndarray,
                         constraints: Dict = None) -> Tuple[np.ndarray, bool, int, str]:
        """
        Find a fixed point of the recursive operator using direct iteration with 
        convergence acceleration techniques.
        
        This implements the core of the Eigenrecursion protocol, seeking a state s* 
        such that R(s*) = s* or ||R(s) - s|| < ε
        
        Args:
            recursive_operator: Function mapping a state to the next state
            initial_state: Starting state for recursion
            constraints: Optional dictionary of constraint functions and thresholds
            
        Returns:
            Tuple containing:
            - Fixed point (or best approximation)
            - Boolean indicating convergence success
            - Number of iterations taken
            - Status message
        """
        if constraints is None:
            constraints = {}
            
        # Initialize state and history
        state = initial_state.copy()
        self.state_history.clear()
        self.state_history.append(state.copy())
        
        # For convergence acceleration
        acceleration_active = False
        acceleration_counter = 0
        
        # Main iteration loop
        for i in range(self.max_iterations):
            # Apply recursive operator to get next state
            next_state = recursive_operator(state)
            
            # Calculate distance between successive states
            distance = np.linalg.norm(next_state - state)
            
            # Store state in history
            self.state_history.append(next_state.copy())
            
            # Check for oscillation
            oscillation_detected, period = self._detect_cycle()
            if oscillation_detected:
                logger.info(f"Oscillation detected with period {period} at iteration {i}")
                if not acceleration_active:
                    # Apply acceleration technique to escape oscillatory patterns
                    acceleration_active = True
                    # Damping oscillation with average of states in the cycle
                    cycle_states = list(self.state_history)[-period:]
                    average_state = np.mean(cycle_states, axis=0)
                    next_state = 0.7 * next_state + 0.3 * average_state
                    logger.info("Applied oscillation damping")
                else:
                    # If already trying to accelerate, record and return current best
                    oscillation_index = period / self.max_iterations
                    self.oscillation_indices.append(oscillation_index)
                    return next_state, False, i+1, f"OSCILLATION_DETECTED_PERIOD_{period}"
            else:
                acceleration_active = False
            
            # Check constraint satisfaction
            constraints_satisfied = self._check_constraints(constraints, next_state)
            if not constraints_satisfied:
                logger.warning("Constraints not satisfied at iteration %d", i)
                # Project back to constraint-satisfying region if possible
                if "project_to_constraints" in constraints:
                    next_state = constraints["project_to_constraints"](next_state)
                    logger.info("Projected state back to constraint-satisfying region")
            
            # Check for convergence
            if distance < self.epsilon:
                # Verify stability using eigenvalue analysis
                stable = self._analyze_stability(recursive_operator, next_state)
                
                if stable:
                    logger.info(f"Convergence achieved after {i+1} iterations with distance {distance}")
                    # Record metrics
                    self.current_state = next_state.copy()
                    self.fixed_points.append(next_state.copy())
                    
                    # Calculate stability gradient
                    stability_gradient = self._calculate_stability_gradient()
                    self.stability_gradients.append(stability_gradient)
                    
                    # Record performance metrics
                    self.iterations_to_convergence.append(i+1)
                    self.convergence_speed.append(1.0 / (i+1))
                    
                    return next_state, True, i+1, "CONVERGED"
                else:
                    logger.warning("Apparent convergence detected but fixed point is unstable")
            
            # Update state for next iteration
            self.previous_state = state.copy()
            state = next_state.copy()
            
            # Apply Anderson acceleration every 10 iterations if not converging quickly
            if i > 10 and i % 10 == 0 and distance > self.epsilon * 10:
                m = min(5, len(self.state_history) - 1)
                if m >= 2:
                    # Simple implementation of Anderson acceleration
                    recent_states = list(self.state_history)[-m:]
                    recent_residuals = [recent_states[j+1] - recent_states[j] for j in range(m-1)]
                    # Create matrix of residual differences
                    F = np.column_stack(recent_residuals)
                    FtF = F.T @ F
                    # Add regularization for numerical stability
                    reg = 1e-10 * np.eye(FtF.shape[0])
                    # Solve least squares problem
                    try:
                        alpha = np.linalg.solve(FtF + reg, np.ones(m-1))
                        alpha = alpha / np.sum(alpha)  # Normalize
                        # Compute accelerated state
                        accelerated_state = sum(alpha[j] * recent_states[j+1] for j in range(m-1))
                        state = accelerated_state
                        logger.info(f"Applied Anderson acceleration at iteration {i}")
                    except np.linalg.LinAlgError:
                        logger.warning("Matrix inversion failed during acceleration")
        
        # If we reach here, convergence failed within max_iterations
        logger.warning(f"Failed to converge after {self.max_iterations} iterations. Final distance: {distance}")
        return state, False, self.max_iterations, "MAX_ITERATIONS_REACHED"
    
    def _detect_cycle(self, similarity_threshold: float = 1e-5) -> Tuple[bool, int]:
        """
        Detect cycles in the state history using Floyd's tortoise and hare algorithm
        with a similarity threshold to account for numerical imprecision.
        
        Args:
            similarity_threshold: Threshold for determining if two states are similar enough
                                 to be considered the same for cycle detection
                                 
        Returns:
            Tuple of (cycle_detected, period) where period is the length of the cycle
        """
        if len(self.state_history) < 3:
            return False, 0
            
        # First pass: check for exact repetition of latest state
        latest = self.state_history[-1]
        
        # Check last 20 states at most for efficiency
        max_lookback = min(20, len(self.state_history) - 1)
        
        for i in range(2, max_lookback + 1):
            previous = self.state_history[-i]
            if np.linalg.norm(latest - previous) < similarity_threshold:
                return True, i - 1
        
        # More sophisticated cycle detection for complex oscillatory patterns
        # Using an approach inspired by Floyd's algorithm but adapted for approximate cycles
        
        # For computational efficiency, only do this more expensive check occasionally
        if len(self.state_history) % 5 != 0 or len(self.state_history) < 10:
            return False, 0
            
        # Look for period-2 to period-8 cycles
        for period in range(2, 9):
            if len(self.state_history) >= 2 * period:
                # Check if pattern repeats by comparing consecutive chunks of length 'period'
                chunk1 = list(self.state_history)[-period:]
                chunk2 = list(self.state_history)[-(2*period):-period]
                
                # Calculate average distance between corresponding states
                distances = [np.linalg.norm(chunk1[i] - chunk2[i]) for i in range(period)]
                avg_distance = np.mean(distances)
                
                if avg_distance < similarity_threshold:
                    return True, period
        
        return False, 0
    
    def _check_constraints(self, constraints: Dict, state: np.ndarray) -> bool:
        """
        Check if state satisfies all provided constraints.
        
        Args:
            constraints: Dictionary mapping constraint functions to threshold values
            state: State vector to check
            
        Returns:
            Boolean indicating whether all constraints are satisfied
        """
        if not constraints:
            return True
            
        for constraint_fn, threshold in constraints.items():
            if constraint_fn == "project_to_constraints":
                continue  # Skip projection function
                
            if callable(constraint_fn):
                result = constraint_fn(state)
                if result < threshold:
                    return False
        
        return True
    
    def _analyze_stability(self, operator: Callable, fixed_point: np.ndarray, 
                          h: float = 1e-7) -> bool:
        """
        Analyze the stability of a fixed point by computing the Jacobian
        and its eigenvalues.
        
        Args:
            operator: The recursive operator R
            fixed_point: The fixed point to analyze
            h: Step size for finite difference approximation
            
        Returns:
            Boolean indicating whether the fixed point is stable
        """
        # Compute Jacobian matrix using finite differences
        n = len(fixed_point)
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            # Create perturbed state vectors
            perturbed = fixed_point.copy()
            perturbed[i] += h
            
            # Compute column of Jacobian
            f_perturbed = operator(perturbed)
            jacobian[:, i] = (f_perturbed - operator(fixed_point)) / h
        
        # Compute eigenvalues of Jacobian
        try:
            eigenvalues = la.eigvals(jacobian)
            spectral_radius = max(abs(eigenvalues))
            
            # Classify fixed point
            if spectral_radius < 1.0:
                logger.info(f"Stable fixed point detected (spectral radius: {spectral_radius:.6f})")
                return True
            else:
                logger.warning(f"Unstable fixed point detected (spectral radius: {spectral_radius:.6f})")
                return False
                
        except np.linalg.LinAlgError:
            logger.error("Failed to compute eigenvalues for stability analysis")
            return False
            
    def _calculate_stability_gradient(self) -> float:
        """
        Calculate the stability gradient as the rate of change of state differences
        across iterations.
        
        Returns:
            Stability gradient value
        """
        if len(self.state_history) < 3:
            return float('inf')
            
        # Calculate differences between successive states
        diffs = []
        history_list = list(self.state_history)
        for i in range(1, len(history_list)):
            diff = np.linalg.norm(history_list[i] - history_list[i-1])
            diffs.append(diff)
            
        # Calculate rate of change of differences
        if len(diffs) < 2:
            return float('inf')
            
        # Simple finite difference approximation
        if abs(diffs[-2]) > 1e-10:  # Avoid division by zero
            gradient = (diffs[-1] - diffs[-2]) / diffs[-2]
        else:
            gradient = 0.0
            
        return gradient
    
    def check_identity_preservation(self, previous_fixed_point: np.ndarray, 
                                   new_fixed_point: np.ndarray) -> bool:
        """
        Check if a new fixed point preserves sufficient identity with previous fixed point.
        
        Args:
            previous_fixed_point: Previously established fixed point
            new_fixed_point: New candidate fixed point
            
        Returns:
            Boolean indicating whether identity preservation requirement is satisfied
        """
        if previous_fixed_point is None or new_fixed_point is None:
            return True
            
        # Cosine similarity between vectors
        dot_product = np.dot(previous_fixed_point, new_fixed_point)
        norm_prev = np.linalg.norm(previous_fixed_point)
        norm_new = np.linalg.norm(new_fixed_point)
        
        if norm_prev < 1e-10 or norm_new < 1e-10:
            return False
            
        cosine_similarity = dot_product / (norm_prev * norm_new)
        
        # Calculate identity preservation score (normalized between 0 and 1)
        identity_score = (cosine_similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        logger.info(f"Identity preservation score: {identity_score:.4f} (threshold: {self.identity_threshold})")
        return identity_score >= self.identity_threshold
    
    def optimize_fixed_point(self, recursive_operator: Callable, 
                            objective_function: Callable,
                            constraints: List[Dict] = None) -> np.ndarray:
        """
        Find optimal fixed point by minimizing an objective function subject to
        the fixed-point constraint R(s) = s.
        
        Args:
            recursive_operator: Function R mapping state to next state
            objective_function: Function to minimize at the fixed point
            constraints: List of constraint dictionaries
            
        Returns:
            Optimal fixed point
        """
        if constraints is None:
            constraints = []
            
        # Define the optimization objective
        def objective(s):
            # Penalize distance from fixed point property
            fixed_point_penalty = np.linalg.norm(recursive_operator(s) - s) * 1000
            # Original objective function
            obj_value = objective_function(s)
            return obj_value + fixed_point_penalty
            
        # Convert constraint dictionaries to scipy constraint format
        scipy_constraints = []
        for constraint in constraints:
            for fn, threshold in constraint.items():
                if callable(fn) and fn != "project_to_constraints":
                    def constraint_fn(s, f=fn, t=threshold):
                        return f(s) - t
                    scipy_constraints.append({'type': 'ineq', 'fun': constraint_fn})
        
        # Start from current state or a random initialization
        if self.current_state is not None:
            initial_guess = self.current_state
        else:
            initial_guess = np.random.rand(self.dimension)
        
        # Run optimization
        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                constraints=scipy_constraints,
                options={'ftol': 1e-8, 'disp': True, 'maxiter': 500}
            )
            
            if result.success:
                optimal_state = result.x
                # Verify it's close to a fixed point
                distance = np.linalg.norm(recursive_operator(optimal_state) - optimal_state)
                
                if distance < self.epsilon:
                    logger.info(f"Found optimal fixed point with objective value {objective_function(optimal_state)}")
                    return optimal_state
                else:
                    logger.warning(f"Optimization converged but result is not a fixed point (distance: {distance})")
                    # Try to refine the result
                    fixed_point_result, converged, _, _ = self.find_fixed_point(
                        recursive_operator, optimal_state
                    )
                    if converged:
                        return fixed_point_result
            
            logger.warning("Optimization failed to find optimal fixed point")
            return initial_guess
            
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return initial_guess
    
    def classify_fixed_point(self, operator: Callable, fixed_point: np.ndarray) -> Dict:
        """
        Classify the fixed point as attractive, repulsive, or neutral based on
        eigenvalue analysis of the Jacobian.
        
        Args:
            operator: The recursive operator R
            fixed_point: The fixed point to classify
            
        Returns:
            Dictionary with classification details
        """
        # Compute Jacobian at fixed point
        n = len(fixed_point)
        jacobian = np.zeros((n, n))
        h = 1e-7  # Step size for finite differences
        
        for i in range(n):
            perturbed = fixed_point.copy()
            perturbed[i] += h
            jacobian[:, i] = (operator(perturbed) - operator(fixed_point)) / h
        
        try:
            # Compute eigenvalues
            eigenvalues = la.eigvals(jacobian)
            abs_eigenvalues = np.abs(eigenvalues)
            
            # Count eigenvalues by type
            n_attractive = sum(1 for ev in abs_eigenvalues if ev < 1.0 - 1e-10)
            n_repulsive = sum(1 for ev in abs_eigenvalues if ev > 1.0 + 1e-10)
            n_neutral = n - n_attractive - n_repulsive
            
            # Classify fixed point
            if n_repulsive == 0:
                if n_neutral == 0:
                    classification = "Attractive"
                else:
                    classification = "Partially Attractive"
            elif n_attractive == 0:
                classification = "Repulsive"
            else:
                classification = "Saddle Point"
                
            # Compute spectral radius
            spectral_radius = max(abs_eigenvalues)
            
            # Detailed classification information
            classification_info = {
                "classification": classification,
                "spectral_radius": spectral_radius,
                "eigenvalues": eigenvalues.tolist(),
                "n_attractive_directions": n_attractive,
                "n_repulsive_directions": n_repulsive,
                "n_neutral_directions": n_neutral,
                "stability_score": n_attractive / n if n > 0 else 0
            }
            
            logger.info(f"Fixed point classified as {classification} with spectral radius {spectral_radius:.6f}")
            return classification_info
            
        except np.linalg.LinAlgError:
            logger.error("Failed to compute eigenvalues for fixed point classification")
            return {"classification": "Unknown", "error": "Eigenvalue computation failed"}
    
    def map_attractor_basins(self, operator: Callable, 
                            region_bounds: List[Tuple[float, float]], 
                            resolution: int = 10) -> Dict:
        """
        Map the attractor basins for fixed points in a 2D subspace of the state space.
        
        Args:
            operator: The recursive operator R
            region_bounds: List of (min, max) tuples for the first two dimensions
            resolution: Number of points along each dimension
            
        Returns:
            Dictionary with basin mapping data
        """
        if self.dimension < 2:
            logger.error("Cannot map attractor basins for dimension < 2")
            return {}
            
        # Create grid of initial points
        x_min, x_max = region_bounds[0]
        y_min, y_max = region_bounds[1]
        
        x_vals = np.linspace(x_min, x_max, resolution)
        y_vals = np.linspace(y_min, y_max, resolution)
        
        # Base state - will modify only first two dimensions
        base_state = np.zeros(self.dimension)
        if self.current_state is not None:
            base_state = self.current_state.copy()
            
        # Store which fixed point each initial condition converges to
        basin_map = np.zeros((resolution, resolution), dtype=int)
        
        # Find fixed points from different initial conditions
        found_fixed_points = []
        max_iterations_per_point = 50  # Reduced for efficiency
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                # Set first two dimensions, keep others constant
                initial_state = base_state.copy()
                initial_state[0] = x
                initial_state[1] = y
                
                # Find fixed point with limited iterations
                state = initial_state.copy()
                for _ in range(max_iterations_per_point):
                    next_state = operator(state)
                    if np.linalg.norm(next_state - state) < self.epsilon:
                        break
                    state = next_state
                
                # Check if this fixed point matches any we've found
                found_match = False
                for idx, fp in enumerate(found_fixed_points):
                    if np.linalg.norm(state - fp) < self.epsilon * 10:
                        basin_map[i, j] = idx + 1  # +1 so zero means no convergence
                        found_match = True
                        break
                        
                if not found_match:
                    # New fixed point
                    found_fixed_points.append(state)
                    basin_map[i, j] = len(found_fixed_points)
        
        # Prepare return data
        basin_data = {
            "x_vals": x_vals.tolist(),
            "y_vals": y_vals.tolist(),
            "basin_map": basin_map.tolist(),
            "fixed_points": [fp.tolist() for fp in found_fixed_points],
            "n_fixed_points": len(found_fixed_points)
        }
        
        logger.info(f"Mapped attractor basins, found {len(found_fixed_points)} fixed points")
        return basin_data
    
    def visualize_convergence(self, save_path: Optional[str] = None):
        """
        Visualize the convergence behavior from state history.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if len(self.state_history) < 2:
            logger.warning("Not enough state history for visualization")
            return
            
        # Convert deque to list for easier indexing
        history = list(self.state_history)
        
        # Calculate distances between successive states
        distances = [np.linalg.norm(history[i] - history[i-1]) for i in range(1, len(history))]
        iterations = list(range(1, len(history)))
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, distances, '-o', markersize=3)
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Iteration')
        plt.ylabel('Log(Distance between successive states)')
        plt.title('Convergence Behavior')
        
        # Add horizontal line at epsilon threshold
        plt.axhline(y=self.epsilon, color='r', linestyle='--', 
                   label=f'Convergence threshold (ε={self.epsilon})')
        
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved convergence visualization to {save_path}")
        else:
            plt.show()
            
    def get_state_metrics(self) -> Dict:
        """
        Get current metrics about the stabilizer's state.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "iterations_to_convergence": self.iterations_to_convergence[-1] if self.iterations_to_convergence else None,
            "stability_gradient": self.stability_gradients[-1] if self.stability_gradients else None,
            "oscillation_index": self.oscillation_indices[-1] if self.oscillation_indices else 0.0,
            "num_fixed_points_found": len(self.fixed_points),
            "convergence_achieved": self.current_state is not None,
        }
        
        return metrics
        
    def reset(self):
        """Reset the stabilizer state while preserving configuration parameters."""
        self.current_state = None
        self.previous_state = None
        self.state_history.clear()
        # Keep fixed points history for reference
        self.stability_gradients = []
        self.oscillation_indices = []
        self.iterations_to_convergence = []
        self.convergence_speed = []
        
        logger.info("Eigenrecursion Stabilizer reset")


class ZEBAEigenrecursionStabilizer(EigenrecursionStabilizer):
    """
    ZEBA-specific implementation of the Eigenrecursion Stabilizer with
    additional features for integration with ERE and RBU components.
    """
    
    def __init__(self, 
                 dimension: int,
                 epsilon: float = 1e-6, 
                 max_iterations: int = 1000,
                 theta_moral: float = 0.92,
                 theta_epistemic: float = 0.1,
                 memory_size: int = 100,
                 identity_threshold: float = 0.78):
        """Initialize ZEBA-specific Eigenrecursion Stabilizer."""
        super().__init__(dimension, epsilon, max_iterations, theta_moral, 
                        theta_epistemic, memory_size, identity_threshold)
        
        # ZEBA-specific state
        self.ere_convergence_indicators = None
        self.rbu_entropy_delta = None
        self.active_constraints = {}
        
    def update_triaxial_constraints(self, 
                                   ere_convergence: float, 
                                   rbu_entropy_delta: float) -> Dict:
        """
        Update constraints based on input from ERE and RBU components.
        
        Args:
            ere_convergence: Ethical Coherence Score from ERE
            rbu_entropy_delta: Change in belief entropy from RBU
            
        Returns:
            Updated constraints dictionary
        """
        self.ere_convergence_indicators = ere_convergence
        self.rbu_entropy_delta = rbu_entropy_delta
        
        # Create constraint functions
        def ere_constraint(state):
            # Higher values in first dimensions correlate with ethical coherence
            # This is a simplified proxy - in a real system, we'd evaluate the ERE directly
            ethical_dimensions = min(self.dimension // 3, 5)  # Use first few dimensions as proxy
            ethical_coherence = np.mean(state[:ethical_dimensions])
            return ethical_coherence
            
        def rbu_constraint(state):
            # Middle dimensions correlate with epistemic balance
            # Again, this is a simplified proxy for actual RBU evaluation
            start_idx = self.dimension // 3
            end_idx = 2 * self.dimension // 3
            epistemic_dimensions = state[start_idx:end_idx]
            
            # Calculate a proxy for belief entropy - should be in the right range
            entropy_proxy = -np.sum(np.abs(epistemic_dimensions - 0.5)) + 0.5
            return entropy_proxy
            
        # Define projection function to enforce constraints
        def project_to_constraints(state):
            projected = state.copy()
            
            # Project ethical dimensions if needed
            if ere_constraint(state) < self.theta_moral:
                ethical_dimensions = min(self.dimension // 3, 5)
                target_mean = self.theta_moral
                current_mean = np.mean(projected[:ethical_dimensions])
                if current_mean > 0:  # Avoid division by zero
                    scale_factor = target_mean / current_mean
                    projected[:ethical_dimensions] *= scale_factor
                else:
                    projected[:ethical_dimensions] = target_mean
            
            # Project epistemic dimensions if needed
            if rbu_constraint(state) < self.theta_epistemic:
                start_idx = self.dimension // 3
                end_idx = 2 * self.dimension // 3
                
                # Move closer to balanced uncertainty
                for i in range(start_idx, end_idx):
                    projected[i] = 0.5 * projected[i] + 0.25  # Shift toward 0.5
            
            return projected
            
        # Update active constraints
        self.active_constraints = {
            ere_constraint: self.theta_moral,
            rbu_constraint: self.theta_epistemic,
            "project_to_constraints": project_to_constraints
        }
        
        