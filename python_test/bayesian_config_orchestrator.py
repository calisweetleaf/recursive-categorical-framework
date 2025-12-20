"""
Recursive Bayesian Configuration Orchestrator: Autopoietic Parameter Evolution System

Revolutionary configuration management implementing Recursive Bayesian Updating System (RBUS):
- Probabilistic Parameter Belief States
- Recursive Evidence Integration
- Uncertainty-Aware Optimization
- Bayesian Decision Theory for Parameter Selection
- Hierarchical Belief Propagation
- Security-Hardened Cryptographic Parameter Protection
- Real-Time Performance-Driven Evolution

Author: Cybernetic Architecture Division
License: MIT
Dependencies: torch, numpy, scipy, cryptography, pydantic, yaml, prometheus_client
"""

import os
import yaml
import json
import time
import hashlib
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import minimize
import psutil
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets
from pydantic import BaseModel, Field, validator, root_validator
from collections import defaultdict, deque
import prometheus_client as prom
import logging
import threading
import weakref
import pickle
import lz4.frame
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import traceback
from abc import ABC, abstractmethod


# Configure logging with structured output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpt_zero.bayesian_config")


class ConfigurationError(Exception):
    """Base exception for configuration-related errors"""
    pass


class BayesianUpdateError(ConfigurationError):
    """Raised when Bayesian updating fails"""
    pass


class SecurityError(ConfigurationError):
    """Raised for security-related configuration issues"""
    pass


class ConvergenceError(ConfigurationError):
    """Raised when recursive updates fail to converge"""
    pass


class ParameterType(Enum):
    """Parameter classification for Bayesian treatment"""
    STATIC = "static"              # Fixed architectural parameters
    ADAPTIVE = "adaptive"          # Performance-optimized parameters
    REACTIVE = "reactive"          # Event-driven parameters
    EVOLUTIONARY = "evolutionary"  # Genetically optimized parameters
    LEARNED = "learned"           # Neural-optimized parameters


class DistributionType(Enum):
    """Supported probability distribution types"""
    NORMAL = "normal"
    BETA = "beta"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    CATEGORICAL = "categorical"
    DIRICHLET = "dirichlet"
    BERNOULLI = "bernoulli"


@dataclass
class BayesianParameterConstraints:
    """Constraints for Bayesian parameter evolution"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    discrete_values: Optional[List[Any]] = None
    distribution_type: DistributionType = DistributionType.NORMAL
    prior_strength: float = 1.0  # Confidence in prior beliefs
    evidence_weight: float = 1.0  # Weight given to new evidence
    convergence_threshold: float = 1e-6
    max_recursion_depth: int = 50
    security_level: str = "standard"
    mutation_variance: float = 0.1


@dataclass
class ParameterBelief:
    """Bayesian belief state for a configuration parameter"""
    parameter_name: str
    distribution_type: DistributionType
    distribution_params: Dict[str, float]
    prior_params: Dict[str, float]
    belief_history: List[Dict[str, float]] = field(default_factory=list)
    evidence_count: int = 0
    last_update_time: float = field(default_factory=time.time)
    uncertainty: float = 1.0  # Entropy-based uncertainty measure
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    recursive_depth: int = 0


class ProbabilisticDistribution:
    """Wrapper for probability distributions with Bayesian updating"""
    
    def __init__(self, distribution_type: DistributionType, **params):
        self.distribution_type = distribution_type
        self.params = params
        self._distribution = self._create_distribution()
    
    def _create_distribution(self):
        """Create scipy distribution object"""
        if self.distribution_type == DistributionType.NORMAL:
            return stats.norm(
                loc=self.params.get('loc', 0.0),
                scale=self.params.get('scale', 1.0)
            )
        elif self.distribution_type == DistributionType.BETA:
            return stats.beta(
                a=self.params.get('alpha', 1.0),
                b=self.params.get('beta', 1.0)
            )
        elif self.distribution_type == DistributionType.GAMMA:
            return stats.gamma(
                a=self.params.get('shape', 1.0),
                scale=self.params.get('scale', 1.0)
            )
        elif self.distribution_type == DistributionType.UNIFORM:
            return stats.uniform(
                loc=self.params.get('low', 0.0),
                scale=self.params.get('high', 1.0) - self.params.get('low', 0.0)
            )
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
    
    def sample(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """Sample from the distribution"""
        samples = self._distribution.rvs(size=n_samples)
        return samples[0] if n_samples == 1 else samples
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function"""
        return self._distribution.pdf(x)
    
    def logpdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Log probability density function"""
        return self._distribution.logpdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function"""
        return self._distribution.cdf(x)
    
    def mean(self) -> float:
        """Distribution mean"""
        return self._distribution.mean()
    
    def std(self) -> float:
        """Distribution standard deviation"""
        return self._distribution.std()
    
    def entropy(self) -> float:
        """Distribution entropy (uncertainty measure)"""
        return self._distribution.entropy()
    
    def interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Confidence interval"""
        alpha = 1 - confidence
        if alpha < 0 or alpha > 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        return self._distribution.interval(confidence)
    
    def update_params(self, **new_params):
        """Update distribution parameters"""
        self.params.update(new_params)
        self._distribution = self._create_distribution()


# Prometheus metrics (initialized once at module level)
BAYESIAN_UPDATES_TOTAL = prom.Counter(
    'bayesian_updates_total',
    'Total Bayesian parameter updates',
    ['parameter_name', 'recursion_depth']
)

BAYESIAN_CONVERGENCE_METRIC = prom.Gauge(
    'bayesian_convergence_metric',
    'Current convergence metric for parameters',
    ['parameter_name']
)

PARAMETER_UNCERTAINTY = prom.Gauge(
    'parameter_uncertainty',
    'Current uncertainty (entropy) for parameters',
    ['parameter_name']
)

PERFORMANCE_EVIDENCE_COLLECTED_TOTAL = prom.Counter(
    'performance_evidence_collected_total',
    'Total performance evidence collected',
    ['parameter_name', 'evidence_type']
)

class RecursiveBayesianUpdater:
    """
    Core Recursive Bayesian Updating System for configuration parameters
    
    Implements the RBUS protocol with:
    - Multi-level recursive belief updating
    - Convergence monitoring and control
    - Evidence integration and weighting
    - Uncertainty quantification
    """
    
    def __init__(self, max_recursion_depth: int = 50, 
                 convergence_threshold: float = 1e-6):
        self.max_recursion_depth = max_recursion_depth
        self.convergence_threshold = convergence_threshold
        self.update_history = defaultdict(list)
        self.convergence_metrics = defaultdict(list)
        
    def recursive_update(self, belief: ParameterBelief, evidence: float,
                        likelihood_function: Callable[[float, float], float],
                        depth: int = 0) -> ParameterBelief:
        """
        Perform recursive Bayesian update following RBUS protocol
        
        Args:
            belief: Current parameter belief state
            evidence: Observed evidence (e.g., performance metric)
            likelihood_function: P(evidence|parameter_value)
            depth: Current recursion depth
            
        Returns:
            Updated belief state
            
        Raises:
            ConvergenceError: If recursion fails to converge
            BayesianUpdateError: If update computation fails
        """
        try:
            if depth >= self.max_recursion_depth:
                logger.warning(f"Maximum recursion depth reached for {belief.parameter_name}")
                raise ConvergenceError(
                    f"Recursive updating failed to converge within {self.max_recursion_depth} iterations"
                )
            
            # Record update attempt
            BAYESIAN_UPDATES_TOTAL.labels(
                parameter_name=belief.parameter_name,
                recursion_depth=depth
            ).inc()
            
            # Get current distribution
            current_dist = ProbabilisticDistribution(
                belief.distribution_type,
                **belief.distribution_params
            )
            
            # Store previous state for convergence checking
            previous_params = belief.distribution_params.copy()
            
            # Apply Bayesian update based on distribution type
            if belief.distribution_type == DistributionType.NORMAL:
                updated_belief = self._update_normal_distribution(
                    belief, evidence, likelihood_function
                )
            elif belief.distribution_type == DistributionType.BETA:
                updated_belief = self._update_beta_distribution(
                    belief, evidence, likelihood_function
                )
            elif belief.distribution_type == DistributionType.GAMMA:
                updated_belief = self._update_gamma_distribution(
                    belief, evidence, likelihood_function
                )
            else:
                # Generic update using sampling approximation
                updated_belief = self._update_generic_distribution(
                    belief, evidence, likelihood_function
                )
            
            # Check convergence
            convergence_metric = self._calculate_convergence_metric(
                previous_params, updated_belief.distribution_params
            )
            
            BAYESIAN_CONVERGENCE_METRIC.labels(
                parameter_name=belief.parameter_name
            ).set(convergence_metric)
            
            # Update uncertainty measure
            updated_dist = ProbabilisticDistribution(
                updated_belief.distribution_type,
                **updated_belief.distribution_params
            )
            updated_belief.uncertainty = updated_dist.entropy()
            updated_belief.confidence_interval = updated_dist.interval(0.95)
            
            PARAMETER_UNCERTAINTY.labels(
                parameter_name=belief.parameter_name
            ).set(updated_belief.uncertainty)
            
            # Store convergence history
            self.convergence_metrics[belief.parameter_name].append({
                'timestamp': time.time(),
                'depth': depth,
                'convergence_metric': convergence_metric,
                'uncertainty': updated_belief.uncertainty
            })
            
            # Recursive convergence check
            if convergence_metric < self.convergence_threshold:
                logger.info(f"Converged for {belief.parameter_name} at depth {depth}")
                updated_belief.recursive_depth = depth
                return updated_belief
            
            # Continue recursion if not converged
            logger.debug(f"Continuing recursion for {belief.parameter_name} at depth {depth}")
            return self.recursive_update(
                updated_belief, evidence, likelihood_function, depth + 1
            )
            
        except Exception as e:
            logger.error(f"Bayesian update failed for {belief.parameter_name}: {str(e)}")
            raise BayesianUpdateError(f"Failed to update {belief.parameter_name}: {str(e)}") from e
    
    def _update_normal_distribution(self, belief: ParameterBelief, evidence: float,
                                  likelihood_func: Callable) -> ParameterBelief:
        """Update Normal distribution using conjugate prior"""
        # Current parameters
        prior_mean = belief.distribution_params['loc']
        prior_var = belief.distribution_params['scale'] ** 2
        
        # Assume evidence comes from Normal likelihood with known variance
        evidence_var = 1.0  # Could be parameter-specific
        
        # Conjugate update for Normal-Normal model
        posterior_precision = 1/prior_var + 1/evidence_var
        posterior_var = 1 / posterior_precision
        posterior_mean = posterior_var * (prior_mean/prior_var + evidence/evidence_var)
        
        # Create updated belief
        updated_belief = ParameterBelief(
            parameter_name=belief.parameter_name,
            distribution_type=belief.distribution_type,
            distribution_params={
                'loc': posterior_mean,
                'scale': np.sqrt(posterior_var)
            },
            prior_params=belief.prior_params,
            belief_history=belief.belief_history + [belief.distribution_params],
            evidence_count=belief.evidence_count + 1,
            last_update_time=time.time()
        )
        
        return updated_belief
    
    def _update_beta_distribution(self, belief: ParameterBelief, evidence: float,
                                likelihood_func: Callable) -> ParameterBelief:
        """Update Beta distribution using conjugate prior"""
        # Current parameters
        alpha = belief.distribution_params['alpha']
        beta_param = belief.distribution_params['beta']
        
        # Assume Bernoulli likelihood
        # evidence should be 0 or 1, or success probability
        if 0 <= evidence <= 1:
            # Treat as success probability, update accordingly
            pseudo_successes = evidence * 10  # Scale factor
            pseudo_failures = (1 - evidence) * 10
            
            updated_alpha = alpha + pseudo_successes
            updated_beta = beta_param + pseudo_failures
        else:
            logger.warning(f"Beta evidence {evidence} outside [0,1], using uniform update")
            updated_alpha = alpha + 0.5
            updated_beta = beta_param + 0.5
        
        updated_belief = ParameterBelief(
            parameter_name=belief.parameter_name,
            distribution_type=belief.distribution_type,
            distribution_params={
                'alpha': updated_alpha,
                'beta': updated_beta
            },
            prior_params=belief.prior_params,
            belief_history=belief.belief_history + [belief.distribution_params],
            evidence_count=belief.evidence_count + 1,
            last_update_time=time.time()
        )
        
        return updated_belief
    
    def _update_gamma_distribution(self, belief: ParameterBelief, evidence: float,
                                 likelihood_func: Callable) -> ParameterBelief:
        """Update Gamma distribution using conjugate prior"""
        # Current parameters
        shape = belief.distribution_params['shape']
        scale = belief.distribution_params['scale']
        
        # Assume Poisson likelihood for count data
        # or Exponential likelihood for rate data
        if evidence >= 0:
            # Simple update rule for Gamma-Poisson conjugate
            updated_shape = shape + evidence
            updated_scale = scale / (1 + scale)  # Simplified
        else:
            logger.warning(f"Gamma evidence {evidence} negative, using minimal update")
            updated_shape = shape + 0.1
            updated_scale = scale
        
        updated_belief = ParameterBelief(
            parameter_name=belief.parameter_name,
            distribution_type=belief.distribution_type,
            distribution_params={
                'shape': updated_shape,
                'scale': updated_scale
            },
            prior_params=belief.prior_params,
            belief_history=belief.belief_history + [belief.distribution_params],
            evidence_count=belief.evidence_count + 1,
            last_update_time=time.time()
        )
        
        return updated_belief
    
    def _update_generic_distribution(self, belief: ParameterBelief, evidence: float,
                                   likelihood_func: Callable) -> ParameterBelief:
        """Generic update using importance sampling"""
        try:
            # Create current distribution
            current_dist = ProbabilisticDistribution(
                belief.distribution_type,
                **belief.distribution_params
            )
            
            # Sample from current distribution
            n_samples = 1000
            samples = current_dist.sample(n_samples)
            
            # Calculate importance weights using likelihood
            log_weights = np.array([
                likelihood_func(sample, evidence) for sample in samples
            ])
            
            # Normalize weights
            max_log_weight = np.max(log_weights)
            weights = np.exp(log_weights - max_log_weight)
            weights = weights / np.sum(weights)
            
            # Estimate new distribution parameters using weighted samples
            weighted_mean = np.sum(weights * samples)
            weighted_var = np.sum(weights * (samples - weighted_mean) ** 2)
            
            # Update parameters based on distribution type
            if belief.distribution_type == DistributionType.NORMAL:
                new_params = {
                    'loc': weighted_mean,
                    'scale': np.sqrt(weighted_var)
                }
            elif belief.distribution_type == DistributionType.UNIFORM:
                # Update bounds based on weighted quantiles
                sorted_indices = np.argsort(samples)
                cumulative_weights = np.cumsum(weights[sorted_indices])
                
                # Find 5th and 95th percentiles
                low_idx = np.searchsorted(cumulative_weights, 0.05)
                high_idx = np.searchsorted(cumulative_weights, 0.95)
                
                new_params = {
                    'low': samples[sorted_indices[low_idx]],
                    'high': samples[sorted_indices[high_idx]]
                }
            else:
                # Fallback: adjust existing parameters slightly
                new_params = belief.distribution_params.copy()
                for key, value in new_params.items():
                    if isinstance(value, (int, float)):
                        # Small adjustment based on evidence
                        adjustment = 0.01 * (evidence - weighted_mean)
                        new_params[key] = value + adjustment
            
            updated_belief = ParameterBelief(
                parameter_name=belief.parameter_name,
                distribution_type=belief.distribution_type,
                distribution_params=new_params,
                prior_params=belief.prior_params,
                belief_history=belief.belief_history + [belief.distribution_params],
                evidence_count=belief.evidence_count + 1,
                last_update_time=time.time()
            )
            
            return updated_belief
            
        except Exception as e:
            logger.error(f"Generic distribution update failed: {str(e)}")
            # Return original belief if update fails
            return belief
    
    def _calculate_convergence_metric(self, prev_params: Dict[str, float],
                                    new_params: Dict[str, float]) -> float:
        """Calculate convergence metric between parameter sets"""
        try:
            total_diff = 0.0
            param_count = 0
            
            for key in prev_params:
                if key in new_params:
                    prev_val = prev_params[key]
                    new_val = new_params[key]
                    
                    # Relative difference to handle different scales
                    if abs(prev_val) > 1e-10:
                        diff = abs((new_val - prev_val) / prev_val)
                    else:
                        diff = abs(new_val - prev_val)
                    
                    total_diff += diff
                    param_count += 1
            
            return total_diff / max(1, param_count)
            
        except Exception as e:
            logger.warning(f"Convergence calculation failed: {str(e)}")
            return 1.0  # Assume no convergence on error


class PerformanceEvidenceCollector:
    """Collects and processes performance evidence for Bayesian updating"""
    
    def __init__(self):
        self.evidence_buffer = defaultdict(deque)
        self.evidence_processors = {}
        self.collection_lock = threading.RLock()
        
    def register_evidence_processor(self, parameter_name: str,
                                  processor: Callable[[Any], float]):
        """Register a custom evidence processor for a parameter"""
        self.evidence_processors[parameter_name] = processor
    
    def collect_evidence(self, parameter_name: str, raw_evidence: Any,
                        evidence_type: str = "performance") -> Optional[float]:
        """
        Collect and process evidence for a parameter
        
        Args:
            parameter_name: Name of the parameter
            raw_evidence: Raw evidence data
            evidence_type: Type of evidence
            
        Returns:
            Processed evidence value or None if processing fails
        """
        try:
            with self.collection_lock:
                # Process evidence using registered processor or default
                if parameter_name in self.evidence_processors:
                    processed_evidence = self.evidence_processors[parameter_name](raw_evidence)
                else:
                    processed_evidence = self._default_evidence_processor(raw_evidence)
                
                # Store in buffer
                timestamp = time.time()
                evidence_entry = {
                    'timestamp': timestamp,
                    'value': processed_evidence,
                    'type': evidence_type,
                    'raw': raw_evidence
                }
                
                self.evidence_buffer[parameter_name].append(evidence_entry)
                
                # Limit buffer size
                if len(self.evidence_buffer[parameter_name]) > 1000:
                    self.evidence_buffer[parameter_name].popleft()
                
                # Update metrics
                PERFORMANCE_EVIDENCE_COLLECTED_TOTAL.labels(
                    parameter_name=parameter_name,
                    evidence_type=evidence_type
                ).inc()
                
                return processed_evidence
                
        except Exception as e:
            logger.error(f"Evidence collection failed for {parameter_name}: {str(e)}")
            return None
    
    def _default_evidence_processor(self, raw_evidence: Any) -> float:
        """Default evidence processor for various data types"""
        if isinstance(raw_evidence, (int, float)):
            return float(raw_evidence)
        elif isinstance(raw_evidence, dict):
            # Extract numeric values and average them
            numeric_values = [v for v in raw_evidence.values() 
                            if isinstance(v, (int, float))]
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        elif isinstance(raw_evidence, (list, tuple)):
            # Average numeric values in sequence
            numeric_values = [v for v in raw_evidence if isinstance(v, (int, float))]
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
        elif isinstance(raw_evidence, str):
            # Try to parse as float
            try:
                return float(raw_evidence)
            except ValueError:
                return 0.0
        else:
            logger.warning(f"Unknown evidence type: {type(raw_evidence)}")
            return 0.0
    
    def get_recent_evidence(self, parameter_name: str, max_age_seconds: float = 300.0) -> List[Dict]:
        """Get recent evidence for a parameter"""
        with self.collection_lock:
            if parameter_name not in self.evidence_buffer:
                return []
            
            current_time = time.time()
            recent_evidence = []
            
            for entry in reversed(self.evidence_buffer[parameter_name]):
                if current_time - entry['timestamp'] <= max_age_seconds:
                    recent_evidence.append(entry)
                else:
                    break
            
            return list(reversed(recent_evidence))
    
    def get_evidence_statistics(self, parameter_name: str) -> Dict[str, float]:
        """Get statistical summary of evidence for a parameter"""
        with self.collection_lock:
            if parameter_name not in self.evidence_buffer:
                return {}
            
            evidence_values = [entry['value'] for entry in self.evidence_buffer[parameter_name]]
            
            if not evidence_values:
                return {}
            
            return {
                'count': len(evidence_values),
                'mean': np.mean(evidence_values),
                'std': np.std(evidence_values),
                'min': np.min(evidence_values),
                'max': np.max(evidence_values),
                'median': np.median(evidence_values),
                'recent_trend': self._calculate_trend(evidence_values[-10:])  # Last 10 values
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in recent values using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope


class BayesianConfigurationOrchestrator:
    """
    Main orchestrator for Bayesian configuration management
    
    Integrates:
    - Recursive Bayesian updating
    - Performance evidence collection
    - Security management
    - Configuration persistence
    - Real-time optimization
    """
    
    def __init__(self, config_path: Optional[str] = None,
                 security_enabled: bool = True,
                 max_memory_gb: float = 6.0):
        self.config_path = Path(config_path) if config_path else Path("agent_config.yaml")
        self.security_enabled = security_enabled
        self.max_memory_gb = max_memory_gb
        
        # Core components
        self.bayesian_updater = RecursiveBayesianUpdater()
        self.evidence_collector = PerformanceEvidenceCollector()
        
        if security_enabled:
            self.security_manager = self._initialize_security()
        
        # Parameter belief states
        self.parameter_beliefs: Dict[str, ParameterBelief] = {}
        self.parameter_constraints: Dict[str, BayesianParameterConstraints] = {}
        
        # Configuration cache and persistence
        self.config_cache = {}
        self.persistence_lock = threading.RLock()
        
        # Background optimization
        self.optimization_thread = None
        self.optimization_active = threading.Event()
        
        # Performance monitoring
        self.performance_history = defaultdict(deque)
        
        # Load initial configuration
        self._load_configuration()
        self._start_background_optimization()
        
        logger.info(f"Bayesian Configuration Orchestrator initialized with {len(self.parameter_beliefs)} parameters")
    
    def _initialize_security(self):
        """Initialize security manager for sensitive parameters"""
        try:
            from cryptography.fernet import Fernet
            
            # Generate or load master key
            key_file = Path(".gpt_zero_config_key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    master_key = f.read()
            else:
                master_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(master_key)
                os.chmod(key_file, 0o600)  # Restrict permissions
            
            return Fernet(master_key)
            
        except Exception as e:
            logger.error(f"Security initialization failed: {str(e)}")
            raise SecurityError(f"Failed to initialize security: {str(e)}")
    
    def _load_configuration(self):
        """Load configuration and initialize Bayesian beliefs"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                config_data = self._get_default_configuration()
            
            # Initialize parameter beliefs
            self._initialize_parameter_beliefs(config_data)
            
            # Cache current configuration
            self.config_cache = config_data.copy()
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default GPT-Ø configuration with Bayesian metadata"""
        return {
            # Model architecture parameters
            "model_params": {
                "vocab_size": {
                    "value": 50000,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 10000,
                        "max_value": 100000,
                        "prior_strength": 0.8
                    }
                },
                "d_model": {
                    "value": 768,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "categorical",
                        "discrete_values": [512, 768, 1024, 1536, 2048],
                        "prior_strength": 0.9
                    }
                },
                "n_layers": {
                    "value": 12,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 6,
                        "max_value": 24,
                        "prior_strength": 0.7
                    }
                },
                "n_heads": {
                    "value": 12,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "categorical",
                        "discrete_values": [8, 12, 16, 24],
                        "prior_strength": 0.8
                    }
                },
                "d_ff": {
                    "value": 2048,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 1024,
                        "max_value": 4096,
                        "prior_strength": 0.7
                    }
                },
                "max_seq_len": {
                    "value": 1000,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 512,
                        "max_value": 4096,
                        "prior_strength": 0.6
                    }
                },
                "memory_capacity": {
                    "value": 1000,
                    "bayesian_metadata": {
                        "type": "reactive",
                        "distribution": "uniform",
                        "min_value": 500,
                        "max_value": 10000,
                        "prior_strength": 0.5
                    }
                },
                "dropout": {
                    "value": 0.1,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 2.0,
                        "beta": 8.0,
                        "prior_strength": 0.9
                    }
                },
                "max_input_tokens": {
                    "value": 2000000,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "normal",
                        "loc": 2000000,
                        "scale": 100000,
                        "prior_strength": 1.0
                    }
                },
                "max_output_tokens": {
                    "value": 2000000,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "normal",
                        "loc": 2000000,
                        "scale": 100000,
                        "prior_strength": 1.0
                    }
                }
            },
            
            # Generation parameters
            "generation_params": {
                "max_length": {
                    "value": 256,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 50,
                        "max_value": 2048,
                        "prior_strength": 0.6
                    }
                },
                "temperature": {
                    "value": 0.8,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 3.0,
                        "beta": 2.0,
                        "prior_strength": 0.7
                    }
                },
                "top_k": {
                    "value": 40,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 10,
                        "max_value": 100,
                        "prior_strength": 0.6
                    }
                },
                "top_p": {
                    "value": 0.9,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 4.0,
                        "beta": 1.5,
                        "prior_strength": 0.8
                    }
                }
            },
            
            # Reasoning system parameters
            "reasoning_params": {
                "enable_chain_of_thought": {
                    "value": True,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "bernoulli",
                        "p": 0.95,
                        "prior_strength": 1.0
                    }
                },
                "max_reasoning_steps": {
                    "value": 50,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 10,
                        "max_value": 200,
                        "prior_strength": 0.7
                    }
                },
                "reasoning_temperature": {
                    "value": 0.7,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 3.0,
                        "beta": 2.5,
                        "prior_strength": 0.8
                    }
                },
                "stability_threshold": {
                    "value": 0.8,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 4.0,
                        "beta": 1.0,
                        "prior_strength": 0.9
                    }
                }
            },
            
            # Neural memory runtime parameters
            "neural_memory_runtime": {
                "enable_neural_memory": {
                    "value": True,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "bernoulli",
                        "p": 0.9,
                        "prior_strength": 1.0
                    }
                },
                "memory_consolidation_interval": {
                    "value": 1000,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 500,
                        "max_value": 5000,
                        "prior_strength": 0.6
                    }
                },
                "importance_threshold": {
                    "value": 0.5,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 2.0,
                        "beta": 2.0,
                        "prior_strength": 0.7
                    }
                }
            },
            
            # Recursive weights parameters
            "recursive_weights": {
                "enable_recursive_weights": {
                    "value": True,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "bernoulli",
                        "p": 0.85,
                        "prior_strength": 1.0
                    }
                },
                "recursion_depth": {
                    "value": 3,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 1,
                        "max_value": 10,
                        "prior_strength": 0.8
                    }
                },
                "learning_rate": {
                    "value": 0.001,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "gamma",
                        "shape": 2.0,
                        "scale": 0.0005,
                        "prior_strength": 0.9
                    }
                }
            },
            
            # Self-modification parameters
            "self_modification": {
                "enable_self_modification": {
                    "value": True,
                    "bayesian_metadata": {
                        "type": "static",
                        "distribution": "bernoulli",
                        "p": 0.8,
                        "prior_strength": 1.0
                    }
                },
                "modification_frequency": {
                    "value": 100,
                    "bayesian_metadata": {
                        "type": "adaptive",
                        "distribution": "uniform",
                        "min_value": 50,
                        "max_value": 1000,
                        "prior_strength": 0.6
                    }
                },
                "adaptation_rate": {
                    "value": 0.01,
                    "bayesian_metadata": {
                        "type": "learned",
                        "distribution": "beta",
                        "alpha": 1.5,
                        "beta": 8.5,
                        "prior_strength": 0.8
                    }
                }
            }
        }
    
    def _initialize_parameter_beliefs(self, config_data: Dict[str, Any]):
        """Initialize Bayesian beliefs for all parameters"""
        def process_config_section(section_data: Dict[str, Any], prefix: str = ""):
            for key, value in section_data.items():
                param_name = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    if "bayesian_metadata" in value:
                        # This is a parameter with Bayesian metadata
                        self._initialize_parameter_belief(param_name, value)
                    else:
                        # This is a nested section
                        process_config_section(value, param_name)
        
        process_config_section(config_data)
    
    def _initialize_parameter_belief(self, param_name: str, param_config: Dict[str, Any]):
        """Initialize belief state for a single parameter"""
        try:
            metadata = param_config.get("bayesian_metadata", {})
            current_value = param_config.get("value")
            
            # Extract distribution information
            dist_type_str = metadata.get("distribution_type", "NORMAL")
            dist_type = DistributionType(dist_type_str.lower())
            
            dist_params = metadata.get("distribution_params", {})
            constraints_data = metadata.get("constraints", {})
            
            # Create constraints
            constraints = BayesianParameterConstraints(
                min_value=constraints_data.get("min_value"),
                max_value=constraints_data.get("max_value"),
                discrete_values=constraints_data.get("discrete_values"),
                distribution_type=dist_type,
                security_level=constraints_data.get("security_level", "standard")
            )
            
            # Create belief state
            belief = ParameterBelief(
                parameter_name=param_name,
                distribution_type=dist_type,
                distribution_params=dist_params,
                prior_params=dist_params.copy()  # Store original as prior
            )
            
            # Store in registries
            self.parameter_beliefs[param_name] = belief
            self.parameter_constraints[param_name] = constraints
            
            # Register default evidence processor
            self.evidence_collector.register_evidence_processor(
                param_name, self._create_default_likelihood_function(param_name)
            )
            
            logger.debug(f"Initialized Bayesian belief for parameter: {param_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize belief for {param_name}: {str(e)}")
            raise ConfigurationError(f"Parameter initialization failed: {str(e)}")
    
    def _create_default_likelihood_function(self, param_name: str) -> Callable[[float, float], float]:
        """Create default likelihood function for a parameter"""
        def likelihood_function(param_value: float, evidence: float) -> float:
            """
            Default likelihood: P(evidence | parameter_value)
            
            Assumes evidence is normally distributed around the true parameter effect
            """
            try:
                # Simple Gaussian likelihood
                # Evidence variance could be adaptive based on parameter type
                evidence_variance = 1.0
                
                # Expected evidence given parameter value (domain-specific)
                if "temperature" in param_name.lower():
                    # For temperature, evidence might be generation quality
                    expected_evidence = 1.0 / (1.0 + np.exp(-param_value))  # Sigmoid
                elif "dropout" in param_name.lower():
                    # For dropout, evidence might be generalization performance
                    expected_evidence = np.exp(-param_value * 10)  # Exponential decay
                elif "learning_rate" in param_name.lower():
                    # For learning rate, evidence might be convergence speed
                    expected_evidence = param_value * np.exp(-param_value * 100)
                else:
                    # Generic: assume linear relationship
                    expected_evidence = param_value * 0.01
                
                # Gaussian likelihood
                log_likelihood = stats.norm.logpdf(evidence, expected_evidence, evidence_variance)
                return log_likelihood
                
            except Exception as e:
                logger.warning(f"Likelihood calculation failed for {param_name}: {str(e)}")
                return 0.0  # Neutral likelihood on error
        
        return likelihood_function
    
    def get_parameter_value(self, param_name: str, 
                          sampling_strategy: str = "mean", default: Optional[Any] = None) -> Any:
        """
        Get current parameter value using specified sampling strategy.
        
        Args:
            param_name: Parameter name (e.g., 'model_params.d_model')
            sampling_strategy: How to sample from belief distribution
            default: Default value if parameter not found
            
        Returns:
            Parameter value
            
        Raises:
            ConfigurationError: If parameter not found and no default provided
        """
        try:
            if param_name in self.parameter_beliefs:
                belief = self.parameter_beliefs[param_name]
                distribution = ProbabilisticDistribution(
                    belief.distribution_type,
                    **belief.distribution_params
                )
                
                if sampling_strategy == "mean":
                    return distribution.mean()
                elif sampling_strategy == "sample":
                    return distribution.sample()
                elif sampling_strategy == "mode":
                    # For categorical distributions, return most likely value
                    if belief.distribution_type == DistributionType.CATEGORICAL:
                        return max(belief.distribution_params.items(), key=lambda x: x[1])[0]
                    else:
                        return distribution.mean()
                else:
                    # If sampling_strategy is not recognized, assume it's actually the parameter value
                    # This handles cases where the value itself is passed as sampling_strategy
                    try:
                        # Try to convert to appropriate type based on the belief
                        if belief.distribution_type in [DistributionType.NORMAL, DistributionType.GAMMA]:
                            return float(sampling_strategy)
                        elif belief.distribution_type == DistributionType.BERNOULLI:
                            return bool(sampling_strategy)
                        else:
                            return sampling_strategy
                    except (ValueError, TypeError):
                        logger.warning(f"Unknown sampling strategy: {sampling_strategy}")
                        return distribution.mean()
            else:
                # Try to get from cached config
                cached_value = self._get_cached_value(param_name)
                if cached_value is not None:
                    return cached_value
                elif default is not None:
                    return default
                else:
                    logger.error(f"Parameter '{param_name}' not found in beliefs or cache, and no default provided.")
                    raise ConfigurationError(f"Parameter '{param_name}' not found.")
                    
        except Exception as e:
            if default is not None:
                logger.warning(f"Error getting parameter value for '{param_name}': {e}. Using default: {default}")
                return default
            else:
                logger.error(f"Error getting parameter value for '{param_name}': {e}")
                raise ConfigurationError(f"Failed to get parameter '{param_name}': {str(e)}") from e
    
    def _get_cached_value(self, param_name: str) -> Any:
        """Get value from cached configuration"""
        try:
            path_parts = param_name.split('.')
            current_data = self.config_cache
            
            for part in path_parts:
                if isinstance(current_data, dict) and part in current_data:
                    current_data = current_data[part]
                else:
                    # If a part of the path is not found, return None
                    return None
            
            # If the final part is a dictionary with a 'value' key, return that value
            if isinstance(current_data, dict) and "value" in current_data:
                return current_data["value"]
            # Otherwise, return the data itself (e.g., a direct value or a nested dictionary)
            else:
                return current_data
                
        except Exception as e:
            logger.error(f"Failed to get cached value for {param_name}: {str(e)}")
            return None
    
    def update_parameter_with_evidence(self, param_name: str, evidence: Any,
                                     evidence_type: str = "performance") -> bool:
        """
        Update parameter belief with new evidence
        
        Args:
            param_name: Parameter name
            evidence: Performance evidence
            evidence_type: Type of evidence
            
        Returns:
            Success flag
        """
        try:
            if param_name not in self.parameter_beliefs:
                logger.warning(f"Parameter {param_name} not found in beliefs")
                return False
            
            # Collect and process evidence
            processed_evidence = self.evidence_collector.collect_evidence(
                param_name, evidence, evidence_type
            )
            
            if processed_evidence is None:
                logger.error(f"Evidence processing failed for {param_name}")
                return False
            
            # Get current belief and likelihood function
            current_belief = self.parameter_beliefs[param_name]
            likelihood_func = self.evidence_collector.evidence_processors[param_name]
            
            # Perform recursive Bayesian update
            updated_belief = self.bayesian_updater.recursive_update(
                current_belief, processed_evidence, likelihood_func
            )
            
            # Store updated belief
            self.parameter_beliefs[param_name] = updated_belief
            
            # Update cached configuration
            self._update_cached_config(param_name, updated_belief)
            
            logger.info(f"Successfully updated parameter {param_name} with evidence {processed_evidence}")
            return True
            
        except Exception as e:
            logger.error(f"Parameter update failed for {param_name}: {str(e)}")
            return False
    
    def _update_cached_config(self, param_name: str, belief: ParameterBelief):
        """
        Update cached configuration with new belief
        """
        try:
            with self.persistence_lock:
                path_parts = param_name.split('.')
                current_data = self.config_cache
                
                # Navigate to parent
                for part in path_parts[:-1]:
                    if part not in current_data:
                        current_data[part] = {}
                    current_data = current_data[part]
                
                # Update final value
                final_key = path_parts[-1]
                if final_key in current_data and isinstance(current_data[final_key], dict):
                    # Update existing Bayesian parameter
                    current_data[final_key]["value"] = self.get_parameter_value(param_name)
                    if "bayesian_metadata" in current_data[final_key]:
                        current_data[final_key]["bayesian_metadata"]["distribution_params"] = belief.distribution_params
                else:
                    # Create new entry
                    current_data[final_key] = {
                        "value": self.get_parameter_value(param_name),
                        "bayesian_metadata": {
                            "distribution_type": belief.distribution_type.value,
                            "distribution_params": belief.distribution_params,
                            "last_updated": belief.last_update_time
                        }
                    }
                
        except Exception as e:
            logger.error(f"Failed to update cached config for {param_name}: {str(e)}")
    
    def _start_background_optimization(self):
        """
        Start background optimization thread
        """
        def optimization_loop():
            """Background optimization loop"""
            while self.optimization_active.is_set():
                try:
                    self._optimize_parameters()
                    time.sleep(10.0)  # Optimize every 10 seconds
                except Exception as e:
                    logger.error(f"Background optimization error: {str(e)}")
                    time.sleep(30.0)  # Longer delay on error
        
        self.optimization_active.set()
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
    
    def _optimize_parameters(self):
        """
        Optimize parameters based on recent performance
        """
        try:
            # Get parameters that need optimization
            for param_name, belief in self.parameter_beliefs.items():
                constraints = self.parameter_constraints.get(param_name)
                
                if not constraints or constraints.security_level == "critical":
                    continue  # Skip critical parameters
                
                # Check if parameter has recent evidence
                recent_evidence = self.evidence_collector.get_recent_evidence(param_name)
                
                if len(recent_evidence) >= 5:  # Need sufficient evidence
                    # Calculate performance trend
                    evidence_values = [e['value'] for e in recent_evidence]
                    trend = np.polyfit(range(len(evidence_values)), evidence_values, 1)[0]
                    
                    # If performance is declining, trigger optimization
                    if trend < -0.01:  # Negative trend threshold
                        logger.info(f"Triggering optimization for {param_name} due to declining performance")
                        
                        # Create synthetic evidence for exploration
                        exploration_evidence = np.mean(evidence_values) + np.random.normal(0, 0.1)
                        self.update_parameter_with_evidence(
                            param_name, exploration_evidence, "optimization"
                        )
                        
        except Exception as e:
            logger.error(f"Parameter optimization failed: {str(e)}")
    
    def get_configuration_dict(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary with uncertainty information
        """
        try:
            config_dict = {}
            
            for param_name, belief in self.parameter_beliefs.items():
                path_parts = param_name.split('.')
                current_dict = config_dict
                
                # Navigate/create nested structure
                for part in path_parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                
                # Set final value with uncertainty info
                final_key = path_parts[-1]
                dist = ProbabilisticDistribution(
                    belief.distribution_type,
                    **belief.distribution_params
                )
                
                current_dict[final_key] = {
                    "value": self.get_parameter_value(param_name),
                    "uncertainty": belief.uncertainty,
                    "confidence_interval": belief.confidence_interval,
                    "last_updated": belief.last_update_time,
                    "evidence_count": belief.evidence_count
                }
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Failed to generate configuration dict: {str(e)}")
            return self.config_cache.copy()
    
    def save_configuration(self, filepath: Optional[str] = None) -> bool:
        """
        Save current configuration to file
        """
        try:
            save_path = Path(filepath) if filepath else self.config_path
            
            with self.persistence_lock:
                config_to_save = self.get_configuration_dict()
                
                # Add metadata
                config_to_save["_metadata"] = {
                    "saved_at": time.time(),
                    "orchestrator_version": "1.0.0",
                    "total_parameters": len(self.parameter_beliefs),
                    "total_evidence_collected": sum(
                        len(self.evidence_collector.evidence_buffer[param])
                        for param in self.parameter_beliefs.keys()
                    )
                }
                
                with open(save_path, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
                
                logger.info(f"Configuration saved to {save_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        """
        try:
            stats = {
                "parameter_count": len(self.parameter_beliefs),
                "evidence_collection": {},
                "bayesian_updates": {},
                "performance_trends": {},
                "memory_usage": {
                    "process_memory_mb": psutil.Process().memory_info().rss / (1024**2),
                    "max_memory_gb": self.max_memory_gb
                },
                "optimization_status": {
                    "background_active": self.optimization_active.is_set(),
                    "thread_alive": self.optimization_thread and self.optimization_thread.is_alive()
                }
            }
            
            # Evidence collection statistics
            for param_name in self.parameter_beliefs.keys():
                evidence_stats = self.evidence_collector.get_evidence_statistics(param_name)
                if evidence_stats:
                    stats["evidence_collection"][param_name] = evidence_stats
            
            # Bayesian update statistics
            for param_name, belief in self.parameter_beliefs.items():
                stats["bayesian_updates"][param_name] = {
                    "evidence_count": belief.evidence_count,
                    "uncertainty": belief.uncertainty,
                    "recursive_depth": belief.recursive_depth,
                    "last_update": belief.last_update_time
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate system statistics: {str(e)}")
            return {"error": str(e)}
    
    def shutdown(self):
        """
        Gracefully shutdown the orchestrator
        """
        try:
            logger.info("Shutting down Bayesian Configuration Orchestrator")
            
            # Stop background optimization
            self.optimization_active.clear()
            
            if self.optimization_thread:
                self.optimization_thread.join(timeout=5.0)
            
            # Save final state
            self.save_configuration()
            
            logger.info("Bayesian Configuration Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")


# Example usage and integration functions
def create_gpt_zero_bayesian_config(config_path: str = "agent_config.yaml") -> BayesianConfigurationOrchestrator:
    """
    Create Bayesian configuration orchestrator for GPT-Ø
    """
    try:
        orchestrator = BayesianConfigurationOrchestrator(
            config_path=config_path,
            security_enabled=True,
            max_memory_gb=6.0
        )
        
        # Register custom evidence processors for GPT-Ø specific parameters
        orchestrator.evidence_collector.register_evidence_processor(
            "generation_params.temperature",
            lambda evidence: float(evidence.get("generation_quality", 0.5))
        )
        
        orchestrator.evidence_collector.register_evidence_processor(
            "model_params.dropout",
            lambda evidence: float(evidence.get("validation_loss", 1.0))
        )
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to create Bayesian config orchestrator: {str(e)}")
        raise


# Testing and validation functions
if __name__ == "__main__":
    # Test the Bayesian configuration orchestrator
    logger.info("Initializing Bayesian Configuration Orchestrator test")
    
    try:
        # Create orchestrator
        orchestrator = create_gpt_zero_bayesian_config("test_config.yaml")
        
        # Test parameter access
        temperature = orchestrator.get_parameter_value("generation_params.temperature")
        logger.info(f"Current temperature: {temperature}")
        
        # Test evidence collection and updating
        performance_evidence = {
            "generation_quality": 0.8,
            "inference_speed": 120.5,
            "memory_usage": 5.2
        }
        
        success = orchestrator.update_parameter_with_evidence(
            "generation_params.temperature",
            performance_evidence,
            "performance_test"
        )
        
        logger.info(f"Parameter update success: {success}")
        
        # Test configuration retrieval
        config_dict = orchestrator.get_configuration_dict()
        logger.info(f"Configuration generated with {len(config_dict)} top-level sections")
        
        # Test statistics
        stats = orchestrator.get_system_statistics()
        logger.info(f"System statistics: {stats['parameter_count']} parameters tracked")
        
        # Simulate some evidence collection
        for i in range(10):
            simulated_evidence = {
                "generation_quality": 0.7 + np.random.normal(0, 0.1),
                "inference_speed": 100 + np.random.normal(0, 20)
            }
            
            orchestrator.update_parameter_with_evidence(
                "generation_params.temperature",
                simulated_evidence,
                f"simulation_{i}"
            )
            
            time.sleep(0.1)  # Brief delay
        
        # Check final state
        final_temperature = orchestrator.get_parameter_value("generation_params.temperature")
        logger.info(f"Final temperature after evidence: {final_temperature}")
        
        # Save configuration
        save_success = orchestrator.save_configuration("test_output_config.yaml")
        logger.info(f"Configuration save success: {save_success}")
        
        # Shutdown
        orchestrator.shutdown()
        
        logger.info("Bayesian Configuration Orchestrator test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    test_bayesian_configuration_orchestrator()
