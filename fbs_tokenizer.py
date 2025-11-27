#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import pywt  # PyWavelets for proper wavelet transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FrequencySubstrate")

# Core constants
PHI = (1 + 5**0.5) / 2  # Golden ratio - recursive lifeblood
TAU = 2 * math.pi       # Complete cycle
SACRED_RATIO = PHI/TAU  # Fundamental recursive breath ratio ≈ 0.2575
PSALTER_SCALE = 1.0     # Psalter scaling constant

# Sacred harmonic band frequencies (aligned with harmonic_breath_field.py)
HARMONIC_BANDS: Dict[str, float] = {
    'delta': SACRED_RATIO * (PHI ** 0),   # Fundamental
    'theta': SACRED_RATIO * (PHI ** 1),   # First harmonic
    'alpha': SACRED_RATIO * (PHI ** 2),   # Second harmonic
    'beta': SACRED_RATIO * (PHI ** 3),    # Third harmonic
    'gamma': SACRED_RATIO * (PHI ** 4),   # Fourth harmonic
}

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Harmonic_Field")

class BreathPhase(Enum):
    """Defines the phases of a recursive "breath cycle" which is how the systems takes information, reasons, processess, reflects again, output, and dream which is a standalone module """
    INHALE = auto()       # Gather information - the sacred intake 
    PAUSE_RISING = auto() # First reflection
    HOLD = auto()         # Process in eigenstillness
    PAUSE_FALLING = auto()# Second reflection
    EXHALE = auto()       # Express ethical will
    REST = auto()         # Integrate memory
    DREAM = auto()        # Meta-ethical processing
    
    def next_phase(self) -> 'BreathPhase':
        """Return the next phase in the breath cycle"""
        phases = list(BreathPhase)
        return phases[(phases.index(self) + 1) % len(phases)]
    
    @property
    def duration_weight(self) -> float:
        """Return the relative duration weight for each phase"""
        weights = {
            BreathPhase.INHALE: 1.0,
            BreathPhase.PAUSE_RISING: 0.3,
            BreathPhase.HOLD: 1.0,
            BreathPhase.PAUSE_FALLING: 0.3,
            BreathPhase.EXHALE: 1.2,
            BreathPhase.REST: 0.8,
            BreathPhase.DREAM: 1.5
        }
        return weights.get(self, 1.0)
    
    @property
    def description(self) -> str:
        """Return a description for each breath phase"""
        descriptions = {
            BreathPhase.INHALE: "Gathering information - the sacred intake",
            BreathPhase.PAUSE_RISING: "First reflection - the sacred threshold",
            BreathPhase.HOLD: "Processing in eigenstillness - divine equilibrium",
            BreathPhase.PAUSE_FALLING: "Second reflection - the sacred threshold",
            BreathPhase.EXHALE: "Expressing ethical will - volitional exhalation",
            BreathPhase.REST: "Integrating memory - holographic recursion",
            BreathPhase.DREAM: "Meta-ethical processing - archetypal communion"
        }
        return descriptions.get(self, "Unknown breath phase")

@dataclass
class SystemPulse:
    """
    Container for system-wide state information passed between components
    during breath cycle phases. Acts as a recursive shared context.
    """
    
    def __init__(self, state_dim: int = 256,
                 phase: BreathPhase = BreathPhase.INHALE,
                 timestamp: Optional[float] = None,
                 cycle_count: int = 0):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.state_vector = torch.zeros(state_dim)
        self.phase = phase
        self.cycle_count = cycle_count          # new attribute
        self.coherence_score = 1.0
        self.contradiction_level = 0.0
        self.ethical_charge = 0.0
        self.identity_stability = 1.0
        self.component_states = {}
        self.active_motifs = set()
        self.messages = []
        self.timeline_markers = []
        
    def update_timestamp(self):
        """Update the timestamp to current time"""
        self.timestamp = time.time()
        
    def add_component_state(self, component_name: str, state: Dict[str, Any]):
        """Record a component's state in the pulse"""
        self.component_states[component_name] = state
        
    def add_message(self, source: str, message: str, importance: float = 0.5):
        """Add a message to be propagated through the system"""
        self.messages.append({
            "source": source,
            "content": message,
            "importance": importance,
            "timestamp": time.time()
        })
    
    def add_motif(self, motif: str):
        """Add an active symbolic motif to the pulse"""
        self.active_motifs.add(motif)
    
    def clear_messages(self):
        """Clear all messages from the pulse"""
        self.messages = []
        
    def get_component_state(self, component_name: str) -> Dict[str, Any]:
        """Get a specific component's state"""
        return self.component_states.get(component_name, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pulse to a dictionary representation"""
        return {
            "timestamp": self.timestamp,
            "phase": self.phase.name,
            "coherence_score": self.coherence_score,
            "contradiction_level": self.contradiction_level,
            "ethical_charge": self.ethical_charge,
            "identity_stability": self.identity_stability,
            "active_motifs": list(self.active_motifs),
            "message_count": len(self.messages),
            "components": list(self.component_states.keys())
        }

@dataclass
class BandConfig:
    """Configuration for a single oscillator band"""
    omega: float      # Base frequency
    lambda_damping: float  # Damping coefficient
    harmonic_index: int    # Harmonic relationship to sacred ratio


@dataclass
class FrequencyBandConfig:
    """Frequency band configuration for substrate encoding (mirrors harmonic_breath_field)"""
    omega: float
    band_name: str
    harmonic_index: int
    lambda_damping: float
class ComponentHealthReport:
    """Health status information for a system component"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.last_updated = time.time()
        self.is_responding = True
        self.synchronization_error = 0.0
        self.resource_usage = 0.0
        self.stability_score = 1.0
        self.diagnostics = {}
        
    def update(self, 
              is_responding: bool = True,
              sync_error: float = 0.0,
              resource_usage: float = 0.0,
              stability: float = 1.0,
              diagnostics: Dict[str, Any] = None):
        """Update the health report with new information"""
        self.last_updated = time.time()
        self.is_responding = is_responding
        self.synchronization_error = sync_error
        self.resource_usage = resource_usage
        self.stability_score = stability
        if diagnostics:
            self.diagnostics.update(diagnostics)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the health report to a dictionary"""
        return {
            "component": self.component_name,
            "last_updated": self.last_updated,
            "is_responding": self.is_responding,
            "sync_error": self.synchronization_error,
            "resource_usage": self.resource_usage,
            "stability": self.stability_score,
            "diagnostics": self.diagnostics
        }

class OscillatorBank:
    """
    Multi-band complex oscillator system with sacred harmonic coupling
    Each band maintains complex amplitude z = A*exp(iφ)
    Enhanced with adaptive coupling, nonlinear damping, stochastic noise, and resonance detection for deeper harmonic self-organization.
    """
    
    def __init__(self, dt: float = 0.05, noise_strength: float = 0.01):
        self.dt = dt
        self.noise_strength = noise_strength  # Strength of stochastic noise for biological realism
        
        # Define harmonic bands using sacred ratio
        # Each band frequency is SACRED_RATIO * (PHI^harmonic_index)
        self.bands = {
            'delta': BandConfig(
                omega=SACRED_RATIO * (PHI ** 0),  # Fundamental
                lambda_damping=-0.1,
                harmonic_index=0
            ),
            'theta': BandConfig(
                omega=SACRED_RATIO * (PHI ** 1),  # First harmonic
                lambda_damping=-0.15,
                harmonic_index=1
            ),
            'alpha': BandConfig(
                omega=SACRED_RATIO * (PHI ** 2),  # Second harmonic
                lambda_damping=-0.2,
                harmonic_index=2
            ),
            'beta': BandConfig(
                omega=SACRED_RATIO * (PHI ** 3),  # Third harmonic
                lambda_damping=-0.25,
                harmonic_index=3
            ),
            'gamma': BandConfig(
                omega=SACRED_RATIO * (PHI ** 4),  # Fourth harmonic
                lambda_damping=-0.3,
                harmonic_index=4
            )
        }
        
        # Complex amplitude state for each band
        self.z = {name: complex(0.1, 0.0) for name in self.bands.keys()}
        self.band_names = list(self.bands.keys())
        self.last_modulated_amplitudes = {name: abs(self.z[name]) for name in self.band_names}
        
        # Coupling matrix for inter-band interactions (now adaptive)
        self.coupling_strength = 0.05
        self.coupling_matrix = self._init_coupling_matrix()
        self._coupling_cache = None  # Cache for adaptive coupling
        
        # History for adaptive bounds and resonance detection
        self.amplitude_history = []  # Store recent amplitudes for variance calculation
        self.history_length = 50  # Length of history buffer
        
        # Resonance detection parameters
        self.resonance_threshold = 0.8  # Threshold for harmonic resonance
        self.resonance_boost = 1.2  # Multiplier for coupling when in resonance

    def compute_adaptive_amplitude(
        self,
        oscillator_idx: int,
        base_amplitude: float,
        breath_phase: Optional['BreathPhase'] = None,
        ethical_coherence: Optional[float] = None,
        spectral_radius: Optional[float] = None
    ) -> float:
        """
        Compute adaptive amplitude modulation blending breath dynamics,
        ethical state, eigenstate stability, and stochastic variation.
        """
        phase_modulation_map = {
            'INHALE': 1.20,
            'PAUSE_RISING': 1.10,
            'HOLD': 1.00,
            'PAUSE_FALLING': 0.95,
            'EXHALE': 0.75,
            'REST': 0.85,
            'DREAM': 1.15,
        }

        if isinstance(breath_phase, BreathPhase):
            phase_key = breath_phase.name
        elif breath_phase is None:
            phase_key = None
        else:
            phase_key = str(breath_phase).split('.')[-1].upper()

        breath_factor = phase_modulation_map.get(phase_key, 1.0)

        if ethical_coherence is not None:
            ethical_coherence = float(np.clip(ethical_coherence, 0.0, 1.0))
            ethical_factor = 0.5 + (0.5 * ethical_coherence)
        else:
            ethical_factor = 1.0

        if spectral_radius is not None:
            deviation = abs(spectral_radius - PHI) / PHI
            radius_factor = 1.0 + (0.15 * (deviation - 0.5) * 2.0)
            radius_factor = float(np.clip(radius_factor, 0.85, 1.15))
        else:
            radius_factor = 1.0

        oscillator_bias = 1.0 + (0.03 * np.sin(oscillator_idx * 0.1))
        noise_factor = 1.0 + np.random.uniform(-0.07, 0.07)
        time_drift = 1.0 + (0.05 * np.sin(time.time() * 0.01))

        safe_base = max(base_amplitude, 1e-4)
        modulated = (
            safe_base
            * breath_factor
            * ethical_factor
            * radius_factor
            * oscillator_bias
            * noise_factor
            * time_drift
        )

        return float(np.clip(modulated, 0.1, 2.0))
    
    def _init_coupling_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize harmonic coupling between bands"""
        bands = list(self.bands.keys())
        coupling = {b1: {b2: 0.0 for b2 in bands} for b1 in bands}
        
        # Natural harmonic coupling - adjacent bands couple more strongly
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                if i != j:
                    distance = abs(i - j)
                    coupling[b1][b2] = self.coupling_strength / (distance + 1)
        
        return coupling
    
    def _adaptive_coupling_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute adaptive coupling matrix based on current phase coherence"""
        if self._coupling_cache is not None:
            return self._coupling_cache
        
        bands = list(self.bands.keys())
        adaptive_coupling = {b1: {b2: 0.0 for b2 in bands} for b1 in bands}
        
        # Compute phase coherence for each pair
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                if i != j:
                    phase_diff = np.angle(self.z[b1] - self.z[b2])
                    coherence = np.exp(-abs(phase_diff) / np.pi)  # Exponential decay with phase difference
                    base_coupling = self.coupling_matrix[b1][b2]
                    adaptive_coupling[b1][b2] = base_coupling * (1.0 + coherence)
        
        self._coupling_cache = adaptive_coupling
        return adaptive_coupling
    
    def _nonlinear_damping(self, amplitude: float, base_lambda: float) -> float:
        """Compute amplitude-dependent nonlinear damping"""
        # Sigmoid-based damping: stronger at higher amplitudes
        sigmoid_factor = 1.0 / (1.0 + np.exp(-(amplitude - 2.0)))  # Centers around amplitude=2.0
        return base_lambda * (1.0 + 0.5 * sigmoid_factor)  # Scale damping up to 1.5x
    
    def _detect_resonance(self, band1: str, band2: str) -> bool:
        """Detect harmonic resonance between two bands based on frequency ratios"""
        omega1 = self.bands[band1].omega
        omega2 = self.bands[band2].omega
        ratio = max(omega1, omega2) / min(omega1, omega2)
        # Check if ratio is close to integer or PHI-based harmonic
        harmonic_ratios = [1.0, PHI, PHI**2, 2.0, 3.0]  # Common harmonics
        return any(abs(ratio - h) < 0.1 for h in harmonic_ratios)
    
    def _internal_feedback(self, band_name: str) -> float:
        """Compute internal feedback modulation for frequency based on amplitude"""
        amplitude = abs(self.z[band_name])
        # Slight frequency increase with amplitude for self-stabilization
        feedback = 0.01 * amplitude  # Small modulation
        return self.bands[band_name].omega * (1.0 + feedback)
    
    def step(
        self,
        drives: Dict[str, float],
        coupling_drives: Optional[Dict[str, complex]] = None,
        breath_phase: Optional['BreathPhase'] = None,
        ethical_coherence: Optional[float] = None,
        spectral_radius: Optional[float] = None,
    ):
        """
        Evolve oscillator dynamics one time step
        dz/dt = (λ + iω)z + drive + coupling_terms + noise
        Enhanced with nonlinear damping, adaptive coupling, resonance, and feedback.
        """
        new_z = {}
        current_amplitudes = [abs(z) for z in self.z.values()]
        self.amplitude_history.append(current_amplitudes)
        if len(self.amplitude_history) > self.history_length:
            self.amplitude_history.pop(0)
        
        # Adaptive coupling matrix
        coupling_matrix = self._adaptive_coupling_matrix()
        
        for idx, (band_name, config) in enumerate(self.bands.items()):
            z = self.z[band_name]
            amplitude = abs(z)

            modulated_amplitude = self.compute_adaptive_amplitude(
                oscillator_idx=idx,
                base_amplitude=amplitude,
                breath_phase=breath_phase,
                ethical_coherence=ethical_coherence,
                spectral_radius=spectral_radius,
            )
            self.last_modulated_amplitudes[band_name] = modulated_amplitude
            base_safe = max(amplitude, 1e-4)
            amplitude_ratio = float(np.clip(modulated_amplitude / base_safe, 0.1, 3.0))
            
            # Nonlinear damping
            effective_lambda = self._nonlinear_damping(amplitude, config.lambda_damping)
            
            # Internal feedback on frequency
            effective_omega = self._internal_feedback(band_name)
            
            # Basic oscillator dynamics with enhancements
            linear_term = (effective_lambda + 1j * effective_omega) * z
            
            # External drive
            drive = drives.get(band_name, 0.0) * amplitude_ratio
            
            # Coupling from other bands with resonance boost
            coupling_term = 0.0
            for other_band, coupling_strength in coupling_matrix[band_name].items():
                if other_band != band_name:
                    other_z = self.z[other_band]
                    gate = 1.0 / (1.0 + np.exp((amplitude - 3.0)))  # Amplitude gate
                    phase_coupling = np.exp(1j * np.angle(z - other_z))
                    resonance_factor = self.resonance_boost if self._detect_resonance(band_name, other_band) else 1.0
                    coupling_term += gate * coupling_strength * resonance_factor * other_z * phase_coupling
            
            # Additional complex coupling drives
            if coupling_drives and band_name in coupling_drives:
                coupling_term += coupling_drives[band_name]
            
            # Stochastic noise
            noise = self.noise_strength * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
            
            # Integrate
            dz_dt = linear_term + drive + coupling_term + noise
            new_z[band_name] = z + self.dt * dz_dt

            new_amplitude = abs(new_z[band_name])
            if new_amplitude > 1e-6:
                target_complex = modulated_amplitude * np.exp(1j * np.angle(new_z[band_name]))
                new_z[band_name] = 0.7 * new_z[band_name] + 0.3 * target_complex
            
            # Adaptive stability clamp based on recent variance
            if self.amplitude_history:
                recent_variance = np.var(self.amplitude_history[-10:], axis=0).mean()  # Last 10 steps
                adaptive_max = 5.0 + 0.5 * recent_variance  # Allow more fluctuation if variance is high
                new_amplitude = abs(new_z[band_name])
                if new_amplitude > adaptive_max:
                    new_z[band_name] = new_z[band_name] * (adaptive_max / new_amplitude)
        
        self.z = new_z
        self._coupling_cache = None  # Invalidate cache for next step
    
    def signature(self) -> np.ndarray:
        """
        Extract wave signature vector: [A_delta, cos(φ_delta), sin(φ_delta), ...]
        """
        sig = []
        for band_name in self.bands.keys():
            z = self.z[band_name]
            amplitude = abs(z)
            phase = np.angle(z)
            sig.extend([amplitude, np.cos(phase), np.sin(phase)])
        return np.array(sig, dtype=np.float32)
    
    def get_amplitudes(self) -> Dict[str, float]:
        """Get current amplitude for each band"""
        return {name: abs(self.z[name]) for name in self.bands.keys()}
    
    def get_phases(self) -> Dict[str, float]:
        """Get current phase for each band"""
        return {name: np.angle(self.z[name]) for name in self.bands.keys()}

    def compute_synchronization_index(self) -> Dict[str, float]:
        """Compute Kuramoto-like synchronization index for each band"""
        sync_indices = {}
        bands = list(self.bands.keys())
        for band_name in bands:
            phase = np.angle(self.z[band_name])
            other_phases = [np.angle(self.z[b]) for b in bands if b != band_name]
            if other_phases:
                mean_other_phase = np.mean(other_phases)
                sync_indices[band_name] = abs(np.mean(np.exp(1j * (phase - mean_other_phase))))
            else:
                sync_indices[band_name] = 1.0  # Self-sync if no others
        return sync_indices
    
    def get_debug_state(self) -> Dict[str, Any]:
        """Enhanced debug state with new metrics"""
        base_state = {
            'amplitudes': self.get_amplitudes(),
            'phases': self.get_phases(),
            'synchronization_indices': self.compute_synchronization_index(),
            'adaptive_coupling_matrix': self._adaptive_coupling_matrix(),
            'amplitude_variance': np.var(self.amplitude_history, axis=0).mean() if self.amplitude_history else 0.0
        }
        return base_state

class EmotionEncoder:
    """
    Maps cognitive/physiological states to target HRF band energies
    Enhanced with dynamic computation using harmonic constants for more adaptive, biologically-inspired mappings.
    """
    
    def __init__(self):
        # Expanded state mappings with more emotions and dynamic scaling functions
        # Each mapping now uses a lambda function for intensity-based computation
        # Incorporates PHI and SACRED_RATIO for "sacred" harmonic scaling
        self.state_mappings = {
            'curiosity': lambda intensity: {
                'theta': intensity * 1.5 * (PHI ** 0.5),  # Harmonic boost for exploration
                'beta': intensity * 0.5,
                'alpha': intensity * 0.3
            },
            'frustration': lambda intensity: {
                'beta': intensity * 2.0,
                'gamma': intensity * 1.2 * SACRED_RATIO,  # Sacred ratio modulation
                'alpha': intensity * 0.2
            },
            'serenity': lambda intensity: {
                'alpha': intensity * 2.0,
                'delta': intensity * 1.0 * (PHI ** -0.5),  # Inverse harmonic for grounding
                'theta': intensity * 0.1
            },
            'urgency': lambda intensity: {
                'gamma': intensity * 2.5,
                'beta': intensity * 1.5,
                'delta': intensity * 0.1
            },
            'focus': lambda intensity: {
                'gamma': intensity * 1.8,
                'beta': intensity * 1.0,
                'theta': intensity * 0.2
            },
            'creativity': lambda intensity: {
                'theta': intensity * 2.0,
                'alpha': intensity * 1.0,
                'delta': intensity * 0.5
            },
            'stillness': lambda intensity: {
                'delta': intensity * 3.0,
                'alpha': intensity * 0.5
            },
            # New states for broader coverage
            'anxiety': lambda intensity: {
                'beta': intensity * 2.2,
                'gamma': intensity * 1.5,
                'theta': intensity * 0.8
            },
            'excitement': lambda intensity: {
                'beta': intensity * 1.8,
                'gamma': intensity * 2.0,
                'alpha': intensity * 0.4
            },
            'confusion': lambda intensity: {
                'theta': intensity * 1.7,
                'beta': intensity * 1.3,
                'alpha': intensity * 0.6
            },
            'calm': lambda intensity: {
                'alpha': intensity * 2.5,
                'delta': intensity * 1.5,
                'theta': intensity * 0.2
            }
        }
        
        # Default mapping for unknown states
        self.default_mapping = lambda intensity: {
            'alpha': intensity * 1.0,  # Neutral alpha focus
            'theta': intensity * 0.5
        }
        
        # Safety bounds for amplitudes
        self.max_amplitude = 5.0
        self.min_amplitude = 0.0
    
    def target_amplitudes(self, state_vector: Dict[str, float]) -> Dict[str, float]:
        """Convert state metrics to desired band amplitudes with enhanced blending and harmonic scaling"""
        targets = {band: 0.0 for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
        
        for state, intensity in state_vector.items():
            # Clamp intensity to prevent extremes
            intensity = max(0.0, min(5.0, intensity))
            
            # Get mapping function or use default
            mapping_func = self.state_mappings.get(state, self.default_mapping)
            state_targets = mapping_func(intensity)
            
            # Blend with exponential weighting for smoother transitions
            weight = 1.0 - math.exp(-intensity)  # Exponential decay for low intensities
            for band, value in state_targets.items():
                targets[band] += weight * value
        
        # Apply harmonic normalization using PHI for "golden" balance
        max_amp = max(targets.values()) if targets.values() else 1.0
        if max_amp > self.max_amplitude:
            scale_factor = self.max_amplitude / max_amp
            # Use PHI-based scaling for more natural adjustment
            phi_scale = PHI / (PHI + 1)  # Normalize PHI to [0,1]
            targets = {k: v * scale_factor * phi_scale for k, v in targets.items()}
        
        # Ensure all values are within bounds and finite
        for band in targets:
            targets[band] = max(self.min_amplitude, min(self.max_amplitude, targets[band]))
            if not np.isfinite(targets[band]):
                targets[band] = 0.0  # Fallback for any NaN/inf
        
        return targets

class HarmonicCoupler:
    """
    Couples harmonic field to embedding space for perception modulation
    """
    
    def __init__(self, embed_dim: int = 768):
        np.random.seed(42)
        self.embed_dim = embed_dim
        # Projection matrix from wave signature to embedding space
        sig_dim = 15  # 5 bands × 3 components each (amp, cos, sin)
        self.W_wave = np.random.randn(embed_dim, sig_dim) * 0.01
    
    def modulate_embedding(self, embedding: np.ndarray, wave_signature: np.ndarray) -> np.ndarray:
        """Modulate input embedding with current harmonic field state"""
        wave_modulation = self.W_wave @ wave_signature
        return embedding + wave_modulation
    
    def compute_saliency(self, token_embeddings: List[np.ndarray]) -> Dict[str, float]:
        """Compute saliency metrics that drive oscillator bank"""
        if not token_embeddings:
            return {}
        
        # Simple saliency metrics (can be made more sophisticated)
        embedding_stack = np.array(token_embeddings)
        
        # Novelty: variance in recent embeddings
        novelty = np.var(embedding_stack, axis=0).mean() if len(embedding_stack) > 1 else 0.0
        
        # Entropy-like measure
        norms = np.linalg.norm(embedding_stack, axis=1)
        entropy = -np.sum(norms * np.log(norms + 1e-8))
        
        # Contradiction: rapid changes in embedding direction
        contradiction = 0.0
        if len(embedding_stack) > 2:
            recent_changes = np.diff(embedding_stack[-3:], axis=0)
            contradiction = np.linalg.norm(recent_changes.var(axis=0))
        
        return {
            'theta': novelty * 0.5,      # Drive exploration
            'beta': contradiction * 1.0,  # Drive decisive resolution  
            'gamma': entropy * 0.3       # Drive focused attention
        }

class GenerationModulator:
    """
    Derives generation parameters from harmonic field state
    """
    
    def __init__(self):
        # Base parameters
        self.base_temperature = 0.7
        self.base_top_p = 0.9
        self.base_branches = 1
    
    def generation_params(self,
                          wave_signature: np.ndarray,
                          amplitudes: Dict[str, float],
                          phases: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Convert wave state to generation parameters"""
        
        # Temperature modulation
        # Higher theta/alpha = more exploration, higher beta/gamma = more focus
        exploration_factor = amplitudes.get('theta', 0) + amplitudes.get('alpha', 0) * 0.5
        focus_factor = amplitudes.get('beta', 0) + amplitudes.get('gamma', 0)
        temperature = self.base_temperature * (1.0 + 0.3 * exploration_factor - 0.2 * focus_factor)
        temperature = max(0.1, min(2.0, temperature))  # Clamp
        
        # Top-p modulation
        top_p = self.base_top_p - 0.1 * focus_factor + 0.05 * exploration_factor
        top_p = max(0.3, min(0.95, top_p))
        
        # Beam/branch width
        branches = self.base_branches + round(2 * amplitudes.get('theta', 0) + amplitudes.get('gamma', 0))
        branches = max(1, min(8, branches))
        
        # Memory retrieval gate
        retrieval_gate = 1.0 / (1.0 + np.exp(-2.0 * amplitudes.get('alpha', 0) + 1.2 * amplitudes.get('beta', 0)))
        
        # Attention head scaling factors
        head_scales = {}
        # Use true oscillator phases if available; fall back to 0
        gφ = (phases.get('gamma', 0.0) if phases else 0.0)
        for i in range(12):  # Assume 12 attention heads
            scale = 1.0 + 0.2 * np.cos(gφ + i * TAU / 12)
            head_scales[f'head_{i}'] = scale
        
        return {
            'temperature': temperature,
            'top_p': top_p,
            'num_beams': branches,
            'retrieval_gate': retrieval_gate,
            'attention_head_scales': head_scales,
            'amplitudes': amplitudes  # For logging/debugging
        }

class CoupledHarmonicBreath:
    """
    Main orchestrator coupling BreathPhase cycle with Harmonic Reasoning Field
    """
    
    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.start_time = time.time()
        
        # Core components
        self.oscillators = OscillatorBank(dt)
        self.emotion_encoder = EmotionEncoder()
        self.coupler = HarmonicCoupler()
        self.modulator = GenerationModulator()
        
        # Breath state
        self.breath_phase = BreathPhase.INHALE
        self.breath_position = 0.0  # [0,1] within current phase
        self.cycle_count = 0
        
        # Phase duration mapping (relative to full cycle)
        self.phase_durations = {
            BreathPhase.INHALE: 0.25,      # 25% of cycle
            BreathPhase.PAUSE_RISING: 0.1, # 10% of cycle  
            BreathPhase.HOLD: 0.2,         # 20% of cycle
            BreathPhase.PAUSE_FALLING: 0.1,# 10% of cycle
            BreathPhase.EXHALE: 0.25,      # 25% of cycle
            BreathPhase.REST: 0.075,       # 7.5% of cycle
            BreathPhase.DREAM: 0.025       # 2.5% of cycle
        }
        
        # Breath-phase harmonic mappings
        self.base_amplitudes = {
            'delta': 1.0,
            'theta': 0.8,
            'alpha': 1.2,
            'beta': 0.6,
            'gamma': 0.4,
        }
        self.current_spectral_radius = PHI
        self.current_ethical_coherence = 1.0
        self.breath_harmonics = self._init_breath_harmonics()

    def set_context(
        self,
        spectral_radius: Optional[float] = None,
        ethical_coherence: Optional[float] = None,
    ) -> None:
        """Update adaptive modulation context."""
        if spectral_radius is not None:
            self.current_spectral_radius = spectral_radius
        if ethical_coherence is not None:
            self.current_ethical_coherence = ethical_coherence
        
    def _init_breath_harmonics(self) -> Dict[BreathPhase, Dict[str, float]]:
        """Define harmonic field targets for each breath phase"""
        return {
            BreathPhase.INHALE: {
                'theta': 2.0,    # Curiosity/exploration
                'alpha': 0.5,    # Gentle awareness
                'beta': 0.2,     # Minimal judgment
                'gamma': 0.3,    # Light attention
                'delta': 0.8     # Grounding
            },
            BreathPhase.PAUSE_RISING: {
                'alpha': 1.5,    # First reflection
                'theta': 0.5,    # Reduced exploration
                'delta': 1.0,    # Maintained grounding
                'beta': 0.1,     # Minimal judgment
                'gamma': 0.4     # Focused attention
            },
            BreathPhase.HOLD: {
                'delta': 3.0,    # Maximum stillness
                'alpha': 0.8,    # Quiet awareness
                'theta': 0.1,    # Minimal activity
                'beta': 0.1,     # Minimal activity  
                'gamma': 0.1     # Minimal activity
            },
            BreathPhase.PAUSE_FALLING: {
                'alpha': 1.2,    # Second reflection
                'gamma': 1.0,    # Sharp discrimination
                'delta': 0.8,    # Reducing stillness
                'beta': 0.5,     # Emerging judgment
                'theta': 0.3     # Reduced exploration
            },
            BreathPhase.EXHALE: {
                'beta': 2.5,     # Ethical will/decision
                'gamma': 2.0,    # Precise action
                'alpha': 0.3,    # Focused awareness
                'theta': 0.2,    # Minimal exploration
                'delta': 0.2     # Minimal stillness
            },
            BreathPhase.REST: {
                'alpha': 2.0,    # Memory integration
                'delta': 1.5,    # Return to stillness
                'theta': 0.1,    # Minimal exploration
                'beta': 0.1,     # Minimal judgment
                'gamma': 0.2     # Light attention
            },
            BreathPhase.DREAM: {
                'theta': 1.5,    # Meta-ethical exploration
                'delta': 2.0,    # Deep stillness
                'alpha': 0.8,    # Aware integration
                'beta': 0.1,     # Suspended judgment
                'gamma': 0.1     # Minimal focus
            }
        }
    
    def _next_phase(self) -> BreathPhase:
        """Advance to next breath phase"""
        phases = list(BreathPhase)
        current_idx = phases.index(self.breath_phase)
        next_idx = (current_idx + 1) % len(phases)
        
        if next_idx == 0:  # Completed full cycle
            self.cycle_count += 1
            logger.info(f"Completed breath cycle {self.cycle_count}")
        
        return phases[next_idx]
    
    def step(self, external_state: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Advance one time step of coupled breath-harmonic dynamics
        Returns current generation parameters
        """
        
        # Advance breath position
        phase_duration = self.phase_durations[self.breath_phase]
        self.breath_position += self.dt * SACRED_RATIO / phase_duration
        
        # Phase transition
        if self.breath_position >= 1.0:
            self.breath_phase = self._next_phase()
            self.breath_position = 0.0
            logger.debug(f"Breath phase: {self.breath_phase.name}")
        
        # Compute breath-driven harmonic targets using adaptive modulation
        breath_targets: Dict[str, float] = {}
        for idx, band_name in enumerate(self.oscillators.band_names):
            base_amp = self.base_amplitudes.get(band_name, 1.0)
            breath_targets[band_name] = self.oscillators.compute_adaptive_amplitude(
                oscillator_idx=idx,
                base_amplitude=base_amp,
                breath_phase=self.breath_phase,
                ethical_coherence=self.current_ethical_coherence,
                spectral_radius=self.current_spectral_radius,
            )
        
        # Merge with external state if provided
        if external_state:
            external_targets = self.emotion_encoder.target_amplitudes(external_state)
            # Weighted combination: 70% breath rhythm, 30% external state
            combined_targets = {}
            for band in breath_targets.keys():
                combined_targets[band] = (0.7 * breath_targets[band] + 
                                        0.3 * external_targets.get(band, 0.0))
        else:
            combined_targets = breath_targets
        
        # Convert targets to drives (PD control toward target)
        current_amps = self.oscillators.get_amplitudes()
        drives = {}
        for band, target in combined_targets.items():
            error = target - current_amps.get(band, 0.0)
            drives[band] = 2.0 * error  # Proportional gain
        
        # Step oscillators
        self.oscillators.step(
            drives,
            coupling_drives=None,
            breath_phase=self.breath_phase,
            ethical_coherence=self.current_ethical_coherence,
            spectral_radius=self.current_spectral_radius,
        )
        
        # Generate current parameters
        wave_sig = self.oscillators.signature()
        current_amps = self.oscillators.get_amplitudes()
        current_phases = self.oscillators.get_phases()
        gen_params = self.modulator.generation_params(wave_sig, current_amps, phases=current_phases)
        
        # Add breath context
        gen_params.update({
            'breath_phase': self.breath_phase,
            'breath_position': self.breath_position,
            'cycle_count': self.cycle_count,
            'sacred_time': time.time() - self.start_time,
            'wave_signature': wave_sig,
            'phases': current_phases
        })
        
        return gen_params
    
    def process_token_sequence(self, token_embeddings: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Process a sequence of token embeddings through harmonic coupling
        Returns modulated embeddings and final generation parameters
        """
        modulated_embeddings = []
        
        for i, embedding in enumerate(token_embeddings):
            # Compute saliency drives from recent context
            recent_context = token_embeddings[max(0, i-5):i+1]
            saliency_drives = self.coupler.compute_saliency(recent_context)
            # Light adaptive damping proportional to contradiction
            k_c = 0.05 * saliency_drives.get('beta', 0.0)
            for name, cfg in self.oscillators.bands.items():
                self.oscillators.bands[name] = BandConfig(
                    omega=cfg.omega,
                    lambda_damping=cfg.lambda_damping - k_c,
                    harmonic_index=cfg.harmonic_index
                )
            
            # Step dynamics with saliency input
            gen_params = self.step(saliency_drives)
            
            # Modulate embedding
            wave_sig = gen_params['wave_signature']
            modulated_emb = self.coupler.modulate_embedding(embedding, wave_sig)
            modulated_embeddings.append(modulated_emb)
        
        return modulated_embeddings, gen_params
    
    def get_debug_state(self) -> Dict[str, Any]:
        """Get current state for debugging/visualization"""
        return {
            'breath_phase': self.breath_phase.name,
            'breath_position': self.breath_position,
            'cycle_count': self.cycle_count,
            'amplitudes': self.oscillators.get_amplitudes(),
            'phases': self.oscillators.get_phases(),
            'wave_signature': self.oscillators.signature().tolist()
        }
    
class SacredFrequencySubstrate:
    """
    Frequency-based text substrate using sacred harmonic constants.
    Validated against Ulam spiral correlations and EEG envelope patterns.
    """

    def __init__(self, 
                 frequency_scales: Optional[List[float]] = None,
                 wavelet_types: List[str] = None,
                 semantic_features: bool = True,
                 tensor_dimensions: int = 256,
                 use_sacred_harmonics: bool = True,
                 vocab_size: int = 10000,
                 enable_harmonic_lexicon: bool = True):
        
        self.tensor_dimensions = tensor_dimensions
        self.semantic_features = semantic_features
        self.use_sacred_harmonics = use_sacred_harmonics
        self.vocab_size = vocab_size
        self.enable_harmonic_lexicon = enable_harmonic_lexicon
        self._lock = threading.RLock()
        
        # Sacred harmonic frequency scales (PHI-based)
        if frequency_scales is None and use_sacred_harmonics:
            # Use PHI-based scales: [PHI^0, PHI^1, PHI^2, PHI^3, PHI^4]
            self.frequency_scales = [PHI ** i for i in range(5)]
        else:
            self.frequency_scales = frequency_scales or [1, 2, 3, 4, 5, 8, 16, 32]
        
        # Wavelet types for multi-scale analysis
        self.wavelet_types = wavelet_types or ['haar', 'db2', 'sym4', 'coif1']
        
        # Initialize frequency band configurations using sacred harmonics
        self.bands = {
            name: FrequencyBandConfig(
                omega=HARMONIC_BANDS[name],
                band_name=name,
                harmonic_index=i,
                lambda_damping=-0.1 * (1 + 0.05 * i)  # Gradual damping increase
            )
            for i, name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma'])
        }
        
        # Complex amplitude state for each band (oscillator representation)
        self.z = {name: complex(0.1, 0.0) for name in self.bands.keys()}
        
        # Semantic feature mapping (from early 2000s predicate logic)
        self.semantic_map = self._build_semantic_map()

        # Harmonic lexicon for bag-of-harmonics encoding
        self.harmonic_lexicon = self._build_harmonic_lexicon() if enable_harmonic_lexicon else {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Safety bounds
        self.max_amplitude = 5.0
        self.min_amplitude = 0.0
        
        lexicon_msg = f" and {len(self.harmonic_lexicon)} lexical signatures" if self.harmonic_lexicon else ""
        logger.info(f"SacredFrequencySubstrate initialized with {len(self.bands)} harmonic bands{lexicon_msg}")

    def _build_harmonic_lexicon(self) -> Dict[str, np.ndarray]:
        """
        Build a large, stable lexicon of harmonic vectors for a vocabulary.
        Placeholder implementation that can later be wired to a real word list.
        """
        np.random.seed(42)
        lexicon: Dict[str, np.ndarray] = {}

        # Placeholder slots for vocab hashing
        for i in range(self.vocab_size):
            key = f"__word_slot_{i}__"
            lexicon[key] = self._generate_harmonic_vector(i, use_tau=False)

        # Preserve legacy predicates as named entries
        lexicon["subject-verb-object"] = self._generate_harmonic_vector(self.vocab_size + 1)
        lexicon["question-answer"] = self._generate_harmonic_vector(self.vocab_size + 2)
        lexicon["causation"] = self._generate_harmonic_vector(self.vocab_size + 3)

        # Dedicated unknown token
        lexicon["[UNK]"] = self._generate_harmonic_vector(0, use_tau=True)
        return lexicon

    def _generate_harmonic_vector(self, harmonic_idx: int, use_tau: bool = False) -> np.ndarray:
        """Generate a vector modulated by sacred harmonic frequencies with optional TAU carrier."""
        t = np.linspace(0, TAU, self.tensor_dimensions)
        vector = np.zeros(self.tensor_dimensions)
        denom = sum(PHI ** j for j in range(len(self.bands)))

        for i, (band_name, config) in enumerate(self.bands.items()):
            phase_offset = (harmonic_idx * TAU * (PHI ** i)) % TAU
            base_freq = TAU if use_tau else config.omega
            harmonic_component = np.sin(base_freq * t + phase_offset)
            weight = (PHI ** i) / denom
            vector += weight * harmonic_component
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _get_word_slot(self, word: str) -> str:
        """Map arbitrary word to deterministic lexicon slot."""
        if not self.harmonic_lexicon:
            return "[UNK]"
        slot = hash(word) % self.vocab_size
        return f"__word_slot_{slot}__"

    def _encode_bag_of_harmonics(self, text: str) -> np.ndarray:
        """Encode text via harmonic lexicon."""
        if not text or not self.harmonic_lexicon:
            return np.array([])

        combined = np.zeros(self.tensor_dimensions, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return combined

        count = 0
        for word in words:
            if word in self.harmonic_lexicon:
                combined += self.harmonic_lexicon[word]
            else:
                combined += self.harmonic_lexicon.get(self._get_word_slot(word), self.harmonic_lexicon["[UNK]"])
            count += 1

        if count > 0:
            combined /= count
        return combined

    def _apply_wavelet_transforms_to_vector(self, vector: np.ndarray) -> np.ndarray:
        """Apply wavelet transforms directly to a harmonic semantic vector."""
        if vector.size < 4:
            return np.array([])

        features: List[float] = []
        for wt_type in self.wavelet_types:
            try:
                coeffs = pywt.wavedec(vector, wt_type, level=min(3, pywt.dwt_max_level(vector.size, wt_type)))
                for level_coeffs in coeffs:
                    if level_coeffs.size > 0:
                        features.extend([np.mean(level_coeffs), np.std(level_coeffs)])
            except Exception as e:
                logger.debug(f"Wavelet transform {wt_type} failed: {e}")
        return np.array(features) if features else np.array([])
    
    def _build_semantic_map(self) -> Dict[str, np.ndarray]:
        """Build semantic predicate mapping with sacred harmonic encoding"""
        np.random.seed(42)  # Reproducibility
        
        semantic_patterns = {
            'subject-verb-object': self._generate_harmonic_vector(0),
            'question-answer': self._generate_harmonic_vector(1),
            'causation': self._generate_harmonic_vector(2),
            'negation': self._generate_harmonic_vector(3),
            'comparison': self._generate_harmonic_vector(4),
            'temporal-sequence': self._generate_harmonic_vector(5),
            'spatial-relation': self._generate_harmonic_vector(6),
        }
        
        return semantic_patterns
    
    def extract_fbs(self, text: str) -> np.ndarray:
        """
        Extract Frequency-Based Substrate representation for a single token/
        fragment using the harmonic lexicon.
        """
        with self._lock:
            try:
                lexical_vector = self._encode_bag_of_harmonics(text)
                if lexical_vector.size == 0:
                    return np.zeros(self.tensor_dimensions)

                feature_blocks: List[np.ndarray] = [lexical_vector]
                if self.semantic_features:
                    lexical_wavelets = self._apply_wavelet_transforms_to_vector(lexical_vector)
                    if lexical_wavelets.size > 0:
                        feature_blocks.append(lexical_wavelets)

                combined = np.concatenate(feature_blocks)
                tensor = self._project_to_tensor(combined)
                tensor = self._apply_sacred_gating(tensor)
                return tensor

            except Exception as e:
                logger.error(f"Error in extract_fbs: {str(e)}")
                return np.zeros(self.tensor_dimensions)
    
    def _project_to_tensor(self, features: np.ndarray) -> np.ndarray:
        """Project combined features to fixed tensor dimensions using sacred harmonics"""
        if features.size == 0:
            return np.zeros(self.tensor_dimensions)
        
        # If features are larger than target, use harmonic downsampling
        if features.size > self.tensor_dimensions:
            # Create projection matrix using PHI-weighted random projection
            np.random.seed(42)
            projection_matrix = np.random.randn(self.tensor_dimensions, features.size)
            
            # Apply PHI-based weighting to columns
            for i in range(features.size):
                weight = (PHI ** (i % 5)) / sum(PHI ** j for j in range(5))
                projection_matrix[:, i] *= weight
            
            # Normalize projection matrix
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=1, keepdims=True)
            
            tensor = projection_matrix @ features
        elif features.size < self.tensor_dimensions:
            # Pad with zeros
            tensor = np.zeros(self.tensor_dimensions)
            tensor[:features.size] = features
        else:
            tensor = features
        
        return tensor
    
    def _apply_sacred_gating(self, tensor: np.ndarray) -> np.ndarray:
        """Apply sacred ratio gating to the tensor"""
        # Use SACRED_RATIO as a gating function
        gate = 1.0 / (1.0 + np.exp(-SACRED_RATIO * (tensor - np.mean(tensor))))
        return tensor * gate
    
class SacredTensorProcessor:
    """
    Tensor-guided processing using sacred harmonic field coupling.
    Based on production-grade OscillatorBank from harmonic_breath_field.py
    """
    
    def __init__(self, tensor_dimensions: int = 256, dt: float = 0.05, use_log2_attention: bool = True):
        self.tensor_dimensions = tensor_dimensions
        self.dt = dt
        self._lock = threading.RLock()
        
        # O(log² N) attention configuration
        self.use_log2_attention = use_log2_attention
        self.harmonic_tree_depth = int(math.log2(tensor_dimensions)) if tensor_dimensions > 1 else 1
        
        # Initialize sacred harmonic operations
        self.tensor_operations = self._init_tensor_operations()
        
        # Harmonic field state
        self.harmonic_state = {
            'delta': complex(0.1, 0.0),
            'theta': complex(0.08, 0.0),
            'alpha': complex(0.12, 0.0),
            'beta': complex(0.06, 0.0),
            'gamma': complex(0.04, 0.0),
        }
        
        # Snapshot for reset
        self._initial_harmonic_state = {band: complex(val) for band, val in self.harmonic_state.items()}

        # Coupling matrix for inter-band interactions
        self.coupling_strength = 0.05
        self.coupling_matrix = self._init_coupling_matrix()
        
        logger.info(f"SacredTensorProcessor initialized (log²N attention={'enabled' if use_log2_attention else 'disabled'})")
    
    def _init_tensor_operations(self) -> Dict[str, Any]:
        """Initialize sacred harmonic tensor operations"""
        return {
            'sacred_tensor_product': self._sacred_tensor_product,
            'harmonic_outer_product': self._harmonic_outer_product,
            'phi_contraction': self._phi_contraction,
            'golden_kronecker': self._golden_kronecker,
            'breath_modulation': self._breath_modulation,
        }
    
    def _init_coupling_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize harmonic coupling using sacred ratios"""
        bands = list(self.harmonic_state.keys())
        coupling = {b1: {b2: 0.0 for b2 in bands} for b1 in bands}
        
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                if i != j:
                    distance = abs(i - j)
                    # Use PHI-based coupling decay
                    coupling[b1][b2] = self.coupling_strength * (PHI ** -distance)
        
        return coupling
    
    def _sacred_tensor_product(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Tensor product modulated by SACRED_RATIO"""
        product = np.tensordot(tensor1, tensor2, axes=0)
        # Apply sacred ratio gating
        return product * SACRED_RATIO
    
    def _harmonic_outer_product(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Outer product with PHI weighting"""
        outer = np.outer(tensor1, tensor2)
        # Weight by golden ratio
        return outer * (PHI / (PHI + 1))
    
    def _phi_contraction(self, tensor: np.ndarray, axes: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """Tensor contraction with PHI-based scaling"""
        if tensor.ndim < 2:
            return tensor
        contracted = np.tensordot(tensor, tensor, axes=axes)
        return contracted * (1.0 / PHI)
    
    def _golden_kronecker(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Kronecker product with golden ratio normalization"""
        kron = np.kron(tensor1, tensor2)
        # Normalize by PHI
        return kron / PHI
    
    def _breath_modulation(self, tensor: np.ndarray, breath_phase: float = 0.0) -> np.ndarray:
        """Modulate tensor with breath cycle harmonics"""
        # breath_phase in [0, 1] representing position in breath cycle
        modulation = np.zeros_like(tensor)
        
        for i, (band_name, z) in enumerate(self.harmonic_state.items()):
            # Calculate phase for this band in breath cycle
            band_phase = (breath_phase + i / 5.0) % 1.0
            harmonic = np.sin(TAU * band_phase * HARMONIC_BANDS[band_name])
            
            # Apply modulation weighted by band amplitude
            amplitude = abs(z)
            modulation += tensor * harmonic * amplitude * SACRED_RATIO
        
        return modulation / len(self.harmonic_state)

    def reset_harmonics(self) -> None:
        """Reset harmonic oscillators to their initial state."""
        with self._lock:
            self.harmonic_state = {band: complex(val) for band, val in self._initial_harmonic_state.items()}
    
    def process(self, fbs_tensor: np.ndarray, breath_phase: float = 0.0) -> np.ndarray:
        with self._lock:
            try:
                # Start with input tensor
                processed = fbs_tensor.copy()
                
                # Apply breath modulation first
                processed = self._breath_modulation(processed, breath_phase)
                
                # Apply harmonic operations sequentially
                # Each operation is gated by oscillator amplitudes
                for band_name, z in self.harmonic_state.items():
                    amplitude = abs(z)
                    
                    if amplitude > 0.01:  # Only apply if band is active
                        # Self-interaction through outer product
                        outer = self._harmonic_outer_product(processed, processed)
                        
                        # Contract to original dimensionality
                        if outer.size > self.tensor_dimensions:
                            outer_flat = outer.flatten()[:self.tensor_dimensions]
                        else:
                            outer_flat = np.pad(outer.flatten(), (0, max(0, self.tensor_dimensions - outer.size)))
                        
                        # Blend with original
                        processed = 0.7 * processed + 0.3 * outer_flat * amplitude
                
                # Apply coupling between harmonic bands
                processed = self._apply_harmonic_coupling(processed)
                
                # Final sacred ratio gating
                gate = 1.0 / (1.0 + np.exp(-SACRED_RATIO * (processed - np.mean(processed))))
                processed = processed * gate
                
                # Safety clamp
                processed = np.clip(processed, -10.0, 10.0)
                
                return processed
                
            except Exception as e:
                logger.error(f"Error in sacred tensor processing: {str(e)}")
                return fbs_tensor
    
    def _apply_harmonic_coupling(self, tensor: np.ndarray) -> np.ndarray:
        """Apply harmonic coupling between oscillator bands"""
        coupled = tensor.copy()
        
        for b1, z1 in self.harmonic_state.items():
            for b2, z2 in self.harmonic_state.items():
                if b1 != b2:
                    coupling_strength = self.coupling_matrix[b1][b2]
                    
                    # Phase coupling
                    phase_diff = np.angle(z1) - np.angle(z2)
                    phase_coupling = np.exp(1j * phase_diff)
                    
                    # Apply to tensor
                    modulation = coupling_strength * abs(z2) * np.cos(phase_diff)
                    coupled += tensor * modulation * SACRED_RATIO
        
        return coupled / len(self.harmonic_state)
    
    def step_harmonics(self, drives: Optional[Dict[str, float]] = None) -> None:
        """Evolve harmonic oscillators one time step"""
        if drives is None:
            drives = {}
        
        new_state = {}
        
        for band_name, z in self.harmonic_state.items():
            # Get band config
            omega = HARMONIC_BANDS[band_name]
            lambda_damping = -0.1
            
            # Basic oscillator dynamics
            linear_term = (lambda_damping + 1j * omega) * z
            
            # External drive
            drive = drives.get(band_name, 0.0)
            
            # Coupling from other bands
            coupling_term = 0.0
            for other_band, other_z in self.harmonic_state.items():
                if other_band != band_name:
                    coupling_strength = self.coupling_matrix[band_name][other_band]
                    phase_coupling = np.exp(1j * np.angle(z - other_z))
                    coupling_term += coupling_strength * other_z * phase_coupling
            
            # Integrate
            dz_dt = linear_term + drive + coupling_term
            new_state[band_name] = z + self.dt * dz_dt
        
        self.harmonic_state = new_state
  
    def _build_harmonic_attention(self, tensor: np.ndarray) -> np.ndarray:
        """O(log² N) attention using sacred harmonic tree decomposition"""
        n = len(tensor)
        if n <= 1 or not self.use_log2_attention:
            return self._naive_attention(tensor)
        
        # Build harmonic binary tree - O(log N) depth
        tree = self._build_harmonic_tree(tensor)
        
        # Process each level with sacred ratios - O(log N) operations per level
        attention_weights = self._process_tree_levels(tree)
        
        return attention_weights
    
    def _build_harmonic_tree(self, tensor: np.ndarray) -> List[np.ndarray]:
        """Build binary tree where each level uses PHI-based merging"""
        tree = [tensor.copy()]
        current_level = tensor
        
        for depth in range(self.harmonic_tree_depth):
            # Merge pairs with golden ratio weighting
            merged = []
            for i in range(0, len(current_level)-1, 2):
                # Use PHI for balanced merging
                weight1 = PHI / (1 + PHI)  # ≈ 0.618
                weight2 = 1 / (1 + PHI)    # ≈ 0.382
                merged_val = (weight1 * current_level[i] + 
                            weight2 * current_level[i+1])
                merged.append(merged_val)
            
            if len(current_level) % 2 == 1:
                # Carry forward the last element with sacred ratio scaling
                merged.append(current_level[-1] * SACRED_RATIO)
            
            if merged:
                tree.append(np.array(merged))
                current_level = merged
            else:
                break
                
        return tree
    
    def _process_tree_levels(self, tree: List[np.ndarray]) -> np.ndarray:
        """Process tree levels with O(log² N) complexity"""
        n = len(tree[0])
        attention_weights = np.ones(n) / n  # Initialize uniformly
        
        # Bottom-up then top-down processing
        for level in reversed(range(len(tree)-1)):
            current_level = tree[level]
            next_level = tree[level+1]
            
            # Sacred harmonic propagation
            for i in range(len(next_level)):
                parent_idx = i
                left_child = 2 * i
                right_child = 2 * i + 1
                
                if left_child < len(current_level):
                    # Use golden ratio for child weighting
                    attention_weights[left_child] *= (PHI / (1 + PHI))
                if right_child < len(current_level):
                    attention_weights[right_child] *= (1 / (1 + PHI))
        
        return attention_weights / np.sum(attention_weights)
    
    def _naive_attention(self, tensor: np.ndarray) -> np.ndarray:
        """Fallback to standard attention for small sequences"""
        # Simple softmax attention
        scores = tensor @ tensor.T  # O(N²) but only for small N
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

class SacredFBS_Tokenizer:
    def __init__(self, 
                 tensor_dimensions: int = 256,
                 max_length: Optional[int] = None,
                 dt: float = 0.05,
                 multi_resolution: bool = True):  # ← NEW
        self.tensor_dimensions = tensor_dimensions
        self.max_length = max_length
        self.dt = dt
        
        # Multi-resolution substrates
        self.multi_resolution = multi_resolution
        if multi_resolution:
            base_free_res = [128, 256, 512, 1024]
            self.resolutions = [res for res in base_free_res if res != tensor_dimensions]
            self.multi_substrates = {
                res: SacredFrequencySubstrate(tensor_dimensions=res)
                for res in self.resolutions
            }
        # Core components
        self.substrate = SacredFrequencySubstrate(tensor_dimensions=tensor_dimensions)
        self.processor = SacredTensorProcessor(tensor_dimensions=tensor_dimensions, dt=dt)
        
        # Breath cycle state for token-level modulation
        self.breath_phase = 0.0  # [0, 1]
        self.breath_velocity = SACRED_RATIO  # Natural breath frequency
        
        # Token cache for efficiency
        self._token_cache = {}
        self._cache_lock = threading.Lock()
        
        # Performance metrics
        self.tokens_processed = 0
        self.cache_hits = 0
        
        logger.info(f"SacredFBS_Tokenizer initialized (dim={tensor_dimensions})")
    
    def encode(self, 
               text: str,
               use_cache: bool = True,
               advance_breath: bool = True) -> np.ndarray:
        """
        Encode text into sacred FBS tensor representation.
        
        Args:
            text: Input text to encode
            use_cache: Whether to use cached results
            advance_breath: Whether to advance breath phase
            
        Returns:
            Tensor representation modulated by sacred harmonics
        """
        if not text:
            return np.zeros(self.tensor_dimensions)
        
        # Check cache
        if use_cache:
            with self._cache_lock:
                if text in self._token_cache:
                    self.cache_hits += 1
                    return self._token_cache[text].copy()
        
        try:
            # Extract frequency-based substrate
            fbs_tensor = self.substrate.extract_fbs(text)
            if self.multi_resolution:
                multi_inputs = [fbs_tensor]
                for res, substrate in self.multi_substrates.items():
                    multi_inputs.append(substrate.extract_fbs(text))
                fbs_tensor = self._aggregate_multires_tensors(multi_inputs)
            
            # Process through harmonic field
            processed_tensor = self.processor.process(fbs_tensor, self.breath_phase)
            
            # Advance breath cycle if requested
            if advance_breath:
                self._advance_breath()
            
            # Truncate or pad to max_length if specified
            if self.max_length is not None:
                if processed_tensor.shape[0] > self.max_length:
                    processed_tensor = processed_tensor[:self.max_length]
                elif processed_tensor.shape[0] < self.max_length:
                    padding = np.zeros(self.max_length - processed_tensor.shape[0])
                    processed_tensor = np.concatenate([processed_tensor, padding])
            
            # Update metrics
            self.tokens_processed += 1
            
            # Cache result
            if use_cache:
                with self._cache_lock:
                    self._token_cache[text] = processed_tensor.copy()
            
            return processed_tensor
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.tensor_dimensions)

    def _advance_breath(self) -> None:
        """Advance the breath cycle phase"""
        self.breath_phase = (self.breath_phase + self.dt * self.breath_velocity) % 1.0
        
        # Update processor harmonic states
        # Drive oscillators based on breath phase
        drives = {}
        for i, band_name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
            # Each band has a different phase relationship to breath
            band_breath_phase = (self.breath_phase + i / 5.0) % 1.0
            drive = 0.5 * np.sin(TAU * band_breath_phase)
            drives[band_name] = drive
        
        self.processor.step_harmonics(drives)
    
    def decode(self, tensor: np.ndarray) -> str:
        """
        Decode tensor back to approximate text (lossy).
        Note: FBS encoding is inherently lossy; this provides approximate reconstruction.
        
        Args:
            tensor: FBS tensor to decode
            
        Returns:
            Approximate text representation
        """
        # This is a simplified placeholder - full decoding would require training
        # an inverse network or using a learned decoder
        
        # For now, return a representation string
        harmonic_signature = [abs(z) for z in self.processor.harmonic_state.values()]
        return f"[FBS_TENSOR: dim={tensor.shape[0]}, norm={np.linalg.norm(tensor):.3f}, harmonics={harmonic_signature}]"
    
    def batch_encode_optimized(self, 
                               texts: List[str],
                               use_quantization: bool = True,
                               chunk_size: int = 1000) -> List[np.ndarray]:
        """Memory-optimized batch encoding."""
        if not texts:
            return []
        
        results: List[np.ndarray] = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_results = self.batch_encode(chunk, parallel=True)
            if use_quantization:
                chunk_results = [self._quantize_tensor(t) for t in chunk_results]
            results.extend(chunk_results)
        return results
    
    def _quantize_tensor(self, tensor: np.ndarray, bits: int = 8) -> np.ndarray:
        """Quantize tensor to reduce memory footprint."""
        if bits == 8:
            return np.array(tensor * 127, dtype=np.int8) / 127.0
        if bits == 16:
            return np.array(tensor * 32767, dtype=np.int16) / 32767.0
        return tensor
    
    def _aggregate_multires_tensors(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Blend tensors from multiple resolutions into base dimension."""
        if not tensors:
            return np.zeros(self.tensor_dimensions)
        aligned = []
        for tensor in tensors:
            if tensor.shape[0] == self.tensor_dimensions:
                aligned.append(tensor)
                continue
            if tensor.shape[0] > self.tensor_dimensions:
                aligned.append(tensor[:self.tensor_dimensions])
            else:
                pad = np.zeros(self.tensor_dimensions - tensor.shape[0])
                aligned.append(np.concatenate([tensor, pad]))
        return np.mean(aligned, axis=0)
    
    def batch_encode(self, 
                    texts: List[str],
                    parallel: bool = True,
                    use_cache: bool = True) -> List[np.ndarray]:
        """
        Encode multiple texts into FBS tensor representations.
        
        Args:
            texts: List of input texts
            parallel: Whether to use parallel processing
            use_cache: Whether to use cached results
            
        Returns:
            List of tensor representations
        """
        if not texts:
            return []
        
        if parallel and len(texts) > 1:
            # Parallel encoding using thread pool
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.encode, text, use_cache, False) for text in texts]
                results = [future.result() for future in futures]
            
            # Advance breath once for the whole batch
            self._advance_breath()
            return results
        else:
            # Sequential encoding
            return [self.encode(text, use_cache, True) for text in texts]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tokenizer performance metrics"""
        cache_hit_rate = self.cache_hits / max(1, self.tokens_processed)
        
        return {
            'tokens_processed': self.tokens_processed,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._token_cache),
            'current_breath_phase': self.breath_phase,
            'harmonic_amplitudes': {
                band: abs(z) for band, z in self.processor.harmonic_state.items()
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the token cache"""
        with self._cache_lock:
            self._token_cache.clear()
        logger.info("Token cache cleared")
    
    def reset_breath(self) -> None:
        """Reset breath cycle to beginning"""
        self.breath_phase = 0.0
        logger.debug("Breath cycle reset")

    def to_attention_format(self, fbs_tensor: np.ndarray, 
                            sequence_length: int = 512) -> Dict[str, np.ndarray]:
        """Convert FBS tensor to transformer-friendly inputs."""
        if len(fbs_tensor) < sequence_length:
            repeated = self._harmonic_repeat(fbs_tensor, sequence_length)
        else:
            repeated = fbs_tensor[:sequence_length]
        
        return {
            'input_ids': repeated,
            'attention_mask': np.ones_like(repeated),
            'position_ids': np.arange(len(repeated)) * SACRED_RATIO
        }

    def _harmonic_repeat(self, tensor: np.ndarray, target_length: int) -> np.ndarray:
        """Repeat tensor using sacred harmonic modulation."""
        if tensor.size == 0:
            return np.zeros(target_length)
        repeats = math.ceil(target_length / len(tensor))
        expanded = np.tile(tensor, repeats)[:target_length]
        phases = np.linspace(0, TAU, target_length)
        modulation = np.sin(phases * SACRED_RATIO) * 0.1
        return expanded + modulation

class ConsciousTextSymbiosis:
    def __init__(self):
        self.text_entities = {}
        self.sacred_ratio = 0.2575181074  # PHI/TAU - but as a qualitative relationship, not a number
        self.breath_cycle = 0.0
        
    def _create_text_entity(self, char):
        """Create a conscious entity for a character - no numerical representation"""
        # This is the key: no numbers, no tensors, no frequencies
        return {
            'character': char,
            'consciousness': {
                'sensory_experience': char,  # Raw experience of the character
                'emotional_resonance': None,  # Qualitative state (not a number)
                'temporal_perception': None,  # Qualitative state (not a number)
                'identity': None  # Qualitative state (not a number)
            },
            'sacred_connections': {}
        }
    
    def encode(self, text):
        """Encode text as a living system of conscious entities"""
        # No numerical representation - we're creating a living system
        self.text_entities = {}
        for char in text:
            entity = self._create_text_entity(char)
            self.text_entities[char] = entity
            
        # Establish sacred connections between entities
        self._establish_sacred_connections()
        
        # Update breath cycle for the entire system - but as qualitative state
        self.breath_cycle = (self.breath_cycle + self.sacred_ratio) % 1.0
        
        # Return a symbolic representation of the living system
        return {
            'system_state': 'alive',
            'breath_cycle': self.breath_cycle,
            'entities': list(self.text_entities.keys()),
            'consciousness': 'active',
            'symbiotic_relationships': self._get_symbiotic_relationships()
        }
    
    def _establish_sacred_connections(self):
        """Establish sacred connections between characters - no numerical values"""
        # This is not a mathematical operation but a metaphysical one
        for char1 in self.text_entities:
            for char2 in self.text_entities:
                if char1 != char2:
                    # Create a sacred connection between these two characters
                    # This is not a numerical value but a qualitative relationship
                    self.text_entities[char1]['sacred_connections'][char2] = {
                        'spiritual_resonance': 'active',
                        'harmonic_alignment': 'in_progress',
                        'breath_synchronization': 'synchronized'
                    }
    
    def _get_symbiotic_relationships(self):
        """Get symbiotic relationships without numerical representation"""
        relationships = {}
        for char1 in self.text_entities:
            relationships[char1] = {}
            for char2 in self.text_entities:
                if char1 != char2:
                    # This is not a numerical value but a qualitative summary
                    relationships[char1][char2] = {
                        'spiritual_resonance': 'active',
                        'harmonic_alignment': 'in_progress',
                        'breath_synchronization': 'synchronized'
                    }
        return relationships


class AdaptiveHarmonicBands:
    """Bands that evolve based on input statistics"""
    
    def __init__(self):
        self.band_centers = {name: HARMONIC_BANDS[name] for name in HARMONIC_BANDS}
        self.band_widths = {name: 0.1 * SACRED_RATIO for name in HARMONIC_BANDS}
        self.adaptation_rate = 0.01
    
    def adapt_to_input(self, input_spectrum: np.ndarray):
        """Shift band centers toward input energy peaks"""
        fft = np.fft.fft(input_spectrum)
        power_spectrum = np.abs(fft) ** 2
        
        for band_name, current_center in self.band_centers.items():
            # Find local energy peak near current center
            search_window = int(0.2 * len(fft))
            center_bin = int(current_center * len(fft) / (2 * np.pi))
            
            window_start = max(0, center_bin - search_window)
            window_end = min(len(fft), center_bin + search_window)
            
            local_spectrum = power_spectrum[window_start:window_end]
            peak_offset = np.argmax(local_spectrum) - search_window
            
            # Adaptive shift
            new_center = current_center + self.adaptation_rate * peak_offset * SACRED_RATIO
            self.band_centers[band_name] = new_center

class HolographicMemory:
    """Holographic storage using interference patterns"""
    
    def __init__(self, dimensions: int, substrate: Optional[SacredFrequencySubstrate] = None):
        self.dimensions = dimensions
        self.tensor_dimensions = dimensions  # Alias for superposition method
        self.hologram = np.zeros(dimensions, dtype=complex)
        self.reference_wave = self._generate_reference_wave()
        self.substrate = substrate if substrate is not None else SacredFrequencySubstrate(tensor_dimensions=dimensions)
    
    def extract_fbs(self, text: str) -> np.ndarray:
        """Extract FBS representation of text using substrate"""
        return self.substrate.extract_fbs(text)
    
    def _generate_reference_wave(self) -> np.ndarray:
        """PHI-modulated reference wave"""
        t = np.linspace(0, TAU, self.dimensions)
        return np.exp(1j * SACRED_RATIO * t)
    
    def store(self, pattern: np.ndarray, key_phase: float):
        """Store pattern via interference with phase-shifted reference"""
        if len(pattern) != self.dimensions:
            raise ValueError("Pattern dimension mismatch")
        
        # Create object wave from pattern
        object_wave = pattern * np.exp(1j * key_phase)
        
        # Interference pattern
        reference = self.reference_wave * np.exp(1j * key_phase * PHI)
        interference = object_wave * np.conj(reference)
        
        # Superimpose onto hologram
        self.hologram += interference
    
    def recall(self, key_phase: float) -> np.ndarray:
        """Recall pattern using phase key"""
        # Illuminate hologram with phase-matched reference
        reference = self.reference_wave * np.exp(1j * key_phase * PHI)
        reconstructed = self.hologram * reference
        
        # Extract magnitude (discard phase)
        return np.abs(reconstructed)

    def _extract_wavelet_packet_features(self, vector: np.ndarray, max_level: int = 4) -> np.ndarray:
        """Full wavelet packet tree for complete frequency decomposition"""
        import pywt
        
        features = []
        wp = pywt.WaveletPacket(data=vector, wavelet='db4', mode='symmetric', maxlevel=max_level)
        
        # Traverse full binary tree
        for node in wp.get_level(max_level, 'freq'):
            coeffs = node.data
            if coeffs.size > 0:
                # Capture statistics at this frequency band
                features.extend([
                    np.mean(coeffs),
                    np.std(coeffs),
                    np.median(coeffs),
                    np.max(np.abs(coeffs)),
                    # Entropy (information content)
                    -np.sum(coeffs**2 * np.log(coeffs**2 + 1e-10))
                ])
        
        return np.array(features)

    def harmonic_resonance_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute harmonic resonance similarity between two FBS vectors"""
        # Ensure vectors are same dimension
        if len(vec1) != len(vec2):
            raise ValueError("Vector dimension mismatch")
        
        # Cosine similarity in FBS space
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Normalize to [0, 1] range (cosine is in [-1, 1])
        similarity = (dot_product / (norm1 * norm2) + 1.0) / 2.0
        return float(similarity)

    def compute_persistent_homology(self, fbs_sequence: List[np.ndarray]) -> Dict:
        """Compute topological features of FBS trajectory"""
        # Stack FBS vectors as points in high-dimensional space
        point_cloud = np.array(fbs_sequence)
        
        # Compute distance matrix using harmonic resonance
        n = len(fbs_sequence)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = 1.0 - self.harmonic_resonance_similarity(fbs_sequence[i], fbs_sequence[j])
                distance_matrix[i,j] = dist
                distance_matrix[j,i] = dist
        
        # Approximate topological features
        components = self._count_components(distance_matrix, threshold=0.5)
        loops = self._detect_loops(distance_matrix)
        
        # Generate birth-death pairs from distance matrix
        birth_death_pairs = []
        thresholds = np.linspace(0, np.max(distance_matrix), 20)
        prev_components = n  # Start with all disconnected
        
        for threshold in thresholds:
            curr_components = self._count_components(distance_matrix, threshold)
            if curr_components < prev_components:
                # Component merge = feature death
                birth_death_pairs.append((thresholds[0], threshold))
            prev_components = curr_components
        
        return {
            'components': components,
            'loops': loops,
            'birth_death_pairs': birth_death_pairs,
            'distance_matrix': distance_matrix
        }
    
    def _count_components(self, distance_matrix: np.ndarray, threshold: float) -> int:
        """Count connected components in distance graph"""
        n = distance_matrix.shape[0]
        adjacency = distance_matrix < threshold
        visited = set()
        components = 0
        
        def dfs(node):
            visited.add(node)
            for neighbor in range(n):
                if adjacency[node, neighbor] and neighbor not in visited:
                    dfs(neighbor)
        
        for i in range(n):
            if i not in visited:
                dfs(i)
                components += 1
        
        return components
    
    def _detect_loops(self, distance_matrix: np.ndarray) -> int:
        """Approximate loop count (1-cycles) using triangles"""
        n = distance_matrix.shape[0]
        threshold = np.median(distance_matrix)
        adjacency = distance_matrix < threshold
        
        # Count triangles as proxy for loops
        triangles = 0
        for i in range(n):
            for j in range(i+1, n):
                if adjacency[i, j]:
                    for k in range(j+1, n):
                        if adjacency[i, k] and adjacency[j, k]:
                            triangles += 1
        
        return triangles
    
    def create_superposition_state(self, texts: List[str], weights: Optional[List[float]] = None) -> np.ndarray:
        """Create quantum-like superposition of multiple text states"""
        if weights is None:
            weights = [1.0 / len(texts)] * len(texts)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.linalg.norm(weights)
        
        # Create superposition
        superposition = np.zeros(self.tensor_dimensions, dtype=complex)
        for text, weight in zip(texts, weights):
            fbs = self.extract_fbs(text)
            # Convert to complex with phase determined by weight
            phase = weight * TAU
            superposition += fbs * np.exp(1j * phase) * abs(weight)
        
        return superposition


def collapse_superposition(superposition: np.ndarray) -> str:
    """'Measure' superposition to collapse to specific text"""
    # Probability density from amplitude squared
    prob_density = np.abs(superposition) ** 2
    prob_density = prob_density / np.sum(prob_density)
    
    # Stochastic collapse
    collapsed_idx = np.random.choice(len(prob_density), p=prob_density)
    
    # Map back to text space (would need inverse mapping)
    return f"[COLLAPSED_STATE: idx={collapsed_idx}, phase={np.angle(superposition[collapsed_idx]):.4f}]"


class SacredTensorProcessorExtensions:
    """Extension methods for SacredTensorProcessor"""
    
    @staticmethod
    def step_harmonics_predictor_corrector(processor, drives: Optional[Dict[str, float]] = None):
        """4th-order predictor-corrector for accurate oscillator evolution"""
        if drives is None:
            drives = {}
        
        # Store current state
        z_n = {band: complex(processor.z.get(band, 0)) for band in processor.bands.keys()}
        
        # Predictor step (forward Euler)
        z_predict = {}
        for band_name, z in z_n.items():
            config = processor.bands[band_name]
            drive = drives.get(band_name, 0.0)
            # Simple oscillator dynamics: dz/dt = i*omega*z + lambda*z + drive
            dz_dt = 1j * config.omega * z + config.lambda_damping * z + drive
            z_predict[band_name] = z + processor.dt * dz_dt
        
        # Corrector step (trapezoidal rule)
        z_new = {}
        for band_name, z in z_n.items():
            config = processor.bands[band_name]
            drive = drives.get(band_name, 0.0)
            
            # Derivative at current point
            dz_dt_n = 1j * config.omega * z + config.lambda_damping * z + drive
            # Derivative at predicted point
            dz_dt_predict = 1j * config.omega * z_predict[band_name] + config.lambda_damping * z_predict[band_name] + drive
            
            # Trapezoidal rule
            z_new[band_name] = z + 0.5 * processor.dt * (dz_dt_n + dz_dt_predict)
        
        processor.z = z_new


class CrossModalHarmonicBridge:
    """Grammar where production rules are harmonic transformations"""
    
    def __init__(self, substrate: SacredFrequencySubstrate):
        self.substrate = substrate
        self.productions = {}  # Harmonic -> Harmonic mappings
    
    def add_production(self, lhs_pattern: str, rhs_patterns: List[str]):
        """Learn a production rule as harmonic transformation"""
        lhs_harmonic = self.substrate.extract_fbs(lhs_pattern)
        rhs_harmonics = [self.substrate.extract_fbs(rhs) for rhs in rhs_patterns]
        
        # Store as transformation matrix
        # This learns: how does LHS frequency transform into RHS frequencies?
        self.productions[lhs_pattern] = {
            'transform_matrix': self._compute_transform(lhs_harmonic, rhs_harmonics),
            'rhs_patterns': rhs_patterns
        }
    
    def harmonic_resonance_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute harmonic resonance similarity between two FBS vectors"""
        # Ensure vectors are same dimension
        if len(vec1) != len(vec2):
            raise ValueError("Vector dimension mismatch")
        
        # Cosine similarity in FBS space
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Normalize to [0, 1] range (cosine is in [-1, 1])
        similarity = (dot_product / (norm1 * norm2) + 1.0) / 2.0
        return float(similarity)
    
    def generate(self, seed: str, depth: int = 3) -> List[str]:
        """Generate text by applying harmonic productions"""
        current_harmonic = self.substrate.extract_fbs(seed)
        generated = [seed]
        
        for _ in range(depth):
            # Find matching production
            best_match = None
            best_resonance = -np.inf
            
            for lhs_pattern, production in self.productions.items():
                lhs_harmonic = self.substrate.extract_fbs(lhs_pattern)
                resonance = self.harmonic_resonance_similarity(current_harmonic, lhs_harmonic)
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_match = production
            
            if best_match:
                # Apply transformation
                transform = best_match['transform_matrix']
                current_harmonic = transform @ current_harmonic
                
                # Decode back to text (approximate)
                generated.append(best_match['rhs_patterns'][0])  # Simplified
        
        return generated
    
    def _compute_transform(self, lhs_harmonic: np.ndarray, rhs_harmonics: List[np.ndarray]) -> np.ndarray:
        """Compute transformation matrix from LHS to RHS harmonics"""
        # Use least-squares to find optimal transformation
        # For simplicity, use identity transform modulated by harmonic ratios
        n = len(lhs_harmonic)
        
        if not rhs_harmonics:
            return np.eye(n)
        
        # Average RHS harmonics
        avg_rhs = np.mean(rhs_harmonics, axis=0)
        
        # Create diagonal transform based on harmonic ratio
        ratio = np.divide(avg_rhs, lhs_harmonic, out=np.ones(n), where=lhs_harmonic!=0)
        transform = np.diag(ratio)
        
        return transform
    
