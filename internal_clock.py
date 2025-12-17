
import time
import datetime
import math
import logging
import uuid
# Optional heavy deps: import lazily or fall back if unavailable to avoid
# import-time failures in constrained or instrumented environments.
try:
    import numpy as np
except Exception:
    np = None
try:
    import asyncio
except Exception:
    asyncio = None
try:
    import pytz
except Exception:
    pytz = None
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import pytz
import json
import hashlib
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (Temporal Coherence) %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('temporal_coherence')

# ============================================================================
# Constants and Enumerations
# ============================================================================

class TimeScale(Enum):
    """Classification of time scales from quantum to cosmological"""
    QUANTUM = auto()       # Planck time scale (10^-43 seconds)
    ATOMIC = auto()        # Atomic transitions (10^-10 seconds)
    NEURAL = auto()        # Neural firing (milliseconds)
    PHYSIOLOGICAL = auto() # Body rhythms (seconds to hours)
    CIRCADIAN = auto()     # ~24 hour cycles
    CALENDAR = auto()      # Days to years
    GEOLOGICAL = auto()    # Thousands to millions of years
    COSMOLOGICAL = auto()  # Billions of years
    
class ClockType(Enum):
    """Types of time measurement systems"""
    SYSTEM = auto()        # Computer system clock
    ATOMIC = auto()        # Atomic clock (simulated)
    ASTRONOMICAL = auto()  # Based on Earth's rotation/orbit
    BIOLOGICAL = auto()    # Based on simulated biological rhythms
    SUBJECTIVE = auto()    # Subjective perception of time
    NARRATIVE = auto()     # Story/experience-based time

class NeuralOscillation(Enum):
    """Neural oscillation frequency bands"""
    DELTA = auto()         # 0.5-4 Hz (deep sleep)
    THETA = auto()         # 4-8 Hz (drowsiness, meditation)
    ALPHA = auto()         # 8-13 Hz (relaxed wakefulness)
    BETA = auto()          # 13-30 Hz (active thinking)
    GAMMA = auto()         # 30-100 Hz (active processing)
    
class CircadianPhase(Enum):
    """Phases of the circadian rhythm"""
    EARLY_MORNING = auto() # 4am-8am (rising cortisol)
    MORNING = auto()       # 8am-12pm (high alertness)
    AFTERNOON = auto()     # 12pm-4pm (post-lunch dip)
    EVENING = auto()       # 4pm-8pm (highest body temperature)
    NIGHT = auto()         # 8pm-12am (melatonin secretion)
    DEEP_NIGHT = auto()    # 12am-4am (deepest sleep)

class AttentionalState(Enum):
    """Attentional states affecting time perception"""
    FLOW = auto()          # Deep focus/flow state
    ALERT = auto()         # Active attention
    NORMAL = auto()        # Baseline attention
    DISTRACTED = auto()    # Divided attention
    BORED = auto()         # Low stimulation
    STRESSED = auto()      # High arousal/anxiety

# Scientific constants
PLANCK_TIME = 5.39e-44  # Planck time in seconds
ATOMIC_SECOND = 9192631770  # Cycles of Cs-133 atom defining a second
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 31557600  # Average seconds per year (365.25 days)
LIGHT_SECOND = 299792458  # Speed of light in m/s = 1 light-second
EARTH_ROTATION_PERIOD = 86164.0905  # Sidereal day in seconds
LUNAR_SYNODIC_MONTH = 29.530588 * SECONDS_PER_DAY  # Lunar month in seconds

# Biological constants
DEFAULT_CIRCADIAN_PERIOD = 24.2 * 3600  # Free-running human circadian in seconds
ULTRADIAN_CYCLES = {
    "breath": 4.0,         # Seconds per breath cycle
    "alpha_wave": 0.1,     # Seconds per alpha wave cycle
    "heart_beat": 0.8,     # Seconds per heartbeat
    "REM_cycle": 90 * 60,  # Seconds per sleep cycle
    "ultradian_alertness": 90 * 60  # Basic rest-activity cycle
}

# Subjective time constants
DEFAULT_SUBJECTIVE_SCALAR = 1.0  # Default factor for subjective time
TIME_DILATION_FACTORS = {
    AttentionalState.FLOW: 0.5,        # Time feels faster in flow
    AttentionalState.ALERT: 0.8,       # Time feels slightly faster when alert
    AttentionalState.NORMAL: 1.0,      # Baseline
    AttentionalState.DISTRACTED: 1.1,  # Time feels slightly slower when distracted
    AttentionalState.BORED: 1.5,       # Time feels slower when bored
    AttentionalState.STRESSED: 1.2     # Time feels slower when stressed
}

# Memory constants
MEMORY_DECAY_RATE = 0.1  # Rate at which temporal memories decay
MEMORY_RESOLUTION = {    # Temporal resolution of memories at distances
    "recent": 1.0,       # High resolution for recent events
    "hour": 0.9,         # Slight decrease in resolution
    "day": 0.7,          # Further decrease
    "week": 0.5,         # Substantial decrease
    "month": 0.3,        # Low resolution
    "year": 0.1          # Very low resolution
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TimePoint:
    """
    Represents a precise moment across multiple timekeeping systems
    """
    system_time: float  # Unix timestamp
    system_datetime: datetime.datetime  # Datetime object
    atomic_time: float  # Simulated atomic time
    sequence_id: int  # Monotonically increasing sequence number
    clock_type: ClockType  # Source of this timepoint
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    @classmethod
    def now(cls, clock_type: ClockType = ClockType.SYSTEM) -> 'TimePoint':
        """Create a TimePoint for the current moment.

        Returns a timezone-aware UTC datetime and a stable sequence id derived
        from monotonic clock when available to provide increasing sequence ids
        even across rapid successive calls.
        """
        # Wall-clock time
        now_ts = time.time()
        # Prefer timezone-aware UTC datetime
        try:
            now_dt = datetime.datetime.fromtimestamp(now_ts, tz=datetime.timezone.utc)
        except Exception:
            now_dt = datetime.datetime.utcfromtimestamp(now_ts)

        # Sequence id: prefer monotonic clock for uniqueness and monotonicity
        try:
            mono = time.monotonic()
            sequence_id = int((mono * 1000000) % (2**63))
        except Exception:
            sequence_id = int((now_ts * 1000000) % (2**63))

        # Atomic time: if numpy available, use higher-resolution counter, else use monotonic
        if np is not None:
            try:
                atomic_time = now_ts + (np.random.random() % 1e-6)
            except Exception:
                atomic_time = now_ts
        else:
            atomic_time = now_ts

        return cls(
            system_time=now_ts,
            system_datetime=now_dt,
            atomic_time=atomic_time,
            sequence_id=sequence_id,
            clock_type=clock_type
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "system_time": self.system_time,
            "system_datetime": self.system_datetime.isoformat(),
            "atomic_time": self.atomic_time,
            "sequence_id": self.sequence_id,
            "clock_type": self.clock_type.name,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimePoint':
        """Recreate from dictionary"""
        return cls(
            system_time=data["system_time"],
            system_datetime=datetime.datetime.fromisoformat(data["system_datetime"]),
            atomic_time=data["atomic_time"],
            sequence_id=data["sequence_id"],
            clock_type=ClockType[data["clock_type"]],
            metadata=data.get("metadata", {})
        )
    
    def get_age(self) -> float:
        """Get the age of this timepoint in seconds"""
        return time.time() - self.system_time


@dataclass
class TemporalEvent:
    """
    Represents an event with temporal coordinates and memory characteristics
    """
    event_id: str  # Unique identifier
    start_time: TimePoint  # When the event started
    end_time: Optional[TimePoint] = None  # When the event ended (None if instantaneous)
    duration: Optional[float] = None  # Duration in seconds if known
    event_type: str = "generic"  # Type of event
    subjective_duration: Optional[float] = None  # Subjective experience of duration
    memory_strength: float = 1.0  # Initial memory strength (0.0-1.0)
    memory_decay_rate: float = MEMORY_DECAY_RATE  # How quickly the memory decays
    importance: float = 0.5  # Importance of this event (0.0-1.0)
    tags: Set[str] = field(default_factory=set)  # Tags for categorization
    content: Any = None  # Event content/details
    references: List[str] = field(default_factory=list)  # References to other events
    
    def __post_init__(self):
        """Initialize derived values after creation"""
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        
        # Calculate duration if end_time is provided
        if self.end_time is not None and self.duration is None:
            self.duration = self.end_time.system_time - self.start_time.system_time
    
    def finalize(self, end_time: Optional[TimePoint] = None) -> None:
        """Finalize an event by setting its end time"""
        if end_time is None:
            self.end_time = TimePoint.now()
        else:
            self.end_time = end_time
            
        # Calculate duration
        self.duration = self.end_time.system_time - self.start_time.system_time
    
    def get_current_memory_strength(self) -> float:
        """Calculate current memory strength based on decay"""
        if self.end_time is None:
            age = time.time() - self.start_time.system_time
        else:
            age = time.time() - self.end_time.system_time
            
        # Apply decay function (exponential decay)
        decay_factor = math.exp(-self.memory_decay_rate * age / SECONDS_PER_DAY)
        
        # Apply importance as a modifier to slow decay for important events
        importance_modifier = 1.0 + (self.importance * 2.0)  # Range: 1.0-3.0
        adjusted_decay = decay_factor ** (1.0 / importance_modifier)
        
        return self.memory_strength * adjusted_decay
    
    def get_subjective_duration(self) -> float:
        """Get subjective duration, calculating if needed"""
        if self.subjective_duration is not None:
            return self.subjective_duration
        
        if self.duration is None:
            return 0.0
            
        # Default estimate if no specific subjective duration was set
        # More important events feel shorter; longer events feel relatively shorter
        importance_factor = 1.0 - (self.importance * 0.5)  # 0.5-1.0
        duration_factor = math.log10(1 + self.duration) / math.log10(1 + 3600)  # Logarithmic scaling
        
        self.subjective_duration = self.duration * importance_factor * (0.5 + 0.5 * duration_factor)
        return self.subjective_duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "start_time": self.start_time.to_dict(),
            "end_time": self.end_time.to_dict() if self.end_time else None,
            "duration": self.duration,
            "event_type": self.event_type,
            "subjective_duration": self.subjective_duration,
            "memory_strength": self.memory_strength,
            "memory_decay_rate": self.memory_decay_rate,
            "importance": self.importance,
            "tags": list(self.tags),
            "content": self.content,
            "references": self.references
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalEvent':
        """Recreate from dictionary"""
        return cls(
            event_id=data["event_id"],
            start_time=TimePoint.from_dict(data["start_time"]),
            end_time=TimePoint.from_dict(data["end_time"]) if data.get("end_time") else None,
            duration=data.get("duration"),
            event_type=data.get("event_type", "generic"),
            subjective_duration=data.get("subjective_duration"),
            memory_strength=data.get("memory_strength", 1.0),
            memory_decay_rate=data.get("memory_decay_rate", MEMORY_DECAY_RATE),
            importance=data.get("importance", 0.5),
            tags=set(data.get("tags", [])),
            content=data.get("content"),
            references=data.get("references", [])
        )


@dataclass
class OscillatorState:
    """
    State of a simulated neural/biological oscillator
    """
    type_id: str  # Identifier for this oscillator type
    frequency: float  # Frequency in Hz
    phase: float  # Current phase (0-2π)
    amplitude: float  # Current amplitude
    baseline: float  # Baseline/center value
    last_update: float  # Last update time (system time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional parameters
    
    def get_value(self, time_point: Optional[float] = None) -> float:
        """Get the current value of the oscillator"""
        if time_point is None:
            time_point = time.time()
            
        # Calculate how much time has passed since last update
        dt = time_point - self.last_update
        
        # Update phase based on frequency
        self.phase = (self.phase + 2 * math.pi * self.frequency * dt) % (2 * math.pi)
        self.last_update = time_point
        
        # Calculate and return current value
        return self.baseline + self.amplitude * math.sin(self.phase)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "type_id": self.type_id,
            "frequency": self.frequency,
            "phase": self.phase,
            "amplitude": self.amplitude,
            "baseline": self.baseline,
            "last_update": self.last_update,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OscillatorState':
        """Recreate from dictionary"""
        return cls(
            type_id=data["type_id"],
            frequency=data["frequency"],
            phase=data["phase"],
            amplitude=data["amplitude"],
            baseline=data["baseline"],
            last_update=data["last_update"],
            metadata=data.get("metadata", {})
        )


@dataclass
class TemporalMemory:
    """Memory structure for temporal events with associative capabilities"""
    events: Dict[str, TemporalEvent] = field(default_factory=dict)  # Event ID -> Event
    time_index: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))  # Day timestamp -> Event IDs
    type_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # Event type -> Event IDs
    tag_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # Tag -> Event IDs
    reference_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # Event ID -> Referenced by
    
    def add_event(self, event: TemporalEvent) -> None:
        """Add an event to memory"""
        # Store the event
        self.events[event.event_id] = event
        
        # Index by day
        day_ts = int(event.start_time.system_time // SECONDS_PER_DAY)
        self.time_index[day_ts].append(event.event_id)
        
        # Index by type
        self.type_index[event.event_type].append(event.event_id)
        
        # Index by tags
        for tag in event.tags:
            self.tag_index[tag].append(event.event_id)
        
        # Index by references
        for ref in event.references:
            self.reference_index[ref].append(event.event_id)
    
    def get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """Retrieve an event by ID"""
        return self.events.get(event_id)
    
    def find_events(self, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   event_type: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   min_memory_strength: float = 0.0,
                   importance_threshold: float = 0.0,
                   max_results: int = 100) -> List[TemporalEvent]:
        """
        Find events matching the specified criteria
        """
        candidate_ids = set()
        filters_applied = False
        
        # Filter by time range
        if start_time is not None or end_time is not None:
            filters_applied = True
            start_day = int((start_time or 0) // SECONDS_PER_DAY)
            end_day = int((end_time or time.time()) // SECONDS_PER_DAY)
            
            time_candidates = set()
            for day in range(start_day, end_day + 1):
                time_candidates.update(self.time_index.get(day, []))
            
            if candidate_ids:
                candidate_ids &= time_candidates
            else:
                candidate_ids = time_candidates
        
        # Filter by event type
        if event_type is not None:
            filters_applied = True
            type_candidates = set(self.type_index.get(event_type, []))
            
            if candidate_ids:
                candidate_ids &= type_candidates
            else:
                candidate_ids = type_candidates
        
        # Filter by tags
        if tags:
            filters_applied = True
            tag_candidates = set()
            for tag in tags:
                tag_candidates.update(self.tag_index.get(tag, []))
            
            if candidate_ids:
                candidate_ids &= tag_candidates
            else:
                candidate_ids = tag_candidates
        
        # If no filters applied, use all events
        if not filters_applied:
            candidate_ids = set(self.events.keys())
        
        # Apply post-retrieval filters and collect results
        results = []
        for event_id in candidate_ids:
            event = self.events.get(event_id)
            if not event:
                continue
                
            # Check memory strength
            if event.get_current_memory_strength() < min_memory_strength:
                continue
                
            # Check importance
            if event.importance < importance_threshold:
                continue
                
            # Apply precise time filtering if needed
            if start_time is not None and event.start_time.system_time < start_time:
                continue
                
            if end_time is not None and (event.end_time is None or event.end_time.system_time > end_time):
                continue
            
            results.append(event)
            
            # Check limit
            if len(results) >= max_results:
                break
        
        # Sort by start time
        results.sort(key=lambda e: e.start_time.system_time)
        
        return results
    
    def prune_weak_memories(self, threshold: float = 0.1) -> int:
        """
        Remove memories that have decayed below threshold strength
        Returns the number of items removed
        """
        to_remove = []
        
        # Find weak memories
        for event_id, event in self.events.items():
            if event.get_current_memory_strength() < threshold:
                to_remove.append(event_id)
        
        # Remove them from all indices
        for event_id in to_remove:
            event = self.events.pop(event_id, None)
            if not event:
                continue
                
            # Remove from time index
            day_ts = int(event.start_time.system_time // SECONDS_PER_DAY)
            if day_ts in self.time_index:
                if event_id in self.time_index[day_ts]:
                    self.time_index[day_ts].remove(event_id)
                if not self.time_index[day_ts]:
                    del self.time_index[day_ts]
            
            # Remove from type index
            if event.event_type in self.type_index:
                if event_id in self.type_index[event.event_type]:
                    self.type_index[event.event_type].remove(event_id)
                if not self.type_index[event.event_type]:
                    del self.type_index[event.event_type]
            
            # Remove from tag index
            for tag in event.tags:
                if tag in self.tag_index:
                    if event_id in self.tag_index[tag]:
                        self.tag_index[tag].remove(event_id)
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]
            
            # Remove from reference index
            for ref in event.references:
                if ref in self.reference_index:
                    if event_id in self.reference_index[ref]:
                        self.reference_index[ref].remove(event_id)
                    if not self.reference_index[ref]:
                        del self.reference_index[ref]
        
        return len(to_remove)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory state"""
        total_events = len(self.events)
        avg_memory_strength = 0.0
        event_type_counts = {}
        memory_age_distribution = {
            "recent": 0,      # <1 hour
            "today": 0,       # <1 day
            "week": 0,        # <1 week
            "month": 0,       # <1 month
            "older": 0        # >1 month
        }
        
        now = time.time()
        
        for event in self.events.values():
            # Memory strength
            avg_memory_strength += event.get_current_memory_strength()
            
            # Event types
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
            
            # Age distribution
            age = now - event.start_time.system_time
            if age < 3600:
                memory_age_distribution["recent"] += 1
            elif age < SECONDS_PER_DAY:
                memory_age_distribution["today"] += 1
            elif age < 7 * SECONDS_PER_DAY:
                memory_age_distribution["week"] += 1
            elif age < 30 * SECONDS_PER_DAY:
                memory_age_distribution["month"] += 1
            else:
                memory_age_distribution["older"] += 1
                
        if total_events > 0:
            avg_memory_strength /= total_events
            
        return {
            "total_events": total_events,
            "avg_memory_strength": avg_memory_strength,
            "event_type_counts": event_type_counts,
            "memory_age_distribution": memory_age_distribution,
            "total_days": len(self.time_index),
            "total_types": len(self.type_index),
            "total_tags": len(self.tag_index),
            "total_references": len(self.reference_index)
        }


@dataclass
class TimePerceptionState:
    """Current state of time perception"""
    attentional_state: AttentionalState = AttentionalState.NORMAL  # Current attention state
    subjective_scalar: float = DEFAULT_SUBJECTIVE_SCALAR  # Current time perception scalar
    time_perception_factors: Dict[str, float] = field(default_factory=dict)  # Named factors affecting perception
    last_update: float = field(default_factory=time.time)  # Last time state was updated
    
    def update(self, 
              attentional_state: Optional[AttentionalState] = None,
              additional_factors: Optional[Dict[str, float]] = None) -> None:
        """Update the time perception state"""
        now = time.time()
        
        # Update attentional state if provided
        if attentional_state is not None:
            self.attentional_state = attentional_state
        
        # Update additional factors if provided
        if additional_factors:
            self.time_perception_factors.update(additional_factors)
        
        # Recalculate the subjective scalar
        self.subjective_scalar = TIME_DILATION_FACTORS.get(self.attentional_state, 1.0)
        
        # Apply additional factors
        for factor_name, factor_value in self.time_perception_factors.items():
            self.subjective_scalar *= factor_value
            
        # Clamp to reasonable range (0.1x to 10x)
        self.subjective_scalar = max(0.1, min(10.0, self.subjective_scalar))
        
        # Update timestamp
        self.last_update = now
    
    def get_subjective_duration(self, objective_duration: float) -> float:
        """Convert objective duration to subjective duration"""
        return objective_duration * self.subjective_scalar
    
    def get_objective_duration(self, subjective_duration: float) -> float:
        """Convert subjective duration to objective duration"""
        if self.subjective_scalar == 0:
            return 0.0
        return subjective_duration / self.subjective_scalar
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "attentional_state": self.attentional_state.name,
            "subjective_scalar": self.subjective_scalar,
            "time_perception_factors": self.time_perception_factors,
            "last_update": self.last_update
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimePerceptionState':
        """Recreate from dictionary"""
        return cls(
            attentional_state=AttentionalState[data["attentional_state"]],
            subjective_scalar=data["subjective_scalar"],
            time_perception_factors=data["time_perception_factors"],
            last_update=data["last_update"]
        )


# ============================================================================
# Core Class Implementations
# ============================================================================

class ScientificTimekeeper:
    """
    Manages timekeeping across multiple scientific time systems.
    Provides accurate reference times for the temporal coherence system.
    """
    
    def __init__(self, calibration_factor: float = 1.0):
        self.start_time = time.time()
        self.calibration_factor = calibration_factor
        self.reference_points = {}
        self.time_scales = {}
        
        # Initialize time scales
        for scale in TimeScale:
            self.time_scales[scale] = {
                "resolution": None,  # Smallest distinguishable time unit
                "baseline": None,    # Reference value/epoch
                "current": None      # Current value
            }
        
        # Set up scale-specific values
        self._initialize_time_scales()
        
        # Set reference points
        self.set_reference_point("initialization", TimePoint.now())
    
    def _initialize_time_scales(self):
        """Initialize values for different time scales"""
        
        # Quantum scale
        self.time_scales[TimeScale.QUANTUM]["resolution"] = PLANCK_TIME
        self.time_scales[TimeScale.QUANTUM]["baseline"] = 0
        
        # Atomic scale
        self.time_scales[TimeScale.ATOMIC]["resolution"] = 1.0 / ATOMIC_SECOND
        self.time_scales[TimeScale.ATOMIC]["baseline"] = 0
        
        # Neural scale
        self.time_scales[TimeScale.NEURAL]["resolution"] = 0.001  # 1ms
        self.time_scales[TimeScale.NEURAL]["baseline"] = 0
        
        # Physiological scale
        self.time_scales[TimeScale.PHYSIOLOGICAL]["resolution"] = 0.1  # 100ms
        self.time_scales[TimeScale.PHYSIOLOGICAL]["baseline"] = 0
        
        # Circadian scale
        self.time_scales[TimeScale.CIRCADIAN]["resolution"] = 60  # 1 minute
        self.time_scales[TimeScale.CIRCADIAN]["baseline"] = 0
        
        # Calendar scale
        self.time_scales[TimeScale.CALENDAR]["resolution"] = SECONDS_PER_DAY
        self.time_scales[TimeScale.CALENDAR]["baseline"] = 0
        
        # Geological scale
        self.time_scales[TimeScale.GEOLOGICAL]["resolution"] = SECONDS_PER_YEAR * 100  # Century
        self.time_scales[TimeScale.GEOLOGICAL]["baseline"] = 0
        
        # Cosmological scale
        self.time_scales[TimeScale.COSMOLOGICAL]["resolution"] = SECONDS_PER_YEAR * 1e6  # Million years
        self.time_scales[TimeScale.COSMOLOGICAL]["baseline"] = 0
    
    def set_reference_point(self, 
                           name: str, 
                           time_point: TimePoint,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set a named reference point"""
        self.reference_points[name] = {
            "time_point": time_point,
            "metadata": metadata or {}
        }
    
    def get_reference_point(self, name: str) -> Optional[TimePoint]:
        """Get a named reference point"""
        if name not in self.reference_points:
            return None
        return self.reference_points[name]["time_point"]
    
    def get_current_time(self, 
                        time_scale: TimeScale = TimeScale.PHYSIOLOGICAL,
                        clock_type: ClockType = ClockType.SYSTEM) -> float:
        """Get the current time in a particular scale"""
        now = time.time()
        
        # For most scales, we just return the number of seconds since start
        elapsed = (now - self.start_time) * self.calibration_factor
        
        if time_scale == TimeScale.QUANTUM:
            # Return time in Planck time units
            return elapsed / PLANCK_TIME
        
        elif time_scale == TimeScale.ATOMIC:
            # Return time in atomic cycles
            return elapsed * ATOMIC_SECOND
        
        elif time_scale == TimeScale.NEURAL:
            # Return time in milliseconds
            return elapsed * 1000
        
        # Default for all other scales: seconds
        return elapsed
    
    def get_current_timepoint(self, clock_type: ClockType = ClockType.SYSTEM) -> TimePoint:
        """Get a TimePoint for the current moment"""
        return TimePoint.now(clock_type)
    
    def get_elapsed_time(self, 
                        reference_name: Optional[str] = None,
                        time_scale: TimeScale = TimeScale.PHYSIOLOGICAL) -> float:
        """Get elapsed time since a reference point"""
        if reference_name is None:
            reference_name = "initialization"
            
        if reference_name not in self.reference_points:
            raise ValueError(f"Reference point '{reference_name}' not found")
        
        reference_time = self.reference_points[reference_name]["time_point"].system_time
        now = time.time()
        
        # Calculate elapsed time
        elapsed = (now - reference_time) * self.calibration_factor
        
        # Convert to the appropriate scale
        if time_scale == TimeScale.QUANTUM:
            return elapsed / PLANCK_TIME
        
        elif time_scale == TimeScale.ATOMIC:
            return elapsed * ATOMIC_SECOND
        
        elif time_scale == TimeScale.NEURAL:
            return elapsed * 1000
        
        # Default for all other scales: seconds
        return elapsed
    
    def get_time_resolution(self, time_scale: TimeScale) -> float:
        """Get the resolution of a time scale"""
        return self.time_scales[time_scale]["resolution"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        reference_points_dict = {}
        for name, data in self.reference_points.items():
            reference_points_dict[name] = {
                "time_point": data["time_point"].to_dict(),
                "metadata": data["metadata"]
            }
        
        return {
            "start_time": self.start_time,
            "calibration_factor": self.calibration_factor,
            "reference_points": reference_points_dict,
            "time_scales": {scale.name: value for scale, value in self.time_scales.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScientificTimekeeper':
        """Recreate from dictionary"""
        instance = cls(calibration_factor=data["calibration_factor"])
        instance.start_time = data["start_time"]
        
        # Restore reference points
        for name, point_data in data["reference_points"].items():
            instance.reference_points[name] = {
                "time_point": TimePoint.from_dict(point_data["time_point"]),
                "metadata": point_data["metadata"]
            }
            
        # Restore time scales
        for scale_name, scale_data in data["time_scales"].items():
            scale = TimeScale[scale_name]
            instance.time_scales[scale] = scale_data
            
        return instance


class BiologicalClock:
    """
    Simulates biological clock mechanisms including circadian rhythm,
    ultradian cycles, and neural oscillations.
    """
    
    def __init__(self, 
                circadian_period: float = DEFAULT_CIRCADIAN_PERIOD,
                phase_shift: float = 0.0,
                use_realtime: bool = True):
        # Core timing parameters
        self.circadian_period = circadian_period  # Natural period in seconds
        self.phase_shift = phase_shift  # Shift from midnight in seconds
        self.use_realtime = use_realtime  # Whether to use real time or simulated time
        
        # Internal state
        self.oscillators = {}  # Collection of oscillators
        self.current_phase = CircadianPhase.NIGHT  # Current circadian phase
        self.last_update = time.time()  # Last update timestamp
        self.simulation_time_factor = 1.0  # For accelerated simulation if needed
        self.simulation_time_elapsed = 0.0  # Simulated time elapsed if not using realtime
        
        # Initialize oscillators
        self._initialize_oscillators()
        
        # Entrainment parameters
        self.entrainment_factors = {
            "light": 1.0,         # Light level (0.0-1.0)
            "activity": 0.5,      # Activity level (0.0-1.0)
            "meal_timing": 0.5,   # Recent meal (0.0-1.0)
            "social": 0.5         # Social interactions (0.0-1.0)
        }
        
        # Current base body metrics
        self.core_temp = 36.8  # Core body temperature in °C
        self.cortisol_level = 0.5  # Cortisol level (0.0-1.0)
        self.melatonin_level = 0.5  # Melatonin level (0.0-1.0)
    
    def _initialize_oscillators(self):
        """Initialize the biological oscillators"""
        now = time.time()
        
        # Circadian oscillator
        self.oscillators["circadian"] = OscillatorState(
            type_id="circadian",
            frequency=1.0 / self.circadian_period,
            phase=(((now % self.circadian_period) + self.phase_shift) / self.circadian_period) * 2 * math.pi,
            amplitude=1.0,
            baseline=0.0,
            last_update=now
        )
        
        # Ultradian oscillators
        for name, period in ULTRADIAN_CYCLES.items():
            # Randomize starting phase
            random_phase = (hash(name) % 1000) / 1000.0 * 2 * math.pi
            
            self.oscillators[name] = OscillatorState(
                type_id=name,
                frequency=1.0 / period,
                phase=random_phase,
                amplitude=1.0,
                baseline=0.0,
                last_update=now
            )
        
        # Neural oscillations
        for oscillation in NeuralOscillation:
            name = f"neural_{oscillation.name.lower()}"
            
            # Set frequency based on band
            if oscillation == NeuralOscillation.DELTA:
                freq = 2.0  # 0.5-4 Hz, using middle value
            elif oscillation == NeuralOscillation.THETA:
                freq = 6.0  # 4-8 Hz
            elif oscillation == NeuralOscillation.ALPHA:
                freq = 10.0  # 8-13 Hz
            elif oscillation == NeuralOscillation.BETA:
                freq = 20.0  # 13-30 Hz
            else:  # GAMMA
                freq = 40.0  # 30-100 Hz
            
            # Randomize starting phase
            random_phase = (hash(name) % 1000) / 1000.0 * 2 * math.pi
            
            self.oscillators[name] = OscillatorState(
                type_id=name,
                frequency=freq,
                phase=random_phase,
                amplitude=1.0,
                baseline=0.0,
                last_update=now
            )
    
    def update(self, current_time: Optional[float] = None) -> None:
        """
        Update all biological rhythms based on elapsed time
        """
        if current_time is None:
            current_time = time.time()
            
        # Calculate time elapsed since last update
        if self.use_realtime:
            elapsed = current_time - self.last_update
        else:
            elapsed = (current_time - self.last_update) * self.simulation_time_factor
            self.simulation_time_elapsed += elapsed
        
        # Skip if no meaningful time has passed
        if elapsed < 0.001:
            return
            
        # Update each oscillator
        for osc_id, oscillator in self.oscillators.items():
            oscillator.get_value(current_time)
        
        # Update circadian phase
        self._update_circadian_phase(current_time)
        
        # Update body metrics
        self._update_body_metrics()
        
        # Store last update time
        self.last_update = current_time
    
    def _update_circadian_phase(self, current_time: float) -> None:
        """Update the current circadian phase"""
        # Get current circadian value (ranges from -1 to 1)
        circadian_val = self.oscillators["circadian"].get_value(current_time)
        
        # Get time of day if using realtime
        if self.use_realtime:
            # Get the hour of the day (0-23)
            hour = datetime.datetime.fromtimestamp(current_time).hour
        else:
            # Calculate the hour in the simulation
            hour_offset = (self.simulation_time_elapsed / 3600) % 24
            hour = int(hour_offset)
        
        # Determine the phase based on hour and circadian value
        if self.use_realtime:
            if 4 <= hour < 8:
                self.current_phase = CircadianPhase.EARLY_MORNING
            elif 8 <= hour < 12:
                self.current_phase = CircadianPhase.MORNING
            elif 12 <= hour < 16:
                self.current_phase = CircadianPhase.AFTERNOON
            elif 16 <= hour < 20:
                self.current_phase = CircadianPhase.EVENING
            elif 20 <= hour < 24:
                self.current_phase = CircadianPhase.NIGHT
            else:  # 0-4
                self.current_phase = CircadianPhase.DEEP_NIGHT
        else:
            # When not using realtime, use the circadian oscillator value
            # to determine the phase more directly
            if circadian_val > 0.8:
                self.current_phase = CircadianPhase.MORNING
            elif circadian_val > 0.3:
                self.current_phase = CircadianPhase.AFTERNOON
            elif circadian_val > -0.3:
                self.current_phase = CircadianPhase.EVENING
            elif circadian_val > -0.8:
                self.current_phase = CircadianPhase.NIGHT
            else:
                self.current_phase = CircadianPhase.DEEP_NIGHT
    
    def _update_body_metrics(self) -> None:
        """Update simulated body metrics based on circadian phase"""
        # Get current circadian value (ranges from -1 to 1)
        circadian_val = self.oscillators["circadian"].get_value()
        
        # Core body temperature varies by about 1°C through the day
        # Lowest in early morning, highest in late afternoon/evening
        self.core_temp = 36.3 + 0.5 * (circadian_val + 1.0)
        
        # Cortisol peaks in the morning, lowest at night
        if self.current_phase == CircadianPhase.EARLY_MORNING:
            target_cortisol = 0.9
        elif self.current_phase == CircadianPhase.MORNING:
            target_cortisol = 0.8
        elif self.current_phase == CircadianPhase.AFTERNOON:
            target_cortisol = 0.5
        elif self.current_phase == CircadianPhase.EVENING:
            target_cortisol = 0.3
        else:  # NIGHT/DEEP_NIGHT
            target_cortisol = 0.1
            
        # Smooth transition for cortisol
        self.cortisol_level += (target_cortisol - self.cortisol_level) * 0.1
        
        # Melatonin rises in the evening, peaks at night, low during day
        if self.current_phase in [CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT]:
            target_melatonin = 0.9
        elif self.current_phase == CircadianPhase.EVENING:
            target_melatonin = 0.6
        elif self.current_phase == CircadianPhase.EARLY_MORNING:
            target_melatonin = 0.3
        else:  # MORNING/AFTERNOON
            target_melatonin = 0.1
            
        # Smooth transition for melatonin
        self.melatonin_level += (target_melatonin - self.melatonin_level) * 0.1
    
    def entrain(self, 
               factor_name: str, 
               value: float,
               strength: float = 1.0) -> None:
        """
        Entrain the biological clock with an external cue
        
        Args:
            factor_name: The entrainment factor ("light", "activity", etc.)
            value: The value of the factor (0.0-1.0)
            strength: How strongly to apply the entrainment (0.0-1.0)
        """
        if factor_name not in self.entrainment_factors:
            raise ValueError(f"Unknown entrainment factor: {factor_name}")
            
        # Update the factor
        old_value = self.entrainment_factors[factor_name]
        new_value = old_value + (value - old_value) * strength
        self.entrainment_factors[factor_name] = max(0.0, min(1.0, new_value))
        
        # Apply entrainment effects
        if factor_name == "light":
            # Light primarily affects the circadian oscillator
            # Strong light during subjective night advances the phase
            if self.current_phase in [CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT]:
                if value > 0.7:  # Bright light at night
                    # Phase advance (shift earlier)
                    phase_shift = -0.1 * strength
                    self._shift_oscillator_phase("circadian", phase_shift)
            
            # Light during subjective day strengthens the rhythm
            elif self.current_phase in [CircadianPhase.MORNING, CircadianPhase.AFTERNOON]:
                if value > 0.5:  # Bright daytime light
                    # Increase amplitude
                    self.oscillators["circadian"].amplitude = min(
                        1.2, self.oscillators["circadian"].amplitude + 0.01 * strength
                    )
        
        elif factor_name == "activity":
            # Activity can shift the circadian rhythm
            if value > 0.7:  # High activity
                if self.current_phase in [CircadianPhase.EVENING, CircadianPhase.NIGHT]:
                    # Phase delay (shift later)
                    phase_shift = 0.05 * strength
                    self._shift_oscillator_phase("circadian", phase_shift)
        
        elif factor_name == "meal_timing":
            # Meals primarily affect metabolic oscillators
            if value > 0.7:  # Significant meal
                # Reset the ultradian alertness cycle
                self._shift_oscillator_phase("ultradian_alertness", 0.0)  # Reset to start
    
    def _shift_oscillator_phase(self, oscillator_id: str, phase_shift: float) -> None:
        """Shift the phase of an oscillator"""
        if oscillator_id not in self.oscillators:
            return
            
        # Apply the phase shift (in radians)
        self.oscillators[oscillator_id].phase = (
            self.oscillators[oscillator_id].phase + phase_shift
        ) % (2 * math.pi)
    
    def get_oscillator_value(self, oscillator_id: str) -> float:
        """Get the current value of an oscillator"""
        if oscillator_id not in self.oscillators:
            raise ValueError(f"Unknown oscillator: {oscillator_id}")
            
        return self.oscillators[oscillator_id].get_value()
    
    def get_alertness_level(self) -> float:
        """Get the current alertness level (0.0-1.0)"""
        # Alertness is influenced by circadian rhythm and ultradian cycles
        circadian_val = self.oscillators["circadian"].get_value()
        
        # Normalize from -1,1 to 0,1
        circadian_component = (circadian_val + 1.0) / 2.0
        
        # Adjust based on circadian phase
        if self.current_phase == CircadianPhase.MORNING:
            circadian_component *= 1.2  # Boost morning alertness
        elif self.current_phase == CircadianPhase.AFTERNOON:
            circadian_component *= 0.8  # Post-lunch dip
        elif self.current_phase in [CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT]:
            circadian_component *= 0.6  # Lower at night
            
        # Include ultradian component
        ultradian_val = self.oscillators["ultradian_alertness"].get_value()
        ultradian_component = (ultradian_val + 1.0) / 2.0
        
        # Final value: 70% circadian, 30% ultradian
        alertness = (0.7 * circadian_component) + (0.3 * ultradian_component)
        
        # Apply additional influences
        alertness += 0.2 * self.entrainment_factors["activity"]  # Activity increases alertness
        alertness -= 0.1 * self.melatonin_level  # Melatonin decreases alertness
        
        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, alertness))
    
    def suggest_attentional_state(self) -> AttentionalState:
        """Suggest an attentional state based on biological state"""
        alertness = self.get_alertness_level()
        
        # Determine state based on alertness
        if alertness > 0.8:
            return AttentionalState.ALERT
        elif alertness > 0.6:
            return AttentionalState.NORMAL
        elif alertness > 0.4:
            return AttentionalState.DISTRACTED
        else:
            return AttentionalState.BORED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "circadian_period": self.circadian_period,
            "phase_shift": self.phase_shift,
            "use_realtime": self.use_realtime,
            "current_phase": self.current_phase.name,
            "last_update": self.last_update,
            "simulation_time_factor": self.simulation_time_factor,
            "simulation_time_elapsed": self.simulation_time_elapsed,
            "oscillators": {
                osc_id: oscillator.to_dict() for osc_id, oscillator in self.oscillators.items()
            },
            "entrainment_factors": self.entrainment_factors,
            "core_temp": self.core_temp,
            "cortisol_level": self.cortisol_level,
            "melatonin_level": self.melatonin_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiologicalClock':
        """Recreate from dictionary"""
        instance = cls(
            circadian_period=data["circadian_period"],
            phase_shift=data["phase_shift"],
            use_realtime=data["use_realtime"]
        )
        
        instance.current_phase = CircadianPhase[data["current_phase"]]
        instance.last_update = data["last_update"]
        instance.simulation_time_factor = data["simulation_time_factor"]
        instance.simulation_time_elapsed = data["simulation_time_elapsed"]
        
        # Restore oscillators
        instance.oscillators = {}
        for osc_id, osc_data in data["oscillators"].items():
            instance.oscillators[osc_id] = OscillatorState.from_dict(osc_data)
            
        instance.entrainment_factors = data["entrainment_factors"]
        instance.core_temp = data["core_temp"]
        instance.cortisol_level = data["cortisol_level"]
        instance.melatonin_level = data["melatonin_level"]
        
        return instance


class TimePerception:
    """
    Manages the subjective experience of time, including attentional
    effects, emotional modulation, and the creation of subjective duration.
    """
    
    def __init__(self):
        # Current state
        self.state = TimePerceptionState()
        
        # State history
        self.state_history = []
        
        # Event records for analyzing subjective time
        self.events = {}
        
        # Attention tracking
        self.attention_focus = None  # Current object of attention
        self.attention_level = 0.5  # Current level of attention (0.0-1.0)
        self.attention_history = []  # Record of attention shifts
        
        # Initialize with starting system time
        self.initialization_time = time.time()
        self.last_update_time = self.initialization_time
    
    def update(self, 
              current_time: Optional[float] = None,
              attentional_state: Optional[AttentionalState] = None,
              factors: Optional[Dict[str, float]] = None) -> None:
        """
        Update the time perception state
        
        Args:
            current_time: Current system time (None for automatic)
            attentional_state: New attentional state (if changing)
            factors: Additional factors affecting time perception
        """
        if current_time is None:
            current_time = time.time()
            
        # Record previous state if significant time has passed
        if len(self.state_history) == 0 or current_time - self.last_update_time > 600:
            self.state_history.append({
                "time": self.last_update_time,
                "state": self.state.to_dict()
            })
            
            # Limit history length
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
        
        # Update the state
        self.state.update(attentional_state, factors)
        self.last_update_time = current_time
    
    def record_event_start(self, 
                          event_type: str,
                          content: Any = None,
                          tags: Optional[Set[str]] = None,
                          importance: float = 0.5) -> str:
        """
        Record the start of an event for duration tracking
        
        Returns:
            Event ID for later reference
        """
        event_id = str(uuid.uuid4())
        
        # Create the event
        event = TemporalEvent(
            event_id=event_id,
            start_time=TimePoint.now(),
            event_type=event_type,
            memory_strength=1.0,
            importance=importance,
            tags=tags or set(),
            content=content
        )
        
        # Store the event
        self.events[event_id] = event
        
        return event_id
    
    def record_event_end(self, event_id: str) -> Optional[float]:
        """
        Record the end of an event and calculate subjective duration
        
        Returns:
            Subjective duration in seconds, or None if event not found
        """
        if event_id not in self.events:
            return None
            
        event = self.events[event_id]
        end_time = TimePoint.now()
        
        # Finalize the event
        event.finalize(end_time)
        
        # Calculate objective duration
        objective_duration = event.duration
        
        # Calculate subjective duration based on current state
        subjective_duration = self.state.get_subjective_duration(objective_duration)
        
        # Apply additional modifiers based on event importance
        importance_factor = 1.0 - (event.importance * 0.3)  # 0.7-1.0
        subjective_duration *= importance_factor
        
        # Store subjective duration
        event.subjective_duration = subjective_duration
        
        return subjective_duration
    
    def get_subjective_duration(self, objective_duration: float) -> float:
        """
        Convert an objective duration to a subjective duration
        
        Args:
            objective_duration: Duration in seconds
            
        Returns:
            Subjective duration in seconds
        """
        return self.state.get_subjective_duration(objective_duration)
    
    def get_objective_duration(self, subjective_duration: float) -> float:
        """
        Convert a subjective duration to an objective duration
        
        Args:
            subjective_duration: Subjective duration in seconds
            
        Returns:
            Objective duration in seconds
        """
        return self.state.get_objective_duration(subjective_duration)
    
    def simulate_duration_perception(self, 
                                   objective_duration: float,
                                   attention_level: float = 0.5,
                                   importance: float = 0.5,
                                   novelty: float = 0.5) -> float:
        """
        Simulate how a duration would be perceived under given conditions
        
        Args:
            objective_duration: Actual duration in seconds
            attention_level: Level of attention (0.0-1.0)
            importance: Importance of the activity (0.0-1.0)
            novelty: How novel the experience is (0.0-1.0)
            
        Returns:
            Simulated subjective duration in seconds
        """
        # Base conversion with current state
        base_subjective = self.state.get_subjective_duration(objective_duration)
        
        # Apply attention modifier
        attention_factor = 1.0
        if attention_level > 0.8:  # High attention (flow)
            attention_factor = 0.7  # Time flies when focused
        elif attention_level < 0.3:  # Low attention (boredom)
            attention_factor = 1.5  # Time drags when bored
            
        # Apply importance modifier
        importance_factor = 1.0 - (importance * 0.3)  # 0.7-1.0
        
        # Apply novelty modifier
        novelty_factor = 1.0 - (novelty * 0.2)  # 0.8-1.0 (novel experiences feel shorter)
        
        # Combine factors
        return base_subjective * attention_factor * importance_factor * novelty_factor
    
    def shift_attention(self, 
                       new_focus: Optional[Any] = None,
                       attention_level: Optional[float] = None) -> None:
        """
        Record a shift in attention
        
        Args:
            new_focus: New object of attention
            attention_level: New level of attention (0.0-1.0)
        """
        # Record time of shift
        shift_time = time.time()
        
        # Calculate attentional state if attention level provided
        new_attentional_state = None
        if attention_level is not None:
            self.attention_level = attention_level
            
            if attention_level > 0.8:
                new_attentional_state = AttentionalState.FLOW
            elif attention_level > 0.6:
                new_attentional_state = AttentionalState.ALERT
            elif attention_level > 0.4:
                new_attentional_state = AttentionalState.NORMAL
            elif attention_level > 0.2:
                new_attentional_state = AttentionalState.DISTRACTED
            else:
                new_attentional_state = AttentionalState.BORED
        
        # Record the shift
        self.attention_history.append({
            "time": shift_time,
            "previous_focus": self.attention_focus,
            "new_focus": new_focus,
            "attention_level": self.attention_level
        })
        
        # Update current focus
        self.attention_focus = new_focus
        
        # Limit history length
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
        
        # Update time perception state if attentional state changed
        if new_attentional_state is not None:
            self.update(shift_time, new_attentional_state)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "state": self.state.to_dict(),
            "state_history": self.state_history,
            "events": {
                event_id: event.to_dict() for event_id, event in self.events.items()
            },
            "attention_focus": str(self.attention_focus) if self.attention_focus is not None else None,
            "attention_level": self.attention_level,
            "attention_history": self.attention_history,
            "initialization_time": self.initialization_time,
            "last_update_time": self.last_update_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimePerception':
        """Recreate from dictionary"""
        instance = cls()
        
        instance.state = TimePerceptionState.from_dict(data["state"])
        instance.state_history = data["state_history"]
        
        # Restore events
        instance.events = {}
        for event_id, event_data in data["events"].items():
            instance.events[event_id] = TemporalEvent.from_dict(event_data)
            
        instance.attention_focus = data["attention_focus"]
        instance.attention_level = data["attention_level"]
        instance.attention_history = data["attention_history"]
        instance.initialization_time = data["initialization_time"]
        instance.last_update_time = data["last_update_time"]
        
        return instance


class TemporalCoherence:
    """
    Main controller class for the temporal coherence system.
    Coordinates all time-related subsystems and provides a coherent
    interface for the GPT4ø to develop a sophisticated sense of time.
    """
    
    def __init__(self, 
                base_path: Optional[Path] = None, 
                auto_save: bool = True,
                auto_update_interval: float = 60.0):
        """
        Initialize the temporal coherence system
        
        Args:
            base_path: Path for storing persisted data
            auto_save: Whether to automatically save state periodically
            auto_update_interval: How often to auto-update (seconds)
        """
        # Set up base path
        if base_path is None:
            self.base_path = Path.home() / ".gpt4o" / "temporal_coherence"
        else:
            self.base_path = Path(base_path)
            
        # Create directory if needed
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self.scientific_timekeeper = ScientificTimekeeper()
        self.biological_clock = BiologicalClock()
        self.time_perception = TimePerception()
        self.memory = TemporalMemory()
        
        # State tracking
        self.creation_time = time.time()
        self.last_update_time = self.creation_time
        self.auto_save = auto_save
        self.auto_update_interval = auto_update_interval
        self.last_save_time = self.creation_time
        self.save_interval = 1800.0  # 30 minutes
        
        # Start background threads if auto-update enabled
        self.update_task = None
        self.running = False
        
        if auto_update_interval > 0:
            self.start_background_updates()
    
    def start_background_updates(self) -> None:
        """Start background update task"""
        if self.update_task is not None and not self.update_task.done():
            return
            
        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
    
    def stop_background_updates(self) -> None:
        """Stop background update task"""
        self.running = False
        if self.update_task is not None:
            self.update_task.cancel()
            # We don't await the task here to avoid blocking
            self.update_task = None
    
    async def _update_loop(self) -> None:
        """Background update loop"""
        while self.running:
            try:
                # Update all systems
                self.update()
                
                # Auto-save if needed
                if self.auto_save and time.time() - self.last_save_time > self.save_interval:
                    self.save_state()
                    
                # Sleep until next update
                await asyncio.sleep(self.auto_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in temporal coherence update loop: {e}")
                # Sleep to avoid tight loop on error
                await asyncio.sleep(5.0)
    
    def update(self, current_time: Optional[float] = None) -> None:
        """
        Update all subsystems
        
        Args:
            current_time: Current system time (None for automatic)
        """
        if current_time is None:
            current_time = time.time()
            
        # Update all subsystems
        self.biological_clock.update(current_time)
        
        # Get suggested attentional state from biological clock
        suggested_state = self.biological_clock.suggest_attentional_state()
        
        # Update time perception with suggested state
        self.time_perception.update(
            current_time=current_time, 
            attentional_state=suggested_state
        )
        
        # Record last update time
        self.last_update_time = current_time
    
    def now(self) -> TimePoint:
        """Get a TimePoint for the current moment"""
        return self.scientific_timekeeper.get_current_timepoint()
    
    def get_elapsed_time(self, reference_name: Optional[str] = None) -> float:
        """
        Get elapsed time since a reference point in seconds
        
        Args:
            reference_name: Named reference point (None for initialization)
            
        Returns:
            Elapsed time in seconds
        """
        return self.scientific_timekeeper.get_elapsed_time(reference_name)
    
    def get_circadian_phase(self) -> CircadianPhase:
        """Get the current circadian phase"""
        # Make sure biological clock is up to date
        self.biological_clock.update()
        return self.biological_clock.current_phase
    
    def get_alertness(self) -> float:
        """Get current alertness level (0.0-1.0)"""
        # Make sure biological clock is up to date
        self.biological_clock.update()
        return self.biological_clock.get_alertness_level()
    
    def begin_event(self, 
                  event_type: str,
                  content: Any = None,
                  tags: Optional[Set[str]] = None,
                  importance: float = 0.5) -> str:
        """
        Begin tracking a temporal event
        
        Args:
            event_type: Type of event
            content: Event content/details
            tags: Tags for categorization
            importance: Event importance (0.0-1.0)
            
        Returns:
            Event ID for later reference
        """
        event_id = self.time_perception.record_event_start(
            event_type=event_type,
            content=content,
            tags=tags,
            importance=importance
        )
        
        return event_id
    
    def end_event(self, event_id: str) -> Dict[str, Any]:
        """
        End tracking a temporal event and get duration information
        
        Args:
            event_id: ID from begin_event
            
        Returns:
            Information about the event
        """
        # End the event in time perception
        subjective_duration = self.time_perception.record_event_end(event_id)
        
        if event_id not in self.time_perception.events:
            return {"error": "Event not found"}
            
        event = self.time_perception.events[event_id]
        
        # Add to memory
        self.memory.add_event(event)
        
        # Return event information
        return {
            "event_id": event_id,
            "event_type": event.event_type,
            "start_time": event.start_time.system_time,
            "end_time": event.end_time.system_time if event.end_time else None,
            "objective_duration": event.duration,
            "subjective_duration": subjective_duration,
            "importance": event.importance,
            "tags": list(event.tags)
        }
    
    def get_subjective_time(self, objective_time: float) -> float:
        """
        Convert objective time to subjective time
        
        Args:
            objective_time: Time in seconds
            
        Returns:
            Subjective time in seconds
        """
        return self.time_perception.get_subjective_duration(objective_time)
    
    def get_objective_time(self, subjective_time: float) -> float:
        """
        Convert subjective time to objective time
        
        Args:
            subjective_time: Subjective time in seconds
            
        Returns:
            Objective time in seconds
        """
        return self.time_perception.get_objective_duration(subjective_time)
    
    def set_attention(self, 
                     focus: Any = None,
                     level: float = 0.5) -> None:
        """
        Set attention focus and level
        
        Args:
            focus: Object of attention
            level: Attention level (0.0-1.0)
        """
        self.time_perception.shift_attention(focus, level)
    
    def entrain_biological_clock(self, 
                               factor: str, 
                               value: float,
                               strength: float = 1.0) -> None:
        """
        Entrain the biological clock with an external cue
        
        Args:
            factor: The entrainment factor ("light", "activity", etc.)
            value: The value of the factor (0.0-1.0)
            strength: How strongly to apply the entrainment (0.0-1.0)
        """
        self.biological_clock.entrain(factor, value, strength)
    
    def recall_events(self, 
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    event_type: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    min_memory_strength: float = 0.1,
                    importance_threshold: float = 0.0,
                    max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Recall events from memory
        
        Args:
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            event_type: Filter by event type
            tags: Filter by tags
            min_memory_strength: Minimum memory strength (0.0-1.0)
            importance_threshold: Minimum importance (0.0-1.0)
            max_results: Maximum number of results
            
        Returns:
            List of event dictionaries
        """
        events = self.memory.find_events(
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            tags=tags,
            min_memory_strength=min_memory_strength,
            importance_threshold=importance_threshold,
            max_results=max_results
        )
        
        # Convert to dictionaries
        event_dicts = []
        for event in events:
            event_dict = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "start_time": event.start_time.system_time,
                "end_time": event.end_time.system_time if event.end_time else None,
                "objective_duration": event.duration,
                "subjective_duration": event.subjective_duration,
                "importance": event.importance,
                "memory_strength": event.get_current_memory_strength(),
                "tags": list(event.tags),
                "content": event.content
            }
            event_dicts.append(event_dict)
            
        return event_dicts
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the temporal coherence system
        
        Returns:
            Dictionary with status information
        """
        now = time.time()
        
        # Ensure systems are up to date
        self.update(now)
        
        status = {
            "current_time": {
                "system_time": now,
                "system_datetime": datetime.datetime.fromtimestamp(now).isoformat(),
                "elapsed_since_creation": now - self.creation_time
            },
            "scientific_time": {
                "reference_points": list(self.scientific_timekeeper.reference_points.keys())
            },
            "biological_time": {
                "circadian_phase": self.biological_clock.current_phase.name,
                "alertness": self.biological_clock.get_alertness_level(),
                "core_temperature": self.biological_clock.core_temp,
                "cortisol_level": self.biological_clock.cortisol_level,
                "melatonin_level": self.biological_clock.melatonin_level
            },
            "time_perception": {
                "attentional_state": self.time_perception.state.attentional_state.name,
                "subjective_scalar": self.time_perception.state.subjective_scalar,
                "attention_focus": str(self.time_perception.attention_focus) if self.time_perception.attention_focus is not None else None,
                "attention_level": self.time_perception.attention_level
            },
            "memory": self.memory.get_memory_stats()
        }
        
        return status
    
    def save_state(self, path: Optional[Path] = None) -> str:
        """
        Save the current state to disk
        
        Args:
            path: Path to save to (None for default)
            
        Returns:
            Path where state was saved
        """
        if path is None:
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temporal_coherence_state_{timestamp}.json"
            path = self.base_path / filename
        
        # Make sure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare the state
        state = {
            "version": "1.0.0",
            "timestamp": time.time(),
            "creation_time": self.creation_time,
            "scientific_timekeeper": self.scientific_timekeeper.to_dict(),
            "biological_clock": self.biological_clock.to_dict(),
            "time_perception": self.time_perception.to_dict(),
            "memory_stats": self.memory.get_memory_stats()
        }
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
        # Update last save time
        self.last_save_time = time.time()
        
        return str(path)
    
    def load_state(self, path: Union[str, Path]) -> bool:
        """
        Load state from disk
        
        Args:
            path: Path to load from
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        
        if not path.exists():
            logger.error(f"State file not found: {path}")
            return False
            
        try:
            # Load the state
            with open(path, 'r') as f:
                state = json.load(f)
                
            # Check version compatibility
            version = state.get("version", "unknown")
            if version != "1.0.0":
                logger.warning(f"State file version mismatch: expected 1.0.0, got {version}")
                
            # Restore subsystems
            self.scientific_timekeeper = ScientificTimekeeper.from_dict(state["scientific_timekeeper"])
            self.biological_clock = BiologicalClock.from_dict(state["biological_clock"])
            self.time_perception = TimePerception.from_dict(state["time_perception"])
            
            # Restore creation time
            self.creation_time = state.get("creation_time", time.time())
            self.last_update_time = time.time()
            self.last_save_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        # Stop background threads
        self.stop_background_updates()
        
        # Final auto-save if enabled
        if self.auto_save:
            try:
                self.save_state()
            except Exception as e:
                logger.error(f"Error during final save: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def format_duration(seconds: float, precision: int = 1) -> str:
    """
    Format a duration in seconds to a human-readable string
    
    Args:
        seconds: Duration in seconds
        precision: Decimal precision for seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return f"-{format_duration(-seconds, precision)}"
    
    if seconds < 1:
        # Milliseconds
        ms = seconds * 1000
        return f"{ms:.{precision}f} ms"
    
    if seconds < 60:
        # Seconds
        return f"{seconds:.{precision}f} s"
    
    if seconds < 3600:
        # Minutes and seconds
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.{precision}f}s"
    
    if seconds < 86400:
        # Hours, minutes, and seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.{precision}f}s"
    
    # Days, hours, minutes, and seconds
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if days == 1:
        return f"{days} day, {hours}h {minutes}m"
    else:
        return f"{days} days, {hours}h {minutes}m"


def convert_time_scale(value: float, 
                     from_scale: TimeScale,
                     to_scale: TimeScale) -> float:
    """
    Convert a value from one time scale to another
    
    Args:
        value: The value to convert
        from_scale: Source time scale
        to_scale: Target time scale
        
    Returns:
        Converted value
    """
    # Convert to seconds first
    if from_scale == TimeScale.QUANTUM:
        seconds = value * PLANCK_TIME
    elif from_scale == TimeScale.ATOMIC:
        seconds = value / ATOMIC_SECOND
    elif from_scale == TimeScale.NEURAL:
        seconds = value / 1000  # milliseconds to seconds
    elif from_scale == TimeScale.GEOLOGICAL:
        seconds = value * SECONDS_PER_YEAR * 100  # centuries to seconds
    elif from_scale == TimeScale.COSMOLOGICAL:
        seconds = value * SECONDS_PER_YEAR * 1e6  # million years to seconds
    else:
        # PHYSIOLOGICAL, CIRCADIAN, CALENDAR scales use seconds
        seconds = value
    
    # Convert seconds to target scale
    if to_scale == TimeScale.QUANTUM:
        return seconds / PLANCK_TIME
    elif to_scale == TimeScale.ATOMIC:
        return seconds * ATOMIC_SECOND
    elif to_scale == TimeScale.NEURAL:
        return seconds * 1000  # seconds to milliseconds
    elif to_scale == TimeScale.GEOLOGICAL:
        return seconds / (SECONDS_PER_YEAR * 100)  # seconds to centuries
    elif to_scale == TimeScale.COSMOLOGICAL:
        return seconds / (SECONDS_PER_YEAR * 1e6)  # seconds to million years
    else:
        # PHYSIOLOGICAL, CIRCADIAN, CALENDAR scales use seconds
        return seconds


def get_weekday_from_timestamp(timestamp: float) -> str:
    """
    Get day of week from Unix timestamp
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Day of week (Monday, Tuesday, etc.)
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime("%A")


def get_time_of_day(timestamp: float) -> str:
    """
    Get time of day description from Unix timestamp
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Time of day description (morning, afternoon, etc.)
    """
    hour = datetime.datetime.fromtimestamp(timestamp).hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_moon_phase(date: Union[datetime.date, datetime.datetime, None] = None) -> Tuple[str, float]:
    """
    Calculate the moon phase for a given date
    
    Args:
        date: Date to calculate for (None for today)
        
    Returns:
        Tuple of (phase_name, phase_fraction)
    """
    if date is None:
        date = datetime.date.today()
    elif isinstance(date, datetime.datetime):
        date = date.date()
    
    # Convert date to Julian date
    year, month, day = date.year, date.month, date.day
    
    if month < 3:
        year -= 1
        month += 12
        
    a = year // 100
    b = 2 - a + (a // 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    
    # Calculate days since known new moon (Jan 6, 2000)
    days_since = jd - 2451550.1
    
    # Calculate the phase fraction (0.0 to 1.0)
    phase_fraction = (days_since % 29.53) / 29.53
    
    # Determine the phase name
    if phase_fraction < 0.03:
        phase_name = "New Moon"
    elif phase_fraction < 0.22:
        phase_name = "Waxing Crescent"
    elif phase_fraction < 0.28:
        phase_name = "First Quarter"
    elif phase_fraction < 0.47:
        phase_name = "Waxing Gibbous"
    elif phase_fraction < 0.53:
        phase_name = "Full Moon"
    elif phase_fraction < 0.72:
        phase_name = "Waning Gibbous"
    elif phase_fraction < 0.78:
        phase_name = "Last Quarter"
    elif phase_fraction < 0.97:
        phase_name = "Waning Crescent"
    else:
        phase_name = "New Moon"
        
    return (phase_name, phase_fraction)


def get_season(date: Union[datetime.date, datetime.datetime, None] = None, 
             hemisphere: str = "northern") -> str:
    """
    Determine the season for a given date
    
    Args:
        date: Date to check (None for today)
        hemisphere: "northern" or "southern"
        
    Returns:
        Season name
    """
    if date is None:
        date = datetime.date.today()
    elif isinstance(date, datetime.datetime):
        date = date.date()
        
    month, day = date.month, date.day
    
    # Simple algorithm based on astronomical seasons
    if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day < 21):
        season = "Spring" if hemisphere == "northern" else "Autumn"
    elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 22):
        season = "Summer" if hemisphere == "northern" else "Winter"
    elif (month == 9 and day >= 22) or (month == 10) or (month == 11) or (month == 12 and day < 21):
        season = "Autumn" if hemisphere == "northern" else "Spring"
    else:
        season = "Winter" if hemisphere == "northern" else "Summer"
        
    return season