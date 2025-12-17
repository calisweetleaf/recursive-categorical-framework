import logging
import time
import numpy as np
from rcf_integration.rsgt.motivation_system import (
    Experience, 
    Value, 
    ValueFormationSystem, 
    ValueConflictResolution,
    EmergentMotivationSystem
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MotivationTest")

def test_vector_determinism():
    logger.info("Testing Vector Determinism...")
    exp1 = Experience(
        source="test_source",
        content={"key": "value", "data": 123},
        valence=0.8,
        intensity=0.7
    )
    
    exp2 = Experience(
        source="test_source",
        content={"key": "value", "data": 123},
        valence=0.8,
        intensity=0.7
    )
    
    vec1 = exp1.get_vector_representation()
    vec2 = exp2.get_vector_representation()
    
    if np.array_equal(vec1, vec2):
        logger.info("PASS: Vectors are deterministic.")
    else:
        logger.error("FAIL: Vectors are NOT deterministic.")
        
def test_tension_calculation():
    logger.info("Testing Tension Calculation...")
    
    # Create two conflicting values
    # Value 1: High intensity, specific features
    val1 = Value(
        id="val1",
        description="Value 1",
        intensity=0.9,
        stability=0.8,
        features={"feature_a": 1.0, "feature_b": 0.0}
    )
    
    # Value 2: High intensity, opposing features
    val2 = Value(
        id="val2",
        description="Value 2",
        intensity=0.9,
        stability=0.8,
        features={"feature_a": 0.0, "feature_b": 1.0}
    )
    
    resolver = ValueConflictResolution()
    tension = resolver._calculate_conflict_potential(val1, val2)
    
    logger.info(f"Calculated Tension: {tension}")
    
    if tension > 0.5:
        logger.info("PASS: High tension detected for conflicting values.")
    else:
        logger.warning(f"FAIL: Tension too low for conflicting values: {tension}")

def test_weight_dynamics():
    logger.info("Testing Weight Dynamics...")
    
    val = Value(
        id="val_weight",
        description="Dynamic Value",
        intensity=0.8,
        stability=0.5,
        features={"context_key": 1.0}
    )
    
    # Context vector that aligns with value features
    # (Simplified alignment for test)
    # In the implementation, we use random projection based on hash
    # So we can't easily manually construct a matching vector without using the same logic
    # But we can check if weight varies with time
    
    t1 = time.time()
    w1 = val.calculate_weight(np.zeros(128), t1)
    
    # Simulate time passing
    val.last_update_time = t1 - 100 # 100 seconds ago
    t2 = time.time()
    w2 = val.calculate_weight(np.zeros(128), t2)
    
    logger.info(f"Weight at t1: {w1}")
    logger.info(f"Weight at t2 (after decay): {w2}")
    
    if w2 < w1: # Should decay if stability is not 1.0
        logger.info("PASS: Weight decays over time as expected.")
    else:
        logger.warning("FAIL: Weight did not decay.")

def test_pattern_recognition():
    logger.info("Testing Pattern Recognition...")
    
    system = ValueFormationSystem()
    
    # Create a sequence of similar experiences
    exp_base = Experience(
        source="pattern_source",
        content={"type": "A", "val": 10},
        valence=0.5,
        intensity=0.5
    )
    
    patterns = system._extract_patterns(exp_base)
    logger.info(f"Initial patterns: {len(patterns)}")
    
    # Another similar experience
    exp_sim = Experience(
        source="pattern_source",
        content={"type": "A", "val": 12}, # Slightly different
        valence=0.5,
        intensity=0.5
    )
    
    patterns_2 = system._extract_patterns(exp_sim)
    logger.info(f"Patterns after second experience: {len(patterns_2)}")
    
    # Should have updated the existing pattern or created a new one that gets merged
    # Check recurrence of patterns
    total_recurrence = sum(p.recurrence for p in system.patterns.values())
    logger.info(f"Total pattern recurrence: {total_recurrence}")
    
    if total_recurrence >= 2:
        logger.info("PASS: Pattern recurrence increased.")
    else:
        logger.warning("FAIL: Pattern recurrence did not increase.")

if __name__ == "__main__":
    test_vector_determinism()
    test_tension_calculation()
    test_weight_dynamics()
    test_pattern_recognition()
