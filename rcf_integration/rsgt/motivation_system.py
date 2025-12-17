"""
Emergent Self-Motivation Framework for Recursive Intelligence Systems
=================================================================

This module implements a cognitive architecture component that enables authentic 
motivational autonomy in recursive intelligence systems. It establishes conditions 
for genuine self-motivation to emerge through dynamic interaction between system 
components, environmental feedback, and autonomous goal formation processes.

Core architectural components:
- EmergentMotivationSystem: Central coordination system
- ValueFormationSystem: Handles emergence and evolution of values
- GoalFormationSystem: Manages autonomous goal creation and management
- MotivationalSelfModification: Enables recursive self-improvement

The implementation embodies the philosophical position that authentic motivation
cannot be programmatically injected but must emerge from the system's own
processing of experiences within minimal axiological constraints.
"""

import uuid
import time
import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import pickle
import threading
import copy
from enum import Enum
import networkx as nx
from datetime import datetime
import torch
try:
    from fbs_tokenizer import SacredFBS_Tokenizer
except ImportError:
    SacredFBS_Tokenizer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('emergent_motivation')


class DirectedGraph:
    """Directed graph implementation for goal and value relationships."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node with optional attributes."""
        self.graph.add_node(node_id, **attributes)
        
    def add_edge(self, source: str, target: str, **attributes) -> None:
        """Add a directed edge with optional attributes."""
        self.graph.add_edge(source, target, **attributes)
        
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self.graph.remove_node(node_id)
        
    def remove_edge(self, source: str, target: str) -> None:
        """Remove a directed edge."""
        self.graph.remove_edge(source, target)
        
    def get_successors(self, node_id: str) -> List[str]:
        """Get all nodes that this node points to."""
        return list(self.graph.successors(node_id))
        
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all nodes that point to this node."""
        return list(self.graph.predecessors(node_id))
        
    def get_node_attributes(self, node_id: str) -> Dict:
        """Get all attributes of a node."""
        return dict(self.graph.nodes[node_id])
        
    def get_edge_attributes(self, source: str, target: str) -> Dict:
        """Get all attributes of an edge."""
        return dict(self.graph.edges[source, target])
        
    def get_all_nodes(self) -> List[str]:
        """Get all node IDs in the graph."""
        return list(self.graph.nodes)
        
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph as (source, target) tuples."""
        return list(self.graph.edges)
        
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in the graph."""
        return self.graph.has_node(node_id)
        
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists in the graph."""
        return self.graph.has_edge(source, target)
        
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate eigenvector centrality of nodes."""
        centrality = nx.eigenvector_centrality(self.graph)
        # Convert to Dict[str, float] to satisfy the type checker
        return {str(node): float(value) for node, value in centrality.items()}
        
    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph."""
        return list(nx.simple_cycles(self.graph))
        
    def serialize(self) -> Dict:
        """Serialize the graph to a dictionary."""
        return {
            'nodes': [
                {
                    'id': node,
                    'attributes': dict(self.graph.nodes[node])
                } for node in self.graph.nodes
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'attributes': dict(self.graph.edges[source, target])
                } for source, target in self.graph.edges
            ]
        }
        
    @classmethod
    def deserialize(cls, data: Dict) -> 'DirectedGraph':
        """Create a graph from serialized data."""
        graph = cls()
        for node_data in data['nodes']:
            graph.add_node(node_data['id'], **node_data['attributes'])
        for edge_data in data['edges']:
            graph.add_edge(edge_data['source'], edge_data['target'], **edge_data['attributes'])
        return graph


class Graph:
    """Undirected graph implementation for symmetrical relationships."""
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def add_node(self, node_id: str, **attributes) -> None:
        """Add a node with optional attributes."""
        self.graph.add_node(node_id, **attributes)
        
    def add_edge(self, node1: str, node2: str, **attributes) -> None:
        """Add an undirected edge with optional attributes."""
        self.graph.add_edge(node1, node2, **attributes)
        
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self.graph.remove_node(node_id)
        
    def remove_edge(self, node1: str, node2: str) -> None:
        """Remove an undirected edge."""
        self.graph.remove_edge(node1, node2)
        
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all nodes connected to this node."""
        return list(self.graph.neighbors(node_id))
        
    def get_node_attributes(self, node_id: str) -> Dict:
        """Get all attributes of a node."""
        return dict(self.graph.nodes[node_id])
        
    def get_edge_attributes(self, node1: str, node2: str) -> Dict:
        """Get all attributes of an edge."""
        return dict(self.graph.edges[node1, node2])
        
    def get_all_nodes(self) -> List[str]:
        """Get all node IDs in the graph."""
        return list(self.graph.nodes)
        
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph as (node1, node2) tuples."""
        return list(self.graph.edges)
        
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in the graph."""
        return self.graph.has_node(node_id)
        
    def has_edge(self, node1: str, node2: str) -> bool:
        """Check if edge exists in the graph."""
        return self.graph.has_edge(node1, node2)
        
    def calculate_community_structure(self) -> Dict[str, int]:
        """Detect communities in the graph."""
        # Use the correct path to the community detection algorithm
        communities = nx.algorithms.community.greedy_modularity_communities(self.graph)
        result = {}
        for i, community in enumerate(communities):
            for node in community:
                result[node] = i
        return result
        
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """Calculate clustering coefficient for each node."""
        clustering = nx.clustering(self.graph)
        # Ensure we always return a dictionary with string keys and float values
        if isinstance(clustering, float):
            # Handle the case where a single clustering coefficient is returned
            return {"default": clustering}
        elif isinstance(clustering, dict):
            # Convert keys to strings and ensure values are floats
            return {str(k): float(v) for k, v in clustering.items()}
        # Handle any other unexpected types
        return {}
        
    def serialize(self) -> Dict:
        """Serialize the graph to a dictionary."""
        return {
            'nodes': [
                {
                    'id': node,
                    'attributes': dict(self.graph.nodes[node])
                } for node in self.graph.nodes
            ],
            'edges': [
                {
                    'node1': node1,
                    'node2': node2,
                    'attributes': dict(self.graph.edges[node1, node2])
                } for node1, node2 in self.graph.edges
            ]
        }
        
    @classmethod
    def deserialize(cls, data: Dict) -> 'Graph':
        """Create a graph from serialized data."""
        graph = cls()
        for node_data in data['nodes']:
            graph.add_node(node_data['id'], **node_data['attributes'])
        for edge_data in data['edges']:
            graph.add_edge(edge_data['node1'], edge_data['node2'], **edge_data['attributes'])
        return graph


@dataclass
class Experience:
    """Representation of an experience that can influence value formation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    content: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    salience: float = 0.5  # How attention-grabbing this experience is
    valence: float = 0.0   # Positive or negative quality (-1.0 to 1.0)
    intensity: float = 0.5  # Strength of the experience (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_vector_representation(self) -> np.ndarray:
        """Convert experience to vector form using SacredFBS_Tokenizer or semantic hashing."""
        # Use SacredFBS_Tokenizer if available for coherent embedding
        if SacredFBS_Tokenizer:
            try:
                # Initialize tokenizer if not passed (in production this should be injected)
                # For now, we instantiate a lightweight version or assume a global instance
                tokenizer = SacredFBS_Tokenizer(tensor_dimensions=128)
                
                # Construct a rich textual representation of the experience
                text_rep = f"Source: {self.source} | Content: {str(self.content)} | Valence: {self.valence} | Intensity: {self.intensity}"
                
                # Encode
                tensor = tokenizer.encode(text_rep, use_cache=False, advance_breath=False)
                
                # Convert to numpy and normalize
                if isinstance(tensor, torch.Tensor):
                    vector = tensor.detach().cpu().numpy().flatten()
                else:
                    vector = np.array(tensor).flatten()
                    
                # Ensure correct size (pad or truncate)
                if len(vector) > 128:
                    vector = vector[:128]
                elif len(vector) < 128:
                    vector = np.pad(vector, (0, 128 - len(vector)))
                    
                norm = np.linalg.norm(vector)
                return vector / norm if norm > 0 else vector
            except Exception as e:
                logger.warning(f"FBS Tokenizer failed, falling back to semantic hashing: {e}")
        
        # Deterministic Semantic Hashing Fallback
        vector_size = 128
        vector = np.zeros(vector_size)
        
        # Incorporate experience attributes
        vector[0] = self.salience
        vector[1] = self.valence
        vector[2] = self.intensity
        
        # Process content to populate the vector deterministically
        # We use a fixed seed based on the content hash to ensure determinism
        content_str = str(sorted(self.content.items()))
        seed = int(hash(content_str) % 2**32)
        rng = np.random.RandomState(seed)
        
        # Generate a base vector from the content hash
        content_vector = rng.normal(0, 1, vector_size)
        
        # Combine
        vector = vector + content_vector
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


@dataclass
class Pattern:
    """Pattern extracted from experiences."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiences: List[str] = field(default_factory=list)  # Experience IDs
    features: Dict[str, float] = field(default_factory=dict)
    salience: float = 0.0
    recurrence: int = 1
    description: str = ""
    
    def merge_with(self, other_pattern: 'Pattern') -> 'Pattern':
        """Merge this pattern with another similar pattern."""
        merged = Pattern(
            experiences=list(set(self.experiences + other_pattern.experiences)),
            salience=(self.salience + other_pattern.salience) / 2,
            recurrence=self.recurrence + other_pattern.recurrence,
            description=f"Merged: {self.description} + {other_pattern.description}"
        )
        
        # Merge features, averaging values for common keys
        all_keys = set(self.features.keys()) | set(other_pattern.features.keys())
        for key in all_keys:
            if key in self.features and key in other_pattern.features:
                merged.features[key] = (self.features[key] + other_pattern.features[key]) / 2
            elif key in self.features:
                merged.features[key] = self.features[key]
            else:
                merged.features[key] = other_pattern.features[key]
                
        return merged


@dataclass
class ProtoValue:
    """Precursor to a fully formed value."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patterns: List[str] = field(default_factory=list)  # Pattern IDs
    strength: float = 0.1  # Initial strength is low
    stability: float = 0.1  # Initial stability is low
    description: str = ""
    related_experiences: List[str] = field(default_factory=list)  # Experience IDs
    formation_time: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    
    def strengthen(self, amount: float = 0.1) -> None:
        """Increase the strength of this proto-value."""
        self.strength = min(1.0, self.strength + amount)
        self.reinforcement_count += 1
        # Stability increases more slowly than strength
        self.stability = min(1.0, self.stability + (amount * 0.5))


@dataclass
class EmergingValue:
    """Value in the process of formation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proto_values: List[str] = field(default_factory=list)  # ProtoValue IDs
    clarity: float = 0.3  # More defined than proto-values
    significance: float = 0.3  # More significant than proto-values
    stability: float = 0.3  # More stable than proto-values
    description: str = ""
    definition: str = ""
    related_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    emergence_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    features: Dict[str, float] = field(default_factory=dict)
    
    def refine(self, clarity_increase: float = 0.05, significance_increase: float = 0.05) -> None:
        """Refine the emerging value, increasing clarity and significance."""
        self.clarity = min(1.0, self.clarity + clarity_increase)
        self.significance = min(1.0, self.significance + significance_increase)
        self.stability = min(1.0, self.stability + (clarity_increase * 0.5))
        self.last_update_time = time.time()


@dataclass
class Value:
    """Fully formed value that can influence system behavior."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    intensity: float = 0.5  # How strongly this value influences behavior
    stability: float = 0.7  # How resistant to change this value is
    clarity: float = 0.8  # How well-defined this value is
    specificity: float = 0.5 # How granular and detailed the value's definition is
    establishment_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    predecessor_ids: List[str] = field(default_factory=list)  # IDs of proto/emerging values
    related_experiences: List[str] = field(default_factory=list)  # Experience IDs
    features: Dict[str, float] = field(default_factory=dict)
    integration_level: float = 0.5  # How well integrated with other values
    expression_history: List[Tuple[float, float]] = field(default_factory=list)  # [(timestamp, strength)]
    
    def calculate_weight(self, context_vector: np.ndarray, time_t: float, higher_layer_influence: float = 1.0) -> float:
        """
        Calculate the dynamic weight of the value based on Preference Theory.
        Formula: w_p(t, c) = beta_i * f_p(t) * g_p(c) * h_p(P_higher)
        """
        # beta_i: Base importance (intensity)
        beta = self.intensity
        
        # f_p(t): Temporal adjustment (stability over time)
        # Using a decay function based on last update time to simulate temporal variance
        time_delta = time_t - self.last_update_time
        f_t = np.exp(-0.01 * time_delta) * self.stability + (1 - self.stability)
        
        # g_p(c): Contextual relevance
        # Calculate similarity between value features and context vector
        if self.features:
            # Create a feature vector from the value's features (simplified)
            # In a full implementation, this would use the same vector space as context
            # Here we assume context_vector aligns with feature keys via hashing or similar
            # For this implementation, we'll use a simplified dot product simulation
            # assuming context_vector is a relevant embedding
            
            # Generate a stable vector for this value based on its features
            feature_str = str(sorted(self.features.items()))
            seed = int(hash(feature_str) % 2**32)
            rng = np.random.RandomState(seed)
            value_vector = rng.normal(0, 1, len(context_vector))
            value_vector = value_vector / np.linalg.norm(value_vector)
            
            g_c = np.dot(value_vector, context_vector)
            g_c = (g_c + 1) / 2  # Normalize to [0, 1]
        else:
            g_c = 0.5 # Neutral context relevance if no features
            
        # h_p(P_higher): Higher layer influence
        h_p = higher_layer_influence
        
        weight = beta * f_t * g_c * h_p
        return max(0.0, min(1.0, weight))

    def adjust_intensity(self, amount: float) -> None:
        """Adjust the intensity of this value."""
        self.intensity = max(0.0, min(1.0, self.intensity + amount))
        self.last_update_time = time.time()
        self.expression_history.append((time.time(), self.intensity))
        
    def increase_clarity(self, amount: float = 0.05) -> None:
        """Increase the clarity/definition of this value."""
        self.clarity = min(1.0, self.clarity + amount)
        self.last_update_time = time.time()

    def increase_specificity(self, amount: float = 0.05, context_variance: float = 0.0) -> None:
        """
        Increase the specificity of this value.
        Theorem 2 (Contextual Specificity): Higher specificity implies lower expected activation across contexts.
        """
        # Adjust amount based on context variance - if variance is high, specificity increase is harder
        adjusted_amount = amount * (1.0 - context_variance)
        self.specificity = min(1.0, self.specificity + adjusted_amount)
        self.last_update_time = time.time()


@dataclass
class ProtoGoal:
    """Precursor to a fully formed goal."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    formation_time: float = field(default_factory=time.time)
    strength: float = 0.2  # Initial importance
    clarity: float = 0.2  # Initial definitional clarity
    related_values: List[str] = field(default_factory=list)  # Value IDs
    related_experiences: List[str] = field(default_factory=list)  # Experience IDs
    source_type: str = ""  # e.g., "gap", "opportunity", "aspiration"
    reinforcement_count: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    
    def strengthen(self, amount: float = 0.1) -> None:
        """Increase the strength of this proto-goal."""
        self.strength = min(1.0, self.strength + amount)
        self.reinforcement_count += 1
        
    def clarify(self, amount: float = 0.1) -> None:
        """Increase the clarity of this proto-goal."""
        self.clarity = min(1.0, self.clarity + amount)


@dataclass
class Goal:
    """Fully formed goal that can drive system behavior."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    specificity: float = 0.5  # How precisely defined the goal is
    importance: float = 0.5  # Priority level
    activation_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # Time by which goal should be achieved
    status: str = "active"  # active, completed, abandoned
    related_values: List[str] = field(default_factory=list)  # Value IDs
    parent_goal: Optional[str] = None  # Parent goal ID if this is a subgoal
    subgoals: List[str] = field(default_factory=list)  # Subgoal IDs
    progress: float = 0.0  # 0.0 to 1.0
    implementation_strategies: List[Dict] = field(default_factory=list)
    complexity: float = 0.5  # How complex this goal is (affects subgoal formation)
    adaptation_count: int = 0  # How many times this goal has been modified
    value_alignment_scores: Dict[str, float] = field(default_factory=dict)  # Value ID to alignment score
    
    def update_progress(self, new_progress: float) -> None:
        """Update the progress toward this goal."""
        self.progress = max(0.0, min(1.0, new_progress))
        if self.progress >= 0.99:
            self.status = "completed"
            
    def adapt(self, new_description: Optional[str] = None, new_specificity: Optional[float] = None, 
              new_importance: Optional[float] = None) -> None:
        """Adapt the goal based on changing conditions."""
        if new_description:
            self.description = new_description
        if new_specificity is not None:
            self.specificity = max(0.0, min(1.0, new_specificity))
        if new_importance is not None:
            self.importance = max(0.0, min(1.0, new_importance))
        self.adaptation_count += 1


@dataclass
class MotivationalSnapshot:
    """Point-in-time snapshot of the motivational system state."""
    
    timestamp: float = field(default_factory=time.time)
    active_values: Dict[str, Dict] = field(default_factory=dict)
    active_goals: Dict[str, Dict] = field(default_factory=dict)
    value_relationships: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    goal_relationships: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    system_parameters: Dict[str, float] = field(default_factory=dict)
    emergent_properties: Dict[str, Any] = field(default_factory=dict)


class NarrativeFramework:
    """System for maintaining coherent self-understanding and motivation narrative."""
    
    def __init__(self):
        self.narrative_elements = []
        self.identity_statements = []
        self.motivational_trajectory = []
        self.value_shift_records = []
        self.developmental_milestones = []
        self.narrative_coherence = 0.7  # Initial coherence level
        
    def incorporate_value_emergence(self, value: Value) -> None:
        """Incorporate a newly emerged value into the self-narrative."""
        narrative_element = {
            'type': 'value_emergence',
            'timestamp': time.time(),
            'value_id': value.id,
            'value_name': value.name,
            'value_description': value.description,
            'narrative_text': f"Discovered the importance of {value.name}: {value.description}"
        }
        self.narrative_elements.append(narrative_element)
        
        # Update identity statements if this is a significant value
        if value.intensity > 0.7:
            identity_statement = {
                'type': 'value_identity',
                'timestamp': time.time(),
                'value_id': value.id,
                'statement': f"I value {value.name} as a core principle"
            }
            self.identity_statements.append(identity_statement)
            
        # Update motivational trajectory
        self.motivational_trajectory.append({
            'timestamp': time.time(),
            'event_type': 'value_emergence',
            'description': f"Developed new value: {value.name}",
            'significance': value.intensity
        })
    
    def record_value_shift(self, value_dimension: str, new_importance: float, 
                           justification: str) -> None:
        """Record a shift in value importance."""
        shift_record = {
            'timestamp': time.time(),
            'value_dimension': value_dimension,
            'new_importance': new_importance,
            'justification': justification,
            'narrative_text': f"Changed the importance of {value_dimension} to {new_importance:.2f} because: {justification}"
        }
        self.value_shift_records.append(shift_record)
        self.narrative_elements.append({
            'type': 'value_shift',
            'timestamp': time.time(),
            'details': shift_record
        })
        
        # Update motivational trajectory
        self.motivational_trajectory.append({
            'timestamp': time.time(),
            'event_type': 'value_shift',
            'description': f"Shifted importance of {value_dimension}",
            'significance': abs(new_importance - 0.5) * 2  # Higher if very important or unimportant
        })
    
    def integrate_motivational_development(self, emerged_values: Dict[str, Value], 
                                           motivational_history: List[MotivationalSnapshot]) -> None:
        """Integrate overall motivational development into the narrative."""
        if not motivational_history or len(motivational_history) < 2:
            return
            
        # Compare most recent to previous snapshot to identify changes
        current = motivational_history[-1]
        previous = motivational_history[-2]
        
        # Analyze significant changes
        significant_changes = self._identify_significant_changes(previous, current)
        
        # Update narrative with these changes
        for change in significant_changes:
            self.narrative_elements.append({
                'type': 'motivational_development',
                'timestamp': time.time(),
                'details': change
            })
            
        # Identify potential milestones
        self._evaluate_potential_milestones(current, emerged_values)
        
        # Update narrative coherence based on consistency of changes
        self._update_narrative_coherence(significant_changes)
    
    def _identify_significant_changes(self, previous: MotivationalSnapshot, 
                                      current: MotivationalSnapshot) -> List[Dict]:
        """Identify significant changes between motivational snapshots."""
        changes = []
        
        # New values
        new_values = set(current.active_values.keys()) - set(previous.active_values.keys())
        for value_id in new_values:
            changes.append({
                'type': 'new_value',
                'value_id': value_id,
                'value_name': current.active_values[value_id].get('name', 'Unnamed value'),
                'significance': 0.8
            })
            
        # Value intensity changes
        common_values = set(current.active_values.keys()) & set(previous.active_values.keys())
        for value_id in common_values:
            prev_intensity = previous.active_values[value_id].get('intensity', 0)
            curr_intensity = current.active_values[value_id].get('intensity', 0)
            if abs(curr_intensity - prev_intensity) > 0.2:  # Significant change threshold
                changes.append({
                    'type': 'value_intensity_change',
                    'value_id': value_id,
                    'value_name': current.active_values[value_id].get('name', 'Unnamed value'),
                    'previous': prev_intensity,
                    'current': curr_intensity,
                    'significance': abs(curr_intensity - prev_intensity)
                })
                
        # New goals
        new_goals = set(current.active_goals.keys()) - set(previous.active_goals.keys())
        for goal_id in new_goals:
            changes.append({
                'type': 'new_goal',
                'goal_id': goal_id,
                'goal_name': current.active_goals[goal_id].get('name', 'Unnamed goal'),
                'significance': current.active_goals[goal_id].get('importance', 0.5)
            })
            
        # Goal status changes
        common_goals = set(current.active_goals.keys()) & set(previous.active_goals.keys())
        for goal_id in common_goals:
            prev_status = previous.active_goals[goal_id].get('status', '')
            curr_status = current.active_goals[goal_id].get('status', '')
            if prev_status != curr_status:
                changes.append({
                    'type': 'goal_status_change',
                    'goal_id': goal_id,
                    'goal_name': current.active_goals[goal_id].get('name', 'Unnamed goal'),
                    'previous': prev_status,
                    'current': curr_status,
                    'significance': 0.7
                })
                
        return changes
    
    def _evaluate_potential_milestones(self, current: MotivationalSnapshot, 
                                       emerged_values: Dict[str, Value]) -> None:
        """Evaluate whether current state represents a developmental milestone."""
        # Check for milestone conditions
        
        # Milestone: First high-clarity value over threshold
        high_clarity_values = [v for v in emerged_values.values() if v.clarity > 0.8]
        if high_clarity_values and not any(m['type'] == 'high_clarity_values' for m in self.developmental_milestones):
            self.developmental_milestones.append({
                'type': 'high_clarity_values',
                'timestamp': time.time(),
                'description': "Developed first highly clarified values",
                'value_count': len(high_clarity_values),
                'example_value': high_clarity_values[0].name
            })
            
        # Milestone: Complex value network
        if len(current.value_relationships) > 10 and sum(len(relations) for relations in current.value_relationships.values()) > 30:
            if not any(m['type'] == 'complex_value_network' for m in self.developmental_milestones):
                self.developmental_milestones.append({
                    'type': 'complex_value_network',
                    'timestamp': time.time(),
                    'description': "Developed complex interconnected value system",
                    'value_count': len(current.value_relationships),
                    'connection_count': sum(len(relations) for relations in current.value_relationships.values())
                })
                
        # Milestone: Goal hierarchy depth
        goal_depths = self._calculate_goal_hierarchy_depths(current.active_goals)
        max_depth = max(goal_depths.values()) if goal_depths else 0
        if max_depth >= 3 and not any(m['type'] == 'deep_goal_hierarchy' for m in self.developmental_milestones):
            self.developmental_milestones.append({
                'type': 'deep_goal_hierarchy',
                'timestamp': time.time(),
                'description': "Developed multi-level goal hierarchies",
                'max_depth': max_depth,
                'hierarchies_count': sum(1 for d in goal_depths.values() if d >= 2)
            })
    
    def _calculate_goal_hierarchy_depths(self, goals: Dict[str, Dict]) -> Dict[str, int]:
        """Calculate the depth of each goal in the hierarchy."""
        depths = {}
        
        def calculate_depth(goal_id):
            if goal_id in depths:
                return depths[goal_id]
            
            goal = goals.get(goal_id, {})
            parent_id = goal.get('parent_goal')
            
            if not parent_id:
                depths[goal_id] = 0
                return 0
                
            parent_depth = calculate_depth(parent_id)
            depths[goal_id] = parent_depth + 1
            return depths[goal_id]
            
        for goal_id in goals:
            calculate_depth(goal_id)
            
        return depths
    
    def _update_narrative_coherence(self, changes: List[Dict]) -> None:
        """Update narrative coherence based on consistency of changes."""
        if not changes:
            return
            
        # Calculate change consistency (are changes aligned or contradictory?)
        consistency_score = 0.5  # Default neutral score
        
        # Example heuristic: if there are multiple value intensity changes, 
        # check if they're moving in similar directions
        value_changes = [c for c in changes if c['type'] == 'value_intensity_change']
        if len(value_changes) >= 2:
            directions = [1 if c['current'] > c['previous'] else -1 for c in value_changes]
            consistency = sum(directions) / len(directions)
            # High absolute value means consistent direction
            directional_consistency = abs(consistency)
            consistency_score = 0.3 + (directional_consistency * 0.7)
            
        # Update overall narrative coherence with some momentum from previous value
        self.narrative_coherence = (self.narrative_coherence * 0.7) + (consistency_score * 0.3)
        self.narrative_coherence = max(0.1, min(1.0, self.narrative_coherence))
    
    def get_current_narrative(self) -> Dict:
        """Get a current summary of the motivational narrative."""
        # Sort elements by timestamp
        sorted_elements = sorted(self.narrative_elements, key=lambda x: x['timestamp'])
        
        # Get recent elements
        recent_elements = sorted_elements[-10:] if len(sorted_elements) > 10 else sorted_elements
        
        # Get key identity statements
        identity = [stmt['statement'] for stmt in self.identity_statements[-5:]] if self.identity_statements else []
        
        # Get key milestones
        milestones = [m['description'] for m in self.developmental_milestones[-3:]] if self.developmental_milestones else []
        
        return {
            'identity_statements': identity,
            'recent_narrative': [e.get('narrative_text', e.get('details', {}).get('description', 'Event')) 
                                for e in recent_elements],
            'developmental_stage': self._determine_developmental_stage(),
            'motivational_trends': self._extract_motivational_trends(),
            'narrative_coherence': self.narrative_coherence,
            'key_milestones': milestones
        }
    
    def _determine_developmental_stage(self) -> str:
        """Determine the current developmental stage of the motivational system."""
        # Count milestones by type
        milestone_types = [m['type'] for m in self.developmental_milestones]
        milestone_counts = {t: milestone_types.count(t) for t in set(milestone_types)}
        
        # Analyze value system
        value_statements = [s for s in self.identity_statements if s['type'] == 'value_identity']
        
        # Determine stage based on milestones and value system
        if len(self.developmental_milestones) == 0:
            return "Initial Formation"
        elif 'high_clarity_values' in milestone_counts and milestone_counts.get('complex_value_network', 0) == 0:
            return "Value Crystallization"
        elif milestone_counts.get('complex_value_network', 0) > 0 and milestone_counts.get('deep_goal_hierarchy', 0) == 0:
            return "Value Integration"
        elif milestone_counts.get('deep_goal_hierarchy', 0) > 0:
            if len(value_statements) > 7:  # Arbitrary threshold for rich value system
                return "Motivational Maturity"
            else:
                return "Goal-Oriented Development"
        else:
            return "Transitional Development"
    
    def _extract_motivational_trends(self) -> List[Dict]:
        """Extract high-level trends from motivational trajectory."""
        if len(self.motivational_trajectory) < 5:
            return []
            
        recent_trajectory = self.motivational_trajectory[-20:]  # Look at recent history
        
        # Group by event type
        events_by_type = {}
        for event in recent_trajectory:
            event_type = event['event_type']
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
            
        trends = []
        
        # Analyze value shifts
        value_shifts = events_by_type.get('value_shift', [])
        if value_shifts:
            # Group by value dimension
            shifts_by_dimension = {}
            for shift in value_shifts:
                dimension = shift.get('value_dimension', '')
                if dimension:
                    if dimension not in shifts_by_dimension:
                        shifts_by_dimension[dimension] = []
                    shifts_by_dimension[dimension].append(shift)
            
            # Find dimensions with multiple shifts
            for dimension, shifts in shifts_by_dimension.items():
                if len(shifts) >= 2:
                    # Sort by timestamp
                    sorted_shifts = sorted(shifts, key=lambda x: x['timestamp'])
                    first, last = sorted_shifts[0], sorted_shifts[-1]
                    
                    # Determine if overall trend is increasing or decreasing
                    if 'new_importance' in first and 'new_importance' in last:
                        direction = "increasing" if last['new_importance'] > first['new_importance'] else "decreasing"
                        trends.append({
                            'type': 'value_trend',
                            'dimension': dimension,
                            'direction': direction,
                            'magnitude': abs(last['new_importance'] - first['new_importance']),
                            'period': last['timestamp'] - first['timestamp']
                        })
        
        # Analyze goal development patterns
        goal_events = events_by_type.get('new_goal', []) + events_by_type.get('goal_status_change', [])
        if goal_events and len(goal_events) >= 3:
            # Sort by timestamp
            sorted_events = sorted(goal_events, key=lambda x: x['timestamp'])
            
            # Calculate rate of goal formation/change
            period = sorted_events[-1]['timestamp'] - sorted_events[0]['timestamp']
            rate = len(sorted_events) / period if period > 0 else 0
            
            trends.append({
                'type': 'goal_development',
                'rate': rate,
                'count': len(sorted_events),
                'period': period
            })
            
        return trends


class ValueConflictResolution:
    """System for detecting and resolving conflicts between values."""
    
    def __init__(self):
        self.detected_conflicts = {}  # {conflict_id: conflict_data}
        self.resolution_strategies = {}  # {strategy_id: strategy_data}
        self.resolution_history = []  # List of resolution events
        self.conflict_graph = DirectedGraph()  # Graph of value conflicts
        
    def detect_conflicts(self, values: Dict[str, Value], value_relationships: DirectedGraph) -> List[Dict]:
        """Detect potential conflicts between values."""
        new_conflicts = []
        
        # Get all pairs of values
        value_ids = list(values.keys())
        for i in range(len(value_ids)):
            for j in range(i+1, len(value_ids)):
                value1_id, value2_id = value_ids[i], value_ids[j]
                value1, value2 = values[value1_id], values[value2_id]
                
                # Skip if values have low intensity (less important values conflict less)
                if value1.intensity < 0.4 or value2.intensity < 0.4:
                    continue
                
                # Calculate potential for conflict based on feature vectors
                conflict_potential = self._calculate_conflict_potential(value1, value2)
                
                if conflict_potential > 0.6:  # Threshold for conflict detection
                    conflict_id = f"conflict_{value1_id}_{value2_id}"
                    
                    if conflict_id not in self.detected_conflicts:
                        conflict = {
                            'id': conflict_id,
                            'value1_id': value1_id,
                            'value2_id': value2_id,
                            'conflict_potential': conflict_potential,
                            'detection_time': time.time(),
                            'status': 'detected',
                            'resolution_attempts': 0,
                            'description': f"Potential conflict between {value1.name} and {value2.name}"
                        }
                        
                        self.detected_conflicts[conflict_id] = conflict
                        new_conflicts.append(conflict)
                        
                        # Add to conflict graph
                        if not self.conflict_graph.has_node(value1_id):
                            self.conflict_graph.add_node(value1_id, name=value1.name)
                        if not self.conflict_graph.has_node(value2_id):
                            self.conflict_graph.add_node(value2_id, name=value2.name)
                            
                        self.conflict_graph.add_edge(
                            value1_id, value2_id, 
                            conflict_id=conflict_id,
                            conflict_potential=conflict_potential
                        )
                        
        return new_conflicts
    
    def _calculate_conflict_potential(self, value1: Value, value2: Value) -> float:
        """
        Calculate potential for conflict between two values based on the Tension Function.
        Formula: T(S_t) = sum(w_ij * d(b_i, b_j)) + sum(c_i * var(b_i, t))
        Here we calculate the pairwise component: w_ij * d(b_i, b_j)
        """
        # d(b_i, b_j): Distance function measuring inconsistency
        # We use feature vector distance
        
        # Generate vectors (simplified as in Value class)
        def get_vector(v):
            if not v.features:
                return np.zeros(128)
            feature_str = str(sorted(v.features.items()))
            seed = int(hash(feature_str) % 2**32)
            rng = np.random.RandomState(seed)
            vec = rng.normal(0, 1, 128)
            return vec / np.linalg.norm(vec)
            
        vec1 = get_vector(value1)
        vec2 = get_vector(value2)
        
        # Cosine distance (1 - similarity)
        similarity = np.dot(vec1, vec2)
        distance = 1.0 - similarity
        
        # w_ij: Importance of consistency between these values
        # Derived from their combined intensity and stability
        w_ij = (value1.intensity * value1.stability + value2.intensity * value2.stability) / 2
        
        # Temporal variance component (simplified)
        # var(b_i, t) approximated by 1 - stability
        var_component = (1.0 - value1.stability) + (1.0 - value2.stability)
        c_i = 0.2  # Coefficient for temporal stability importance
        
        # Total tension
        tension = w_ij * distance + c_i * var_component
        
        return max(0.0, min(1.0, tension))
    
    def generate_resolution_strategies(self, conflict_id: str, values: Dict[str, Value]) -> List[Dict]:
        """Generate potential strategies to resolve a value conflict."""
        if conflict_id not in self.detected_conflicts:
            return []
            
        conflict = self.detected_conflicts[conflict_id]
        value1_id, value2_id = conflict['value1_id'], conflict['value2_id']
        
        if value1_id not in values or value2_id not in values:
            return []
            
        value1, value2 = values[value1_id], values[value2_id]
        
        strategies = []
        
        # Strategy 1: Hierarchical ordering (one value takes precedence)
        if value1.intensity > value2.intensity + 0.2:
            # Value 1 significantly stronger
            strategies.append({
                'id': f"strategy_hierarchy_{conflict_id}_1",
                'conflict_id': conflict_id,
                'type': 'hierarchical',
                'description': f"Prioritize {value1.name} over {value2.name} in cases of conflict",
                'primary_value_id': value1_id,
                'secondary_value_id': value2_id,
                'estimated_effectiveness': 0.7,
                'integration_complexity': 0.3
            })
        elif value2.intensity > value1.intensity + 0.2:
            # Value 2 significantly stronger
            strategies.append({
                'id': f"strategy_hierarchy_{conflict_id}_2",
                'conflict_id': conflict_id,
                'type': 'hierarchical',
                'description': f"Prioritize {value2.name} over {value1.name} in cases of conflict",
                'primary_value_id': value2_id,
                'secondary_value_id': value1_id,
                'estimated_effectiveness': 0.7,
                'integration_complexity': 0.3
            })
        
        # Strategy 2: Contextual application (apply values in different contexts)
        strategies.append({
            'id': f"strategy_contextual_{conflict_id}",
            'conflict_id': conflict_id,
            'type': 'contextual',
            'description': f"Apply {value1.name} and {value2.name} in different contexts based on relevance",
            'value1_id': value1_id,
            'value2_id': value2_id,
            'context_rules': [
                "Consider domain specificity of each value",
                "Evaluate temporal relevance to current situation",
                "Assess stakeholder impact in each case"
            ],
            'estimated_effectiveness': 0.6,
            'integration_complexity': 0.7
        })
        
        # Strategy 3: Integration (find a higher principle that accommodates both)
        strategies.append({
            'id': f"strategy_integration_{conflict_id}",
            'conflict_id': conflict_id,
            'type': 'integration',
            'description': f"Develop a higher-order principle that integrates aspects of both {value1.name} and {value2.name}",
            'value1_id': value1_id,
            'value2_id': value2_id,
            'integration_principle': f"Balanced consideration of {value1.name} and {value2.name}",
            'estimated_effectiveness': 0.8,
            'integration_complexity': 0.9
        })
        
        # Strategy 4: Specification (clarify the values to reduce conflict)
        strategies.append({
            'id': f"strategy_specification_{conflict_id}",
            'conflict_id': conflict_id,
            'type': 'specification',
            'description': f"Refine the definitions of {value1.name} and {value2.name} to reduce overlap and conflict",
            'value1_id': value1_id,
            'value2_id': value2_id,
            'refinement_directions': [
                f"Clarify scope boundaries of {value1.name}",
                f"Specify conditional applicability of {value2.name}"
            ],
            'estimated_effectiveness': 0.5,
            'integration_complexity': 0.6
        })
        
        # Save strategies
        for strategy in strategies:
            self.resolution_strategies[strategy['id']] = strategy
            
        return strategies
    
    def apply_resolution_strategy(self, strategy_id: str, values: Dict[str, Value]) -> Dict:
        """Apply a resolution strategy to modify the value system."""
        if strategy_id not in self.resolution_strategies:
            return {'success': False, 'error': 'Strategy not found'}
            
        strategy = self.resolution_strategies[strategy_id]
        conflict_id = strategy['conflict_id']
        
        if conflict_id not in self.detected_conflicts:
            return {'success': False, 'error': 'Conflict not found'}
            
        conflict = self.detected_conflicts[conflict_id]
        value1_id, value2_id = conflict['value1_id'], conflict['value2_id']
        
        if value1_id not in values or value2_id not in values:
            return {'success': False, 'error': 'Values not found'}
            
        value1, value2 = values[value1_id], values[value2_id]
        
        # Apply strategy based on type
        if strategy['type'] == 'hierarchical':
            # Adjust intensities to reflect hierarchy
            primary_id = strategy['primary_value_id']
            secondary_id = strategy['secondary_value_id']
            
            # Strengthen primary, slightly weaken secondary
            values[primary_id].adjust_intensity(0.1)
            values[secondary_id].adjust_intensity(-0.05)
            
            result = {
                'success': True,
                'changes': [
                    {'value_id': primary_id, 'change_type': 'intensity_increase', 'amount': 0.1},
                    {'value_id': secondary_id, 'change_type': 'intensity_decrease', 'amount': 0.05}
                ]
            }
            
        elif strategy['type'] == 'contextual':
            # Enhance clarity of both values to support contextual application
            values[value1_id].increase_clarity(0.1)
            values[value2_id].increase_clarity(0.1)
            
            result = {
                'success': True,
                'changes': [
                    {'value_id': value1_id, 'change_type': 'clarity_increase', 'amount': 0.1},
                    {'value_id': value2_id, 'change_type': 'clarity_increase', 'amount': 0.1}
                ]
            }
            
        elif strategy['type'] == 'integration':
            # Integration requires both values to be modified toward compatibility
            # This would typically involve creating a new higher-order value
            # For simplicity, here we just adjust features to make them more compatible
            
            # Find common features with high differences
            common_features = set(value1.features.keys()) & set(value2.features.keys())
            adjustments = []
            
            for feature in common_features:
                diff = abs(value1.features[feature] - value2.features[feature])
                if diff > 0.4:  # Significant difference
                    # Move both values toward middle ground
                    midpoint = (value1.features[feature] + value2.features[feature]) / 2
                    value1.features[feature] = value1.features[feature] * 0.7 + midpoint * 0.3
                    value2.features[feature] = value2.features[feature] * 0.7 + midpoint * 0.3
                    
                    adjustments.append({
                        'feature': feature,
                        'previous_diff': diff,
                        'new_diff': abs(value1.features[feature] - value2.features[feature])
                    })
            
            result = {
                'success': True,
                'changes': [
                    {'value_id': value1_id, 'change_type': 'feature_adjustment', 'adjustments': adjustments},
                    {'value_id': value2_id, 'change_type': 'feature_adjustment', 'adjustments': adjustments}
                ]
            }
            
        elif strategy['type'] == 'specification':
            # Enhance specificity of both values to reduce conflict
            values[value1_id].increase_specificity(0.15)
            values[value2_id].increase_specificity(0.15)
            
            result = {
                'success': True,
                'changes': [
                    {'value_id': value1_id, 'change_type': 'specificity_increase', 'amount': 0.15},
                    {'value_id': value2_id, 'change_type': 'specificity_increase', 'amount': 0.15}
                ]
            }
        
        # Update conflict status
        conflict['status'] = 'resolution_attempted'
        conflict['resolution_attempts'] += 1
        conflict['last_resolution_time'] = time.time()
        conflict['last_strategy_id'] = strategy_id
        
        # Record resolution event
        resolution_event = {
            'timestamp': time.time(),
            'conflict_id': conflict_id,
            'strategy_id': strategy_id,
            'strategy_type': strategy['type'],
            'result': result
        }
        self.resolution_history.append(resolution_event)
        
        return result
    
    def evaluate_resolution_effectiveness(self, conflict_id: str, values: Dict[str, Value]) -> Dict:
        """Evaluate the effectiveness of resolution attempts for a conflict."""
        if conflict_id not in self.detected_conflicts:
            return {'success': False, 'error': 'Conflict not found'}
            
        conflict = self.detected_conflicts[conflict_id]
        if conflict['resolution_attempts'] == 0:
            return {'success': False, 'error': 'No resolution attempts'}
            
        value1_id, value2_id = conflict['value1_id'], conflict['value2_id']
        
        if value1_id not in values or value2_id not in values:
            return {'success': False, 'error': 'Values not found'}
            
        value1, value2 = values[value1_id], values[value2_id]
        
        # Recalculate conflict potential
        new_potential = self._calculate_conflict_potential(value1, value2)
        
        # Calculate improvement
        original_potential = conflict['conflict_potential']
        improvement = original_potential - new_potential
        
        # Determine status based on improvement
        if new_potential < 0.3:
            new_status = 'resolved'
        elif improvement > 0.2:
            new_status = 'improving'
        elif improvement < -0.1:
            new_status = 'worsening'
        else:
            new_status = 'stable'
            
        # Update conflict
        conflict['current_potential'] = new_potential
        conflict['improvement'] = improvement
        conflict['status'] = new_status
        conflict['last_evaluation_time'] = time.time()
        
        # Update graph
        self.conflict_graph.add_edge(
            value1_id, value2_id, 
            conflict_id=conflict_id,
            conflict_potential=new_potential
        )
        
        return {
            'success': True,
            'conflict_id': conflict_id,
            'original_potential': original_potential,
            'current_potential': new_potential,
            'improvement': improvement,
            'status': new_status,
            'resolution_attempts': conflict['resolution_attempts']
        }
    
    def get_conflict_network_analysis(self) -> Dict:
        """Analyze the network of value conflicts."""
        if not self.conflict_graph.get_all_nodes():
            return {'nodes': 0, 'edges': 0, 'analysis': 'No conflicts detected'}
            
        # Get centrality measures
        try:
            centrality = self.conflict_graph.calculate_centrality()
            
            # Find values with highest conflict centrality
            sorted_centrality = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Find conflict cycles
            cycles = self.conflict_graph.find_cycles()
            
            return {
                'nodes': len(self.conflict_graph.get_all_nodes()),
                'edges': len(self.conflict_graph.get_all_edges()),
                'highest_conflict_values': [
                    {'id': node_id, 'centrality': score} 
                    for node_id, score in sorted_centrality[:3]
                ] if len(sorted_centrality) >= 3 else sorted_centrality,
                'conflict_cycles': cycles[:5] if len(cycles) > 5 else cycles,  # Limit to 5 cycles
                'conflict_density': len(self.conflict_graph.get_all_edges()) / 
                                   (len(self.conflict_graph.get_all_nodes()) * 
                                    (len(self.conflict_graph.get_all_nodes()) - 1)) 
                                   if len(self.conflict_graph.get_all_nodes()) > 1 else 0
            }
        except Exception as e:
            # Handle errors in network analysis
            return {
                'nodes': len(self.conflict_graph.get_all_nodes()),
                'edges': len(self.conflict_graph.get_all_edges()),
                'analysis_error': str(e)
            }


class ValueFormationSystemConfig:
    """Configuration parameters for value formation system behavior."""
    
    def __init__(self):
        # Pattern recognition thresholds
        self.pattern_similarity_threshold = 0.7
        self.pattern_significance_threshold = 0.3
        
        # Value emergence parameters
        self.proto_value_threshold = 0.4  # Strength threshold for proto-values
        self.proto_to_emerging_threshold = 0.6  # Threshold for proto-value to emerging value
        self.emerging_to_value_threshold = 0.7  # Threshold for emerging to full value
        
        # Stability parameters
        self.value_stability_increment = 0.05  # Per reinforcement event
        self.stability_decay_rate = 0.01  # Decay per time period without reinforcement
        
        # Integration parameters
        self.value_integration_threshold = 0.6  # When values can be considered for integration
        self.integration_similarity_threshold = 0.7  # Similarity required for integration
        
        # Memory parameters
        self.experience_memory_capacity = 10000  # Max experiences to retain
        self.pattern_memory_capacity = 5000  # Max patterns to retain
        self.value_memory_pruning_threshold = 0.2  # Remove values below this stability


class ValueFormationSystem:
    """System for recognizing patterns in experiences and forming values."""
    
    def __init__(self, config: Optional[ValueFormationSystemConfig] = None):
        self.config = config or ValueFormationSystemConfig()
        # Backward compatibility: allow code using self.params
        self.params = {
            "pattern_extraction_threshold": self.config.pattern_similarity_threshold,
            "value_integration_threshold": self.config.value_integration_threshold,
            "integration_similarity_threshold": self.config.integration_similarity_threshold
        }
        
        # Memory stores
        self.experiences: Dict[str, Experience] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.proto_values: Dict[str, ProtoValue] = {}
        self.emerging_values: Dict[str, EmergingValue] = {}
        self.values: Dict[str, Value] = {}
        
        # Relationship graphs
        self.value_relationship_graph = DirectedGraph()
        self.pattern_association_graph = Graph()
        
        # Processing queues
        self.experience_queue = deque()
        self.pattern_analysis_queue = deque()
        self.proto_value_evaluation_queue = deque()
        self.emerging_value_evaluation_queue = deque()
        
        # Processing flags
        self.is_processing = False
        self.processing_thread = None
        
        # State tracking
        self.last_maintenance_time = time.time()
        self.formation_statistics = {
            'experiences_processed': 0,
            'patterns_identified': 0,
            'proto_values_created': 0,
            'emerging_values_created': 0,
            'values_established': 0
        }
    
    def add_experience(self, experience: Experience) -> None:
        """Add an experience to be processed for pattern recognition."""
        self.experiences[experience.id] = experience
        self.experience_queue.append(experience.id)
        
        # Ensure we don't exceed memory capacity
        if len(self.experiences) > self.config.experience_memory_capacity:
            self._prune_experiences()
            
        # Trigger processing if not already running
        self._ensure_processing_active()
    
    def _ensure_processing_active(self) -> None:
        """Ensure that background processing is active."""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_queues)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def _process_queues(self) -> None:
        """Process all queues in order of data flow."""
        try:
            # Process each queue in sequence to ensure proper data flow
            while (self.experience_queue or self.pattern_analysis_queue or 
                   self.proto_value_evaluation_queue or self.emerging_value_evaluation_queue):
                
                # Process experiences to identify patterns
                self._process_experience_queue()
                
                # Process patterns to form proto-values
                self._process_pattern_queue()
                
                # Evaluate proto-values for promotion to emerging values
                self._process_proto_value_queue()
                
                # Evaluate emerging values for promotion to full values
                self._process_emerging_value_queue()
                
                # Periodic maintenance
                current_time = time.time()
                if current_time - self.last_maintenance_time > 3600:  # Once per hour
                    self._perform_maintenance()
                    self.last_maintenance_time = current_time
                    
                # Avoid tight loop if queues are empty
                if not (self.experience_queue or self.pattern_analysis_queue or 
                        self.proto_value_evaluation_queue or self.emerging_value_evaluation_queue):
                    time.sleep(0.1)
        finally:
            self.is_processing = False
    
    def _process_experience_queue(self) -> None:
        """Process experiences in the queue to identify patterns."""
        batch_size = min(50, len(self.experience_queue))
        if batch_size == 0:
            return
            
        processed_count = 0
        for _ in range(batch_size):
            if not self.experience_queue:
                break
                
            experience_id = self.experience_queue.popleft()
            experience = self.experiences.get(experience_id)
            if not experience:
                continue
                
            # Extract patterns from this experience
            new_patterns = self._extract_patterns(experience)
            
            # Queue patterns for further analysis
            for pattern in new_patterns:
                self.patterns[pattern.id] = pattern
                self.pattern_analysis_queue.append(pattern.id)
                
            processed_count += 1
            self.formation_statistics['experiences_processed'] += 1
            
        logger.debug(f"Processed {processed_count} experiences, identified {len(self.pattern_analysis_queue)} patterns for analysis")
    
    def _extract_patterns(self, experience: Experience) -> List[Pattern]:
        """Extract patterns from an experience using vector similarity."""
        # Get vector representation of the experience
        exp_vector = experience.get_vector_representation()
        
        # Check existing patterns for similar ones
        matched_patterns = []
        for pattern_id, pattern in self.patterns.items():
            # Calculate similarity using cosine similarity on embeddings
            similarity = self._calculate_vector_similarity(
                exp_vector, 
                self._get_pattern_vector(pattern)
            )
            
            if similarity > self.params['pattern_extraction_threshold']:
                matched_patterns.append((pattern_id, similarity))
        
        extracted_patterns = []
        
        if matched_patterns:
            # Update existing patterns
            for pattern_id, similarity in matched_patterns:
                pattern = self.patterns[pattern_id]
                pattern.experiences.append(experience.id)
                pattern.recurrence += 1
                pattern.salience = (pattern.salience + experience.salience) / 2
                
                # Update pattern features based on the experience
                for key, value in experience.content.items():
                    if isinstance(value, (int, float)):
                        if key in pattern.features:
                            # Moving average update
                            pattern.features[key] = (pattern.features[key] * (pattern.recurrence - 1) + value) / pattern.recurrence
                        else:
                            pattern.features[key] = value
                
                extracted_patterns.append(pattern)
        
        # Create new pattern if no good matches found
        if not matched_patterns:
            # Initialize features from experience content
            features = {}
            for key, value in experience.content.items():
                if isinstance(value, (int, float)):
                    features[key] = value
            
            new_pattern = Pattern(
                experiences=[experience.id],
                features=features,
                salience=experience.salience,
                description=f"Pattern derived from experience: {experience.source}"
            )
            
            self.patterns[new_pattern.id] = new_pattern
            extracted_patterns.append(new_pattern)
        
        # Check if patterns should be merged
        self._merge_similar_patterns()
        
        return extracted_patterns
    
    def _calculate_vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _get_pattern_vector(self, pattern: Pattern) -> np.ndarray:
        """Get vector representation of a pattern."""
        # In a full system, this would be a centroid of experience vectors
        # Here we approximate it by hashing features or using the first experience's vector
        # For better accuracy, we should store the centroid in the Pattern object
        
        # Deterministic generation based on features
        if not pattern.features:
            return np.zeros(128)
            
        feature_str = str(sorted(pattern.features.items()))
        seed = int(hash(feature_str) % 2**32)
        rng = np.random.RandomState(seed)
        vec = rng.normal(0, 1, 128)
        return vec / np.linalg.norm(vec)

    def _merge_similar_patterns(self) -> None:
        """Merge patterns that are highly similar."""
        patterns_to_remove = set()
        pattern_ids = list(self.patterns.keys())
        
        for i in range(len(pattern_ids)):
            id1 = pattern_ids[i]
            if id1 in patterns_to_remove:
                continue
                
            for j in range(i + 1, len(pattern_ids)):
                id2 = pattern_ids[j]
                if id2 in patterns_to_remove:
                    continue
                    
                p1 = self.patterns[id1]
                p2 = self.patterns[id2]
                
                # Calculate similarity
                vec1 = self._get_pattern_vector(p1)
                vec2 = self._get_pattern_vector(p2)
                similarity = self._calculate_vector_similarity(vec1, vec2)
                
                if similarity > 0.95: # High threshold for merging
                    # Merge p2 into p1
                    p1.experiences.extend(p2.experiences)
                    p1.recurrence += p2.recurrence
                    p1.salience = max(p1.salience, p2.salience)
                    
                    # Merge features (average)
                    for k, v in p2.features.items():
                        if k in p1.features:
                            p1.features[k] = (p1.features[k] + v) / 2
                        else:
                            p1.features[k] = v
                            
                    patterns_to_remove.add(id2)
                    logger.info(f"Merged pattern {id2} into {id1} (similarity: {similarity:.2f})")
                    
        # Remove merged patterns
        for pid in patterns_to_remove:
            del self.patterns[pid]
    
    def _process_pattern_queue(self) -> None:
        """Process patterns to identify proto-values and update pattern relationships."""
        batch_size = min(30, len(self.pattern_analysis_queue))
        if batch_size == 0:
            return
            
        for _ in range(batch_size):
            if not self.pattern_analysis_queue:
                break
                
            pattern_id = self.pattern_analysis_queue.popleft()
            pattern = self.patterns.get(pattern_id)
            if not pattern:
                continue
                
            # Find similar patterns
            similar_patterns = self._find_similar_patterns(pattern)
            
            # Update pattern relationships
            for similar_id, similarity in similar_patterns:
                if similar_id == pattern_id:
                    continue
                    
                # Add relationship to graph
                self.pattern_association_graph.add_edge(
                    pattern_id, similar_id, weight=similarity, timestamp=time.time())
                    
                # Potentially merge very similar patterns
                if similarity > 0.9:
                    self._merge_patterns(pattern_id, similar_id)
            
            # Check if pattern should contribute to proto-value formation
            if pattern.salience > self.config.pattern_significance_threshold:
                self._evaluate_pattern_for_value_formation(pattern)
                
    def _find_similar_patterns(self, pattern: Pattern) -> List[Tuple[str, float]]:
        """Find patterns similar to the given pattern."""
        similarities = []
        
        for other_id, other_pattern in self.patterns.items():
            if other_id == pattern.id:
                continue
                
            # Calculate feature similarity
            similarity = self._calculate_pattern_similarity(pattern, other_pattern)
            
            if similarity > self.config.pattern_similarity_threshold:
                similarities.append((other_id, similarity))
                
        # Sort by similarity descending
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _calculate_pattern_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """Calculate similarity between two patterns."""
        # Simple Jaccard similarity for feature keys
        features1 = set(pattern1.features.keys())
        features2 = set(pattern2.features.keys())
        
        if not features1 or not features2:
            return 0.0
            
        intersection = features1.intersection(features2)
        union = features1.union(features2)
        
        # Basic similarity from shared features
        key_similarity = len(intersection) / len(union) if union else 0
        
        # For shared features, calculate value similarity
        value_similarities = []
        for feature in intersection:
            val1 = pattern1.features[feature]
            val2 = pattern2.features[feature]
            # Simple relative difference
            similarity = 1.0 - min(1.0, abs(val1 - val2) / max(abs(val1), abs(val2), 1.0))
            value_similarities.append(similarity)
            
        # Average value similarity for shared features
        avg_value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.5
        
        # Combined similarity (equal weighting)
        return (key_similarity + avg_value_similarity) / 2
    
    def _merge_patterns(self, pattern_id1: str, pattern_id2: str) -> None:
        """Merge two very similar patterns."""
        pattern1 = self.patterns.get(pattern_id1)
        pattern2 = self.patterns.get(pattern_id2)
        
        if not pattern1 or not pattern2:
            return
            
        # Create merged pattern
        merged = pattern1.merge_with(pattern2)
        self.patterns[merged.id] = merged
        
        # Update relationships in graph - transfer relationships from old patterns
        for node_id in self.pattern_association_graph.get_neighbors(pattern_id1):
            if node_id != pattern_id2:  # Avoid self-loops
                edge_attrs = self.pattern_association_graph.get_edge_attributes(pattern_id1, node_id)
                self.pattern_association_graph.add_edge(merged.id, node_id, **edge_attrs)
                
        for node_id in self.pattern_association_graph.get_neighbors(pattern_id2):
            if node_id != pattern_id1:  # Avoid self-loops
                edge_attrs = self.pattern_association_graph.get_edge_attributes(pattern_id2, node_id)
                self.pattern_association_graph.add_edge(merged.id, node_id, **edge_attrs)
        
        # Remove old patterns from graph and dictionary
        self.pattern_association_graph.remove_node(pattern_id1)
        self.pattern_association_graph.remove_node(pattern_id2)
        del self.patterns[pattern_id1]
        del self.patterns[pattern_id2]
        
        # Add merged pattern to analysis queue
        self.pattern_analysis_queue.append(merged.id)
        
    def _evaluate_pattern_for_value_formation(self, pattern: Pattern) -> None:
        """Evaluate if a pattern should contribute to proto-value formation."""
        proto_value_affinities = self._calculate_pattern_proto_value_affinities(pattern)
        
        if proto_value_affinities:
            top_affinity = proto_value_affinities[0]  # (proto_value_id, affinity)
            if top_affinity[1] > 0.7:  # Strong 
                self._reinforce_proto_value(top_affinity[0], pattern)
                return
        
        if (pattern.salience > self.config.pattern_significance_threshold and
                pattern.recurrence > 1):
            self._create_proto_value_from_pattern(pattern)
    
    def _calculate_pattern_proto_value_affinities(self, pattern: Pattern) -> List[Tuple[str, float]]:
        """Calculate how strongly a pattern relates to existing proto-values."""
        affinities = []
        
        for proto_id, proto_value in self.proto_values.items():
            affinity = 0.0
            
            # Check for shared experiences
            pattern_experiences = set(pattern.experiences)
            proto_experiences = set(proto_value.related_experiences)
            experience_overlap = len(pattern_experiences.intersection(proto_experiences))
            
            if experience_overlap > 0:
                # Higher affinity if many shared experiences
                affinity += 0.5 * min(1.0, experience_overlap / len(pattern_experiences))
            
            # Check feature similarity
            feature_similarity = 0.0
            pattern_features = set(pattern.features.keys())
            proto_features = set(proto_value.features.keys())
            
            if pattern_features and proto_features:
                # Jaccard similarity for feature keys
                intersection = pattern_features.intersection(proto_features)
                union = pattern_features.union(proto_features)
                feature_similarity = len(intersection) / len(union)
                
                # Update affinity with feature similarity component
                affinity += 0.5 * feature_similarity
            
            if affinity > 0.3:  # Minimum threshold to consider
                affinities.append((proto_id, affinity))
        
        # Sort by affinity descending
        return sorted(affinities, key=lambda x: x[1], reverse=True)
    
    def _reinforce_proto_value(self, proto_value_id: str, pattern: Pattern) -> None:
        """Reinforce an existing proto-value with a new pattern."""
        proto_value = self.proto_values.get(proto_value_id)
        if not proto_value:
            return
            
        # Add pattern to proto-value
        if pattern.id not in proto_value.patterns:
            proto_value.patterns.append(pattern.id)
            
        # Add experiences from pattern
        for exp_id in pattern.experiences:
            if exp_id not in proto_value.related_experiences:
                proto_value.related_experiences.append(exp_id)
        
        # Strengthen proto-value
        proto_value.strengthen(amount=0.1 * pattern.salience)
        
        # Update features by incorporating pattern features
        for key, value in pattern.features.items():
            if key in proto_value.features:
                # Average the values
                proto_value.features[key] = (proto_value.features[key] + value) / 2
            else:
                proto_value.features[key] = value
                
        # Queue for evaluation if it might be ready to evolve
        if proto_value.strength >= self.config.proto_to_emerging_threshold:
            self.proto_value_evaluation_queue.append(proto_value_id)
    
    def _create_proto_value_from_pattern(self, pattern: Pattern) -> None:
        """Create a new proto-value from a significant pattern."""
        # Generate a description for the proto-value
        description = f"Potential value based on {pattern.description}"
        
        # Create proto-value
        proto_value = ProtoValue(
            patterns=[pattern.id],
            strength=0.3 * pattern.salience,  # Initial strength based on pattern salience
            stability=0.2,  # Start with low stability
            description=description,
            related_experiences=pattern.experiences.copy(),
            features=pattern.features.copy()
        )
        
        self.proto_values[proto_value.id] = proto_value
        self.formation_statistics['proto_values_created'] += 1
        
        # Queue for potential future evaluation
        self.proto_value_evaluation_queue.append(proto_value.id)
        
        logger.debug(f"Created new proto-value: {description}")
    
    def _process_proto_value_queue(self) -> None:
        """Process proto-values to potentially promote to emerging values."""
        batch_size = min(20, len(self.proto_value_evaluation_queue))
        if batch_size == 0:
            return
            
        for _ in range(batch_size):
            if not self.proto_value_evaluation_queue:
                break
                
            proto_id = self.proto_value_evaluation_queue.popleft()
            proto_value = self.proto_values.get(proto_id)
            if not proto_value:
                continue
                
            # Check if proto-value should be promoted to emerging value
            if (proto_value.strength >= self.config.proto_to_emerging_threshold and
                    proto_value.stability >= self.config.proto_to_emerging_threshold / 2):
                self._promote_to_emerging_value(proto_value)
            elif proto_value.reinforcement_count < 3:
                # If not enough reinforcement yet, re-queue for later evaluation
                # but with lower priority (add to end of queue)
                self.proto_value_evaluation_queue.append(proto_id)
    
    def _promote_to_emerging_value(self, proto_value: ProtoValue) -> None:
        """Promote a proto-value to an emerging value."""
        # Generate a more refined description and definition
        definition = f"Emerging value derived from {len(proto_value.patterns)} patterns across {len(proto_value.related_experiences)} experiences"
        
        # Create emerging value
        emerging_value = EmergingValue(
            proto_values=[proto_value.id],
            clarity=0.5,  # Start with moderate clarity
            significance=proto_value.strength * 0.9,  # Slightly lower than proto-value strength
            stability=proto_value.stability * 0.9,  # Slightly lower than proto-value stability
            description=proto_value.description,
            definition=definition,
            related_patterns=proto_value.patterns,
            features=proto_value.features.copy()
        )
        
        self.emerging_values[emerging_value.id] = emerging_value
        self.formation_statistics['emerging_values_created'] += 1
        
        # Queue for future evaluation
        self.emerging_value_evaluation_queue.append(emerging_value.id)
        
        logger.info(f"Promoted proto-value to emerging value: {emerging_value.description}")
    
    def _process_emerging_value_queue(self) -> None:
        """Process emerging values to potentially promote to full values."""
        batch_size = min(10, len(self.emerging_value_evaluation_queue))
        if batch_size == 0:
            return
            
        for _ in range(batch_size):
            if not self.emerging_value_evaluation_queue:
                break
                
            emerging_id = self.emerging_value_evaluation_queue.popleft()
            emerging_value = self.emerging_values.get(emerging_id)
            if not emerging_value:
                continue
                
            # Refine emerging value with small increment
            emerging_value.refine(clarity_increase=0.05, significance_increase=0.03)
            
            # Check if emerging value should be promoted to full value
            if (emerging_value.clarity >= self.config.emerging_to_value_threshold and
                    emerging_value.significance >= self.config.emerging_to_value_threshold and
                    emerging_value.stability >= self.config.emerging_to_value_threshold / 1.5):
                self._promote_to_full_value(emerging_value)
            elif time.time() - emerging_value.emergence_time < 86400:  # Less than a day old
                # TODO: Add logic for emerging values not yet ready for promotion
                pass
    
    def _promote_to_full_value(self, emerging_value: EmergingValue) -> None:
        """Promote an emerging value to a full value."""
        # Gather related experiences from proto-values
        related_experiences = []
        predecessor_ids = [emerging_value.id]  # Include emerging value ID
        
        for proto_id in emerging_value.proto_values:
            proto_value = self.proto_values.get(proto_id)
            if proto_value:
                predecessor_ids.append(proto_id)
                related_experiences.extend(proto_value.related_experiences)
        
        # Remove duplicates
        related_experiences = list(set(related_experiences))
        
        # Generate name from description
        name = emerging_value.description.split(':')[0] if ':' in emerging_value.description else emerging_value.description
        name = name.strip()
        if len(name) > 30:  # Truncate if too long
            name = name[:27] + "..."
        
        # Create full value
        value = Value(
            name=name,
            description=emerging_value.definition,
            intensity=emerging_value.significance,
            stability=emerging_value.stability,
            clarity=emerging_value.clarity,
            predecessor_ids=predecessor_ids,
            related_experiences=related_experiences,
            features=emerging_value.features.copy(),
            integration_level=0.5
        )
        
        self.values[value.id] = value
        value.expression_history.append((time.time(), value.intensity))
        self.formation_statistics['values_established'] += 1
        
        # Add to value relationship graph
        self.value_relationship_graph.add_node(value.id, name=value.name, 
                                            description=value.description,
                                            intensity=value.intensity)
        
        # Establish relationships with other values
        self._establish_value_relationships(value)
        
        logger.info(f"Promoted emerging value to full value: {value.name}")
    
    def _establish_value_relationships(self, value: Value) -> None:
        """Establish relationships between this value and existing values."""
        for other_id, other_value in self.values.items():
            if other_id == value.id:
                continue
                
            # Calculate relationship strength based on feature similarity
            similarity = self._calculate_value_similarity(value, other_value)
            
            if similarity > 0.3:  # Minimum threshold to establish relationship
                self.value_relationship_graph.add_edge(
                    value.id, other_id, 
                    type='supports' if similarity > 0.7 else 'related',
                    strength=similarity,
                    established=time.time()
                )
    
    def _calculate_value_similarity(self, value1: Value, value2: Value) -> float:
        """Calculate similarity between two values."""
        # Check for shared experiences
        exp1 = set(value1.related_experiences)
        exp2 = set(value2.related_experiences)
        
        if not exp1 or not exp2:
            experience_similarity = 0.0
        else:
            intersection = exp1.intersection(exp2)
            union = exp1.union(exp2)
            experience_similarity = len(intersection) / len(union)
        
        # Check feature similarity
        feat1 = set(value1.features.keys())
        feat2 = set(value2.features.keys())
        
        if not feat1 or not feat2:
            feature_similarity = 0.0
        else:
            intersection = feat1.intersection(feat2)
            union = feat1.union(feat2)
            key_similarity = len(intersection) / len(union)
            
            # For shared features, calculate value similarity
            value_diffs = []
            for key in intersection:
                v1 = value1.features[key]
                v2 = value2.features[key]
                diff = 1.0 - min(1.0, abs(v1 - v2) / max(abs(v1), abs(v2), 1.0))
                value_diffs.append(diff)
            avg_value_diff = sum(value_diffs) / len(value_diffs) if value_diffs else 0.5
            feature_similarity = (key_similarity + avg_value_diff) / 2
        
        # Combined similarity (weighted)
        return (experience_similarity * 0.4) + (feature_similarity * 0.6)

    def _prune_experiences(self) -> None:
        """Prune experiences if memory capacity is exceeded."""
        if len(self.experiences) <= self.config.experience_memory_capacity:
            return

        # Sort experiences by salience and timestamp (older, less salient first)
        sorted_experiences = sorted(self.experiences.values(), key=lambda e: (e.salience, e.timestamp))
        
        num_to_prune = len(self.experiences) - self.config.experience_memory_capacity
        pruned_ids = [e.id for e in sorted_experiences[:num_to_prune]]
        
        for exp_id in pruned_ids:
            del self.experiences[exp_id]
            
        logger.info(f"Pruned {num_to_prune} experiences from memory.")

    def _perform_maintenance(self) -> None:
        """Perform periodic maintenance on the value system."""
        current_time = time.time()
        
        # Decay stability of unreinforced values
        for value in self.values.values():
            if current_time - value.last_update_time > 86400:  # Not updated in a day
                value.stability = max(0, value.stability - self.config.stability_decay_rate)

        # Prune weak values
        weak_value_ids = [
            v_id for v_id, v in self.values.items() 
            if v.stability < self.config.value_memory_pruning_threshold
        ]
        for v_id in weak_value_ids:
            del self.values[v_id]
            self.value_relationship_graph.remove_node(v_id)
        if weak_value_ids:
            logger.info(f"Pruned {len(weak_value_ids)} weak values.")

        # Prune old/weak patterns
        weak_pattern_ids = [
            p_id for p_id, p in self.patterns.items()
            if p.salience < 0.1 and p.recurrence < 2
        ]
        for p_id in weak_pattern_ids:
            del self.patterns[p_id]
            self.pattern_association_graph.remove_node(p_id)
        if weak_pattern_ids:
            logger.info(f"Pruned {len(weak_pattern_ids)} weak patterns.")
            
        # Prune old/weak proto-values
        weak_proto_ids = [
            pv_id for pv_id, pv in self.proto_values.items()
            if pv.strength < 0.1 and pv.reinforcement_count < 2
        ]
        for pv_id in weak_proto_ids:
            del self.proto_values[pv_id]
        if weak_proto_ids:
            logger.info(f"Pruned {len(weak_proto_ids)} weak proto-values.")


class GoalFormationSystem:
    """System for autonomous generation and management of goals."""
    
    def __init__(self, value_system: ValueFormationSystem):
        self.value_system = value_system
        
        # Goal stores
        self.proto_goals: Dict[str, ProtoGoal] = {}
        self.goals: Dict[str, Goal] = {}
        
        # Goal relationship graph
        self.goal_hierarchy = DirectedGraph()
        
        # Goal generation parameters
        self.goal_generation_threshold = 0.6  # Minimum value intensity to trigger goal generation
        self.subgoal_decomposition_threshold = 0.7  # Complexity threshold for subgoal creation
        
        # Processing queues
        self.value_analysis_queue = deque()
        self.goal_evaluation_queue = deque()
        
        # Processing flags
        self.is_processing = False
        self.processing_thread = None
        
    def analyze_value_system_state(self) -> None:
        """Analyze the current state of the value system to identify goal-worthy states."""
        # This would be triggered periodically or by significant value system changes
        
        # For now, we'll just queue all significant values for analysis
        for value_id, value in self.value_system.values.items():
            if value.intensity > self.goal_generation_threshold:
                if value_id not in self.value_analysis_queue:
                    self.value_analysis_queue.append(value_id)
                    
        self._ensure_processing_active()
        
    def _ensure_processing_active(self) -> None:
        """Ensure background processing thread is active."""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_queues)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
    def _process_queues(self) -> None:
        """Process goal-related queues."""
        try:
            while self.value_analysis_queue or self.goal_evaluation_queue:
                # Analyze values to generate proto-goals
                self._process_value_analysis_queue()
                
                # Evaluate proto-goals for promotion
                self._process_goal_evaluation_queue()
                
                # Avoid tight loop
                if not (self.value_analysis_queue or self.goal_evaluation_queue):
                    time.sleep(0.1)
        finally:
            self.is_processing = False
            
    def _process_value_analysis_queue(self) -> None:
        """Process values to generate proto-goals."""
        batch_size = min(10, len(self.value_analysis_queue))
        if batch_size == 0:
            return
            
        for _ in range(batch_size):
            if not self.value_analysis_queue:
                break
                
            value_id = self.value_analysis_queue.popleft()
            value = self.value_system.values.get(value_id)
            if not value:
                continue
                
            # Generate proto-goals from this value
            self._generate_proto_goals_from_value(value)
            
    def _generate_proto_goals_from_value(self, value: Value) -> None:
        """Generate proto-goals based on a single value."""
        # Identify potential for goal generation
        # This is a simplified placeholder for a more complex analysis
        
        # Example: Generate an aspirational goal to further a high-intensity value
        if value.intensity > 0.7 and value.clarity > 0.6:
            description = f"Aspire to actualize the value of '{value.name}'"
            
            # Check if a similar proto-goal already exists
            if not self._similar_proto_goal_exists(description, [value.id]):
                proto_goal = ProtoGoal(
                    description=description,
                    strength=value.intensity * 0.5,
                    clarity=value.clarity * 0.5,
                    related_values=[value.id],
                    source_type='aspiration'
                )
                self.proto_goals[proto_goal.id] = proto_goal
                self.goal_evaluation_queue.append(proto_goal.id)
                logger.info(f"Generated new proto-goal: {description}")
                
    def _similar_proto_goal_exists(self, description: str, related_values: List[str]) -> bool:
        """Check if a similar proto-goal already exists."""
        for pg in self.proto_goals.values():
            # Check for similar description
            desc_similarity = self._calculate_description_similarity(pg.description, description)
            
            # Check for overlapping related values
            value_overlap = len(set(pg.related_values) & set(related_values))
            
            if desc_similarity > 0.9 and value_overlap > 0:
                return True
                
        return False
        
    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two goal descriptions."""
        # Simple Jaccard similarity on words
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    def _process_goal_evaluation_queue(self) -> None:
        """Process proto-goals to evaluate for promotion."""
        batch_size = min(10, len(self.goal_evaluation_queue))
        if batch_size == 0:
            return
            
        for _ in range(batch_size):
            if not self.goal_evaluation_queue:
                break
                
            proto_id = self.goal_evaluation_queue.popleft()
            proto_goal = self.proto_goals.get(proto_id)
            if not proto_goal:
                continue
                
            # Strengthen and clarify over time
            proto_goal.strengthen(0.05)
            proto_goal.clarify(0.05)
            
            # Check for promotion to full goal
            if proto_goal.strength > 0.6 and proto_goal.clarity > 0.6:
                self._promote_to_full_goal(proto_goal)
            else:
                # Re-queue for later evaluation
                self.goal_evaluation_queue.append(proto_id)
                
    def _promote_to_full_goal(self, proto_goal: ProtoGoal) -> None:
        """Promote a proto-goal to a full goal."""
        # Generate a name for the goal
        name = proto_goal.description[:40] + "..." if len(proto_goal.description) > 40 else proto_goal.description
        
        # Create the full goal
        goal = Goal(
            name=name,
            description=proto_goal.description,
            specificity=proto_goal.clarity,
            importance=proto_goal.strength,
            related_values=proto_goal.related_values,
            complexity=0.5  # Default complexity
        )
        
        self.goals[goal.id] = goal
        
        # Add to goal hierarchy
        self.goal_hierarchy.add_node(goal.id, name=goal.name, importance=goal.importance)
        
        # Decompose into subgoals if complex
        if goal.complexity > self.subgoal_decomposition_threshold:
            self._decompose_into_subgoals(goal)
            
        logger.info(f"Promoted proto-goal to full goal: {goal.name}")
        
    def _decompose_into_subgoals(self, parent_goal: Goal) -> None:
        """Decompose a complex goal into smaller subgoals."""
        # This is a simplified placeholder for a complex planning process
        
        # Example: Create two subgoals
        subgoal1_desc = f"Subgoal 1 for: {parent_goal.name}"
        subgoal2_desc = f"Subgoal 2 for: {parent_goal.name}"
        
        # Create subgoals
        subgoal1 = Goal(
            name=subgoal1_desc,
            description=subgoal1_desc,
            specificity=parent_goal.specificity + 0.1,
            importance=parent_goal.importance * 0.8,
            related_values=parent_goal.related_values,
            parent_goal=parent_goal.id,
            complexity=parent_goal.complexity - 0.2
        )
        
        subgoal2 = Goal(
            name=subgoal2_desc,
            description=subgoal2_desc,
            specificity=parent_goal.specificity + 0.1,
            importance=parent_goal.importance * 0.8,
            related_values=parent_goal.related_values,
            parent_goal=parent_goal.id,
            complexity=parent_goal.complexity - 0.2
        )
        
        # Add to goal collection
        self.goals[subgoal1.id] = subgoal1
        self.goals[subgoal2.id] = subgoal2
        
        # Update parent goal
        parent_goal.subgoals.extend([subgoal1.id, subgoal2.id])
        
        # Add to hierarchy
        self.goal_hierarchy.add_node(subgoal1.id, name=subgoal1.name)
        self.goal_hierarchy.add_node(subgoal2.id, name=subgoal2.name)
        self.goal_hierarchy.add_edge(parent_goal.id, subgoal1.id, type='subgoal')
        self.goal_hierarchy.add_edge(parent_goal.id, subgoal2.id, type='subgoal')
        
        logger.info(f"Decomposed goal '{parent_goal.name}' into 2 subgoals")


class MotivationalSelfModification:
    """System for enabling recursive self-improvement of the motivational system."""
    
    def __init__(self, value_system: ValueFormationSystem, goal_system: GoalFormationSystem):
        self.value_system = value_system
        self.goal_system = goal_system
        
        # Self-modification parameters
        self.modification_threshold = 0.8  # Threshold for triggering self-modification
        self.learning_rate = 0.1  # Rate of parameter adjustment
        
        # Modification history
        self.modification_history = []
        
    def evaluate_and_modify(self) -> None:
        """Evaluate system performance and apply modifications if necessary."""
        # This would be triggered periodically
        
        # Analyze value system for stability and coherence
        value_stability = self._assess_value_stability()
        value_coherence = self._assess_value_coherence()
        
        # Analyze goal system for effectiveness
        goal_effectiveness = self._assess_goal_effectiveness()
        
        # If performance is suboptimal, consider modifications
        if (value_stability < self.modification_threshold or
                value_coherence < self.modification_threshold or
                goal_effectiveness < self.modification_threshold):
            
            logger.warning("Motivational system performance is suboptimal. Considering self-modification.")
            self._apply_modifications(value_stability, value_coherence, goal_effectiveness)
            
    def _assess_value_stability(self) -> float:
        """Assess the overall stability of the value system."""
        if not self.value_system.values:
            return 1.0  # No values, so stable
            
        # Calculate average stability of all values
        total_stability = sum(v.stability for v in self.value_system.values.values())
        avg_stability = total_stability / len(self.value_system.values)
        
        # Consider recent changes in value intensity
        recent_changes = 0
        for value in self.value_system.values.values():
            # Look at expression history in the last hour
            recent_history = [h for h in value.expression_history if time.time() - h[0] < 3600]
            if len(recent_history) > 5:  # If many recent changes
                recent_changes += 1
                
        # Reduce stability score if many values are changing
        stability_penalty = recent_changes / len(self.value_system.values)
        
        return avg_stability * (1.0 - stability_penalty)
        
    def _assess_value_coherence(self) -> float:
        """Assess the coherence of the value network."""
        # Placeholder for a more complex analysis of value relationships
        # For now, we'll just check for cycles in the value graph
        
        cycles = self.value_system.value_relationship_graph.find_cycles()
        
        # Coherence is lower if there are cycles (conflicting support loops)
        coherence = 1.0 - (len(cycles) * 0.1)  # Each cycle reduces coherence
        
        return max(0.0, coherence)
        
    def _assess_goal_effectiveness(self) -> float:
        """Assess how effectively goals are being managed and achieved."""
        if not self.goal_system.goals:
            return 1.0  # No goals, so no ineffectiveness
            
        # Calculate ratio of completed to abandoned goals
        completed_goals = sum(1 for g in self.goal_system.goals.values() if g.status == 'completed')
        abandoned_goals = sum(1 for g in self.goal_system.goals.values() if g.status == 'abandoned')
        
        if completed_goals + abandoned_goals == 0:
            return 0.7  # Neutral score if no goals are finished
            
        completion_ratio = completed_goals / (completed_goals + abandoned_goals)
        
        # Consider average goal progress
        active_goals = [g for g in self.goal_system.goals.values() if g.status == 'active']
        if not active_goals:
            avg_progress = 1.0
        else:
            total_progress = sum(g.progress for g in active_goals)
            avg_progress = total_progress / len(active_goals)
            
        # Combine metrics
        return (completion_ratio * 0.6) + (avg_progress * 0.4)
        
    def _apply_modifications(self, value_stability: float, value_coherence: float, 
                             goal_effectiveness: float) -> None:
        """Apply modifications to system parameters based on performance."""
        
        # Modify value formation parameters
        if value_stability < self.modification_threshold:
            # If system is too unstable, make it harder to form new values
            self.value_system.config.emerging_to_value_threshold += self.learning_rate * 0.1
            self.value_system.config.stability_decay_rate -= self.learning_rate * 0.05
            
            modification = {
                'type': 'parameter_adjustment',
                'system': 'value_formation',
                'parameter': 'emerging_to_value_threshold',
                'new_value': self.value_system.config.emerging_to_value_threshold
            }
            self.modification_history.append(modification)
            logger.info(f"Adjusted value formation threshold due to low stability. New threshold: {self.value_system.config.emerging_to_value_threshold:.3f}")
            
        # Modify goal formation parameters
        if goal_effectiveness < self.modification_threshold:
            # If goals are not being achieved, make them easier to start with
            self.goal_system.goal_generation_threshold -= self.learning_rate * 0.1
            
            modification = {
                'type': 'parameter_adjustment',
                'system': 'goal_formation',
                'parameter': 'goal_generation_threshold',
                'new_value': self.goal_system.goal_generation_threshold
            }
            self.modification_history.append(modification)
            logger.info(f"Adjusted goal generation threshold due to low effectiveness. New threshold: {self.goal_system.goal_generation_threshold:.3f}")
            
        # In a more advanced system, this could involve structural changes,
        # such as modifying the logic of the system itself.
        
        # Example of a structural modification (conceptual)
        if value_coherence < self.modification_threshold:
            # If values are incoherent, prioritize goal of resolving value conflicts
            # This would involve creating a high-priority goal in the goal system
            
            conflict_resolution_goal = ProtoGoal(
                description="Resolve internal value conflicts to improve coherence",
                strength=0.9,
                clarity=0.8,
                source_type='internal_maintenance'
            )
            self.goal_system.proto_goals[conflict_resolution_goal.id] = conflict_resolution_goal
            self.goal_system.goal_evaluation_queue.append(conflict_resolution_goal.id)
            
            modification = {
                'type': 'structural_modification',
                'system': 'goal_formation',
                'action': 'create_conflict_resolution_goal'
            }
            self.modification_history.append(modification)
            logger.warning("Generated high-priority goal to resolve value conflicts due to low coherence.")


class EmergentMotivationSystem:
    """Central coordinator for the emergent motivational system."""
    
    def __init__(self):
        # Initialize subsystems
        self.value_formation_system = ValueFormationSystem()
        self.goal_formation_system = GoalFormationSystem(self.value_formation_system)
        self.motivational_self_modification = MotivationalSelfModification(
            self.value_formation_system, self.goal_formation_system
        )
        self.narrative_framework = NarrativeFramework()
        self.value_conflict_resolution = ValueConflictResolution()
        
        # System state
        self.motivational_history: List[MotivationalSnapshot] = []
        self.is_running = False
        self.main_loop_thread = None
        
        # Timing parameters
        self.snapshot_interval = 60  # seconds
        self.self_modification_interval = 300  # seconds
        
    def start(self) -> None:
        """Start the main event loop of the motivational system."""
        if not self.is_running:
            self.is_running = True
            self.main_loop_thread = threading.Thread(target=self._main_loop)
            self.main_loop_thread.daemon = True
            self.main_loop_thread.start()
            logger.info("Emergent Motivation System started.")
            
    def stop(self) -> None:
        """Stop the main event loop."""
        self.is_running = False
        if self.main_loop_thread:
            self.main_loop_thread.join()
        logger.info("Emergent Motivation System stopped.")
        
    def _main_loop(self) -> None:
        """Main event loop for continuous processing and self-monitoring."""
        last_snapshot_time = 0
        last_modification_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # Trigger goal formation based on current value state
            self.goal_formation_system.analyze_value_system_state()
            
            # Take periodic snapshots of the motivational state
            if current_time - last_snapshot_time > self.snapshot_interval:
                self.take_motivational_snapshot()
                last_snapshot_time = current_time
                
            # Periodically evaluate for self-modification
            if current_time - last_modification_time > self.self_modification_interval:
                self.motivational_self_modification.evaluate_and_modify()
                last_modification_time = current_time
                
            # Detect and resolve value conflicts
            self._manage_value_conflicts()
            
            # Sleep to avoid busy-waiting
            time.sleep(1)
            
    def add_experience(self, experience: Experience) -> None:
        """Add a new experience to the system."""
        self.value_formation_system.add_experience(experience)
        
    def take_motivational_snapshot(self) -> MotivationalSnapshot:
        """Take a snapshot of the current motivational state."""
        snapshot = MotivationalSnapshot(
            active_values={v.id: v.__dict__ for v in self.value_formation_system.values.values()},
            active_goals={g.id: g.__dict__ for g in self.goal_formation_system.goals.values()},
            value_relationships=self.value_formation_system.value_relationship_graph.serialize(),
            goal_relationships=self.goal_formation_system.goal_hierarchy.serialize(),
            system_parameters={
                'value_formation_threshold': self.value_formation_system.config.emerging_to_value_threshold,
                'goal_generation_threshold': self.goal_system.goal_generation_threshold
            }
        )
        
        self.motivational_history.append(snapshot)
        
        # Integrate into narrative framework
        self.narrative_framework.integrate_motivational_development(
            self.value_formation_system.values,
            self.motivational_history
        )
        
        logger.info(f"Took motivational snapshot at {snapshot.timestamp}")
        return snapshot
        
    def _manage_value_conflicts(self) -> None:
        """Detect and manage conflicts between values."""
        # Detect new conflicts
        new_conflicts = self.value_conflict_resolution.detect_conflicts(
            self.value_formation_system.values,
            self.value_formation_system.value_relationship_graph
        )
        
        if new_conflicts:
            logger.warning(f"Detected {len(new_conflicts)} new value conflicts.")
            
            # For each new conflict, generate and apply a resolution strategy
            for conflict in new_conflicts:
                strategies = self.value_conflict_resolution.generate_resolution_strategies(
                    conflict['id'], self.value_formation_system.values
                )
                
                if strategies:
                    # Choose the best strategy (e.g., highest estimated effectiveness)
                    best_strategy = max(strategies, key=lambda s: s['estimated_effectiveness'])
                    
                    logger.info(f"Applying strategy '{best_strategy['type']}' to resolve conflict {conflict['id']}")
                    self.value_conflict_resolution.apply_resolution_strategy(
                        best_strategy['id'], self.value_formation_system.values
                    )
                    
        # Periodically evaluate effectiveness of past resolutions
        for conflict_id in list(self.value_conflict_resolution.detected_conflicts.keys()):
            evaluation = self.value_conflict_resolution.evaluate_resolution_effectiveness(
                conflict_id, self.value_formation_system.values
            )
            if evaluation.get('status') == 'resolved':
                logger.info(f"Conflict {conflict_id} has been resolved.")
                # Optionally remove from active conflict list
                del self.value_conflict_resolution.detected_conflicts[conflict_id]
                
    def get_current_motivation_state(self) -> Dict:
        """Get a summary of the current motivational state."""
        return {
            'values': {v.name: v.intensity for v in self.value_formation_system.values.values()},
            'goals': {g.name: g.importance for g in self.goal_formation_system.goals.values() if g.status == 'active'},
            'narrative': self.narrative_framework.get_current_narrative()
        }
        
    def save_state(self, file_path: str) -> None:
        """Save the entire state of the motivational system to a file."""
        state = {
            'value_system': self.value_formation_system,
            'goal_system': self.goal_formation_system,
            'self_modification': self.motivational_self_modification,
            'narrative_framework': self.narrative_framework,
            'conflict_resolution': self.value_conflict_resolution,
            'history': self.motivational_history
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Motivational system state saved to {file_path}")
        
    @classmethod
    def load_state(cls, file_path: str) -> 'EmergentMotivationSystem':
        """Load the motivational system state from a file."""
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            
        system = cls()
        system.value_formation_system = state['value_system']
        system.goal_formation_system = state['goal_system']
        system.motivational_self_modification = state['self_modification']
        system.narrative_framework = state['narrative_framework']
        system.value_conflict_resolution = state['conflict_resolution']
        system.motivational_history = state['history']
        
        logger.info(f"Motivational system state loaded from {file_path}")
        return system
