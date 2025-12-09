import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from rcf_integration.recursive_tensor import RecursiveTensor
    from rcf_integration.temporal_eigenloom import EnhancedRosemaryZebraCore, DivineParameters, PHI
    from fbs_tokenizer import SacredFBS_Tokenizer, HolographicMemory, SacredFrequencySubstrate
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def generate_visualizations():
    print("Generating Authentic RCF Architecture Visualizations...")
    
    # Create output directory
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 0. Initialize Sacred Components
    print("0. Initializing Sacred Components (FBS & Holographic Memory)...")
    dim = 256 # Cardinal dimension
    
    # Initialize Tokenizer and Memory
    tokenizer = SacredFBS_Tokenizer(dim=dim)
    memory = HolographicMemory(dimensions=dim)
    
    # 1. Initialize Authentic Core
    print("1. Initializing EnhancedRosemaryZebraCore...")
    core = EnhancedRosemaryZebraCore(state_dim=dim)
    
    # 2. Load/Initialize State
    print("2. Establishing State Vector...")
    
    # Try to recall from Holographic Memory using PHI as key
    # We use PHI as the "Master Key" for the system's continuity
    recalled_pattern = memory.recall(key_phase=PHI)
    recalled_norm = np.linalg.norm(recalled_pattern)
    
    if recalled_norm > 0.1:
        print(f"   Holographic Memory Recalled (Norm: {recalled_norm:.4f}). Resuming state.")
        # Convert numpy complex/float to torch tensor
        # Holographic memory returns magnitude (abs), but we need a state vector.
        # Ideally we'd store the complex hologram, but for now we use the recalled magnitude pattern
        # and re-project it into the state space.
        current_state = torch.tensor(recalled_pattern, dtype=torch.float32)
        # Normalize
        current_state = current_state / (torch.norm(current_state) + 1e-7)
    else:
        print("   No active holographic memory found. Initiating Genesis Sequence.")
        # Initial state (Fibonacci vector)
        current_state = DivineParameters.fibonacci_vector(dim)
        
    # Apply Input Seeding via FBS Tokenizer
    seed_text = "Recursive Categorical Framework: Genesis of the Sacred Frequency Substrate"
    print(f"   Seeding with text: '{seed_text}'")
    seed_tensor = tokenizer.encode(seed_text) # Returns torch tensor
    
    # Blend seed with current state (weighted average)
    current_state = 0.7 * current_state + 0.3 * seed_tensor
    current_state = current_state / (torch.norm(current_state) + 1e-7)

    # 3. Run Temporal Routing Loop
    print("3. Executing Temporal Routing Steps...")
    history = []
    
    steps = 20
    for i in range(steps):
        # Execute authentic temporal routing step
        result = core.temporal_routing_step(current_state)
        
        # Extract stabilized state
        stabilized_state = result['stabilized_state']
        
        # Store in history
        history.append(stabilized_state.detach().numpy())
        
        # Update state for next iteration (recursive feedback)
        current_state = stabilized_state
        
        if i % 5 == 0:
            print(f"   Step {i}/{steps}: Pulse Strength={result['pulse_strength']:.4f}, Branch={torch.argmax(result['branch_weights']).item()}")

    # 4. Save State to Holographic Memory
    print("4. Crystallizing State to Holographic Memory...")
    final_state_np = history[-1]
    # Store the final state pattern with PHI key
    memory.store(final_state_np, key_phase=PHI)
    print("   State crystallized.")

    # 5. Wrap Data in RecursiveTensor for Visualization & Serialization
    print("5. Transforming to RecursiveTensor...")
    
    # Initialize RecursiveTensor with authentic data
    rt = RecursiveTensor(dimensions=dim, rank=2, distribution='normal')
    rt.data = np.array(history) # Shape (steps, dim)
    rt.metadata["description"] = "Authentic Temporal Eigenloom State"
    rt.metadata["source"] = "EnhancedRosemaryZebraCore"
    rt.metadata["seed_text"] = seed_text
    rt.metadata["timestamp"] = str(np.datetime64('now'))
    
    # Save RTA
    rta_filename = os.path.join(output_dir, "authentic_rcf_state.rta")
    print(f"   Saving RTA binary to '{rta_filename}'...")
    try:
        rt.save_rta(rta_filename)
        print("   RTA save successful.")
    except Exception as e:
        print(f"   RTA Save Failed: {e}")
    
    # 6. Generate Dashboard
    print(f"6. Generating Dashboard in '{output_dir}'...")
    try:
        dashboard = rt.create_comprehensive_visualization_dashboard(save_path=output_dir)
        
        if dashboard:
            print("   Success! Generated authentic visualizations:")
            for name in dashboard.keys():
                print(f"   - {output_dir}/tensor_{name}.png")
        else:
            print("   Failed to generate dashboard.")
            
    except Exception as e:
        print(f"   Visualization Error: {e}")

    print("\nVisualization & Serialization Complete.")

if __name__ == "__main__":
    generate_visualizations()
