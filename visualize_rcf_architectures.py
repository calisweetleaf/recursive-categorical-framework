import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from rcf_integration.recursive_tensor import RecursiveTensor
    from rcf_integration.temporal_eigenloom import EnhancedRosemaryZebraCore, DivineParameters
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def generate_visualizations():
    print("Generating Authentic RCF Architecture Visualizations...")
    
    # Create output directory
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Initialize Authentic Core
    print("1. Initializing EnhancedRosemaryZebraCore...")
    dim = 256 # Cardinal dimension
    core = EnhancedRosemaryZebraCore(state_dim=dim)
    
    # 2. Run Temporal Routing Loop
    print("2. Executing Temporal Routing Steps...")
    history = []
    
    # Initial state (Fibonacci vector)
    current_state = DivineParameters.fibonacci_vector(dim)
    
    steps = 20
    for i in range(steps):
        # Execute authentic temporal routing step
        result = core.temporal_routing_step(current_state)
        
        # Extract stabilized state
        stabilized_state = result['stabilized_state']
        history.append(stabilized_state.detach().numpy())
        
        # Update state for next iteration (recursive feedback)
        current_state = stabilized_state
        
        if i % 5 == 0:
            print(f"   Step {i}/{steps}: Pulse Strength={result['pulse_strength']:.4f}, Branch={torch.argmax(result['branch_weights']).item()}")

    # 3. Wrap Data in RecursiveTensor for Visualization
    print("3. Transforming to RecursiveTensor for Dashboard Generation...")
    
    # Create a RecursiveTensor from the final state
    # We use the history to populate the tensor's internal state for "evolution" plots
    final_state_np = history[-1]
    
    # Initialize RecursiveTensor with authentic data
    rt = RecursiveTensor(dimensions=dim, rank=2, distribution='custom')
    rt.data = np.array(history) # Shape (steps, dim) - treating time as a dimension for visualization
    rt.metadata["description"] = "Authentic Temporal Eigenloom State"
    rt.metadata["source"] = "EnhancedRosemaryZebraCore"
    
    # 4. Generate Dashboard
    print(f"4. Generating Dashboard in '{output_dir}'...")
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

    print("\nVisualization Complete.")

if __name__ == "__main__":
    generate_visualizations()
