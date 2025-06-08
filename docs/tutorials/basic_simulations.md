# Basic Simulations Tutorial

## 🎯 Running Your First EG-QGEM Simulation

This tutorial will guide you through running your first simulations with EG-QGEM, from simple test cases to more complex scenarios.

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ Completed the [Quick Start Guide](quick_start.md)
- ✅ Successfully run `python verify_gui.py`
- ✅ Basic understanding of [Core Principles](../theory/core_principles.md)

## 🏃‍♂️ Your First Simulation

### Option 1: Command Line Interface

```bash
# Navigate to the project directory
cd TOE

# Run a basic spacetime emergence simulation
python -m simulations.basic_emergence --grid-size 32 --time-steps 100 --output results/first_sim/
```

**Expected Output:**

```
🌌 EG-QGEM Basic Emergence Simulation
📊 Grid Size: 32³ = 32,768 points
⏱️  Time Steps: 100
💾 Output Directory: results/first_sim/

⚡ Starting simulation...
   Step   10/100 | Time: 0.10s | Energy: -1.234e-03
   Step   20/100 | Time: 0.20s | Energy: -1.235e-03
   ...
✅ Simulation completed successfully!
📈 Results saved to: results/first_sim/
```

### Option 2: Python Script

Create a file called `my_first_simulation.py`:

```python
#!/usr/bin/env python3
"""
My First EG-QGEM Simulation
"""

from simulations.spacetime_emergence import SpacetimeEmergenceSimulator
from theory.entanglement_dynamics import EntanglementField
from visualization.basic_plots import SimulationVisualizer

def run_basic_simulation():
    """Run a basic spacetime emergence simulation"""

    # Initialize the simulator
    simulator = SpacetimeEmergenceSimulator(
        grid_size=(32, 32, 32),
        spatial_extent=10.0,
        time_step=0.01
    )

    # Set initial conditions
    initial_entanglement = EntanglementField.create_gaussian_distribution(
        center=(5.0, 5.0, 5.0),
        width=2.0,
        amplitude=1.0
    )

    simulator.set_initial_conditions(
        entanglement_field=initial_entanglement,
        metric_perturbations=None  # Start with flat spacetime
    )

    # Run simulation
    print("🌌 Starting basic emergence simulation...")
    results = simulator.evolve(num_steps=100, save_interval=10)

    # Visualize results
    visualizer = SimulationVisualizer()
    visualizer.plot_entanglement_evolution(results['entanglement_history'])
    visualizer.plot_metric_emergence(results['metric_history'])
    visualizer.save_animation('my_first_simulation.mp4')

    print("✅ Simulation completed! Check the output files.")

    return results

if __name__ == "__main__":
    results = run_basic_simulation()
```

Run your script:

```bash
python my_first_simulation.py
```

## 🔍 Understanding the Results

### Output Files

After running a simulation, you'll find several output files:

```
results/first_sim/
├── simulation_parameters.json    # Input parameters used
├── entanglement_data.h5         # Entanglement field evolution
├── metric_data.h5               # Spacetime metric evolution
├── energy_conservation.csv      # Energy conservation check
├── diagnostics.json             # Simulation diagnostics
└── visualizations/
    ├── entanglement_3d.png      # 3D entanglement visualization
    ├── metric_curvature.png     # Spacetime curvature plot
    └── evolution_animation.mp4  # Time evolution movie
```

### Key Metrics to Examine

1. **Energy Conservation**: Should remain constant (within numerical errors)
2. **Entanglement Entropy**: Measures quantum correlations
3. **Spacetime Curvature**: Shows how geometry responds to entanglement
4. **Constraint Violations**: Should be small (< 10⁻⁸)

### Reading the Data

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load simulation results
with h5py.File('results/first_sim/entanglement_data.h5', 'r') as f:
    entanglement_data = f['entanglement_field'][:]
    time_stamps = f['time'][:]

# Plot entanglement evolution at center point
center_idx = entanglement_data.shape[1] // 2
center_evolution = entanglement_data[:, center_idx, center_idx, center_idx]

plt.figure(figsize=(10, 6))
plt.plot(time_stamps, center_evolution)
plt.xlabel('Time')
plt.ylabel('Entanglement Density')
plt.title('Entanglement Evolution at Center')
plt.grid(True)
plt.show()
```

## 🎛️ Basic Simulation Types

### 1. Spacetime Emergence

Study how spacetime geometry emerges from entanglement:

```python
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator

# Create simulator
sim = SpacetimeEmergenceSimulator(grid_size=(64, 64, 64))

# Set Gaussian entanglement distribution
sim.set_gaussian_entanglement(center=(32, 32, 32), width=5.0)

# Evolve and watch spacetime emerge
results = sim.evolve(num_steps=200)
```

### 2. Black Hole Formation

Simulate black hole formation from entanglement concentration:

```python
from simulations.black_hole_dynamics import BlackHoleSimulator

# Create black hole simulator
sim = BlackHoleSimulator(
    grid_size=(128, 128, 128),
    central_mass=1.0,
    initial_entanglement='concentrated'
)

# Run formation simulation
results = sim.simulate_formation(evolution_time=10.0)
```

### 3. Quantum Decoherence

Study environmental decoherence effects:

```python
from simulations.decoherence_study import DecoherenceSimulator

# Create decoherence simulator
sim = DecoherenceSimulator(
    system_size=10,
    environment_size=100,
    coupling_strength=0.1
)

# Study decoherence timescales
results = sim.measure_decoherence_time()
```

### 4. Cosmological Evolution

Simulate expanding universe scenarios:

```python
from simulations.cosmology import CosmologicalSimulator

# Create cosmological simulator
sim = CosmologicalSimulator(
    box_size=100.0,
    initial_conditions='nearly_uniform',
    hubble_parameter=0.7
)

# Evolve cosmic structure
results = sim.evolve_structure_formation(redshift_range=(10, 0))
```

## 🔧 Simulation Parameters

### Common Parameters

```python
SIMULATION_PARAMETERS = {
    # Grid Configuration
    'grid_size': (64, 64, 64),        # Spatial resolution
    'spatial_extent': 10.0,           # Physical size (Planck units)
    'boundary_conditions': 'periodic', # 'periodic', 'absorbing', 'reflecting'

    # Time Evolution
    'time_step': 0.01,                # Time step (Planck units)
    'total_time': 10.0,               # Total evolution time
    'adaptive_stepping': True,         # Adaptive time steps

    # Physical Parameters
    'planck_mass': 1.0,               # Planck mass (natural units)
    'coupling_strength': 1.0,          # Entanglement-gravity coupling
    'decoherence_rate': 0.01,         # Environmental decoherence

    # Numerical Parameters
    'tolerance': 1e-8,                # Numerical tolerance
    'max_iterations': 1000,           # Maximum solver iterations
    'conservation_check': True,        # Monitor conservation laws

    # Output Control
    'save_interval': 10,              # Save every N steps
    'output_format': 'hdf5',          # 'hdf5', 'numpy', 'ascii'
    'compression': True,              # Compress output files
}
```

### Adjusting Parameters for Different Physics

```python
# For black hole simulations - high resolution near center
BLACK_HOLE_PARAMS = {
    'grid_size': (128, 128, 128),
    'adaptive_mesh_refinement': True,
    'refinement_levels': 5,
    'time_step': 0.001,  # Smaller time step for stability
}

# For cosmological simulations - large volume, coarser resolution
COSMOLOGY_PARAMS = {
    'grid_size': (256, 256, 256),
    'spatial_extent': 1000.0,  # Much larger volume
    'time_step': 0.1,          # Larger time step for efficiency
}

# For quantum decoherence - fine time resolution
DECOHERENCE_PARAMS = {
    'time_step': 0.0001,       # Very fine time steps
    'total_time': 1.0,         # Shorter total time
    'decoherence_rate': 0.1,   # Strong environmental coupling
}
```

## 📊 Monitoring Simulation Progress

### Real-time Monitoring

```python
def monitor_simulation_progress(simulator):
    """Monitor simulation in real-time"""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Set up real-time plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def update_plots(frame):
        # Get current simulation state
        state = simulator.get_current_state()

        # Update entanglement plot
        axes[0,0].clear()
        axes[0,0].imshow(state['entanglement'][:,:,16], cmap='viridis')
        axes[0,0].set_title(f'Entanglement (t={state["time"]:.2f})')

        # Update curvature plot
        axes[0,1].clear()
        axes[0,1].imshow(state['curvature'][:,:,16], cmap='RdBu')
        axes[0,1].set_title('Spacetime Curvature')

        # Update energy conservation
        axes[1,0].clear()
        axes[1,0].plot(state['time_history'], state['energy_history'])
        axes[1,0].set_title('Energy Conservation')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Total Energy')

        # Update constraint violations
        axes[1,1].clear()
        axes[1,1].semilogy(state['time_history'], state['constraint_violations'])
        axes[1,1].set_title('Constraint Violations')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Violation Magnitude')

    # Create animation
    anim = FuncAnimation(fig, update_plots, interval=1000, blit=False)
    plt.show()

    return anim
```

### Progress Callbacks

```python
def create_progress_callback():
    """Create a callback function to monitor progress"""

    def callback(step, time, state):
        # Print progress
        if step % 10 == 0:
            energy = np.sum(state['energy_density'])
            max_curvature = np.max(np.abs(state['curvature']))

            print(f"Step {step:4d} | Time: {time:6.3f} | "
                  f"Energy: {energy:8.3e} | Max Curvature: {max_curvature:8.3e}")

        # Check for problems
        if np.any(np.isnan(state['entanglement'])):
            raise RuntimeError(f"NaN detected in entanglement field at step {step}")

        if max_curvature > 1e3:
            print(f"⚠️  Warning: Very large curvature detected at step {step}")

    return callback

# Use the callback in simulation
simulator.set_progress_callback(create_progress_callback())
results = simulator.evolve(num_steps=500)
```

## 🐛 Troubleshooting Common Issues

### Issue 1: Simulation Crashes with NaN Values

**Symptoms:**

```
RuntimeError: NaN detected in entanglement field at step 42
```

**Solutions:**

1. Reduce time step: `time_step = 0.005` instead of `0.01`
2. Enable adaptive stepping: `adaptive_stepping = True`
3. Check initial conditions: avoid extreme values
4. Increase numerical tolerance: `tolerance = 1e-6`

### Issue 2: Very Slow Performance

**Symptoms:**

- Simulation takes hours for simple problems
- High memory usage

**Solutions:**

1. Reduce grid size: `grid_size = (32, 32, 32)` for testing
2. Enable GPU acceleration: `use_gpu = True`
3. Use lower precision: `precision = 'float32'`
4. Enable compression: `compression = True`

### Issue 3: Unphysical Results

**Symptoms:**

- Energy not conserved
- Negative entanglement values
- Constraint violations > 1e-6

**Solutions:**

1. Check boundary conditions
2. Verify initial conditions are physical
3. Reduce time step
4. Enable constraint damping

### Issue 4: Memory Errors

**Symptoms:**

```
MemoryError: Unable to allocate array
```

**Solutions:**

1. Reduce grid size
2. Enable streaming mode: `streaming = True`
3. Use data compression: `compression = True`
4. Reduce save frequency: `save_interval = 100`

## ✅ Validation and Testing

### Verify Your Results

```python
def validate_simulation_results(results):
    """Validate simulation results for physical consistency"""

    checks = {
        'energy_conservation': check_energy_conservation(results),
        'constraint_satisfaction': check_constraint_violations(results),
        'causality': check_causality_conditions(results),
        'positivity': check_entanglement_positivity(results)
    }

    print("🔍 Validation Results:")
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name}: {status}")

    return all(checks.values())

# Run validation
is_valid = validate_simulation_results(results)
if is_valid:
    print("🎉 All validation checks passed!")
else:
    print("⚠️  Some validation checks failed - review results carefully")
```

### Compare with Known Solutions

```python
def compare_with_analytics():
    """Compare numerical results with analytical solutions"""

    # Run simulation with known initial conditions
    sim = SpacetimeEmergenceSimulator(grid_size=(32, 32, 32))
    sim.set_known_initial_conditions('flat_spacetime')
    results = sim.evolve(num_steps=100)

    # Compare with analytical flat spacetime solution
    analytical_metric = np.eye(4)  # Minkowski metric
    numerical_metric = results['final_metric']

    difference = np.max(np.abs(numerical_metric - analytical_metric))
    print(f"Difference from analytical solution: {difference:.2e}")

    return difference < 1e-6  # Should be small for flat spacetime
```

## 🎯 Next Steps

After mastering basic simulations, you can:

1. **Explore Advanced Examples**: Check out [advanced_examples.md](advanced_examples.md)
2. **Learn the GUI Interface**: See [gui_interface.md](gui_interface.md)
3. **Study the Theory**: Dive deeper into [mathematical_framework.md](../theory/mathematical_framework.md)
4. **Contribute**: Help improve the code and documentation

## 📚 Additional Resources

- **Research Papers**: See [published_results.md](../research/published_results.md)
- **API Documentation**: Complete reference in [api/](../api/)
- **Example Scripts**: Browse [examples/](../examples/)
- **Community Forum**: Join discussions about theoretical and computational aspects

Happy simulating! 🌌
