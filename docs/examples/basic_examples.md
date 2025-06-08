# Basic Examples

This document provides fundamental examples demonstrating core EG-QGEM functionality for new users.

## Overview

These examples cover basic operations including simulation setup, field computation, and visualization. Each example is self-contained and can be run independently.

## Example 1: Simple Metric Field Simulation

This example demonstrates how to set up and run a basic gravitational field simulation.

### Code

```python
#!/usr/bin/env python3
"""
Basic metric field simulation example.
Simulates a static spherically symmetric gravitational field.
"""

import numpy as np
from egqgem.simulations import Simulation, SimulationConfig
from egqgem.theory import SpacetimeMetric
from egqgem.visualization import FieldVisualizer

def main():
    # Create simulation configuration
    config = SimulationConfig()
    config.grid_size = (64, 64, 64)
    config.grid_bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    config.time_steps = 100
    config.dt = 0.01

    # Initialize simulation
    sim = Simulation(config)

    # Set up initial conditions - Schwarzschild metric approximation
    def schwarzschild_metric(r, M=1.0):
        """Simple Schwarzschild metric components."""
        rs = 2.0 * M  # Schwarzschild radius
        g00 = -(1.0 - rs/r) if r > rs else -0.01
        g11 = 1.0/(1.0 - rs/r) if r > rs else 100.0
        return np.diag([g00, g11, r**2, r**2 * np.sin(np.pi/4)**2])

    # Initialize metric field
    x, y, z = sim.get_coordinates()
    r = np.sqrt(x**2 + y**2 + z**2)

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                radius = r[i, j, k]
                if radius < 0.1:  # Avoid singularity
                    radius = 0.1
                sim.fields.metric[i, j, k] = schwarzschild_metric(radius)

    # Run simulation
    print("Running simulation...")
    sim.run()

    # Visualize results
    visualizer = FieldVisualizer(sim.get_data())

    # Plot metric component
    fig = visualizer.plot_metric_field(
        component='g00',
        slice_type='xy',
        save_path='schwarzschild_g00.png'
    )

    # Create time evolution animation
    anim = visualizer.create_animation(
        field_type='metric',
        component='g00',
        save_path='metric_evolution.mp4'
    )

    print("Results saved to schwarzschild_g00.png and metric_evolution.mp4")

if __name__ == "__main__":
    main()
```

### Expected Output

- Static plot of g₀₀ metric component showing gravitational potential
- Animation showing metric field evolution (should be stable for static case)
- Console output showing simulation progress

### Key Learning Points

1. Basic simulation setup and configuration
2. Initial condition specification for metric fields
3. Simple visualization of results
4. Handling coordinate singularities

## Example 2: Entanglement Field Computation

This example shows how to compute and visualize quantum entanglement fields.

### Code

```python
#!/usr/bin/env python3
"""
Basic entanglement field example.
Demonstrates quantum entanglement tensor computation.
"""

import numpy as np
from egqgem.simulations import Simulation, SimulationConfig
from egqgem.theory import EntanglementTensor
from egqgem.visualization import EntanglementVisualizer

def main():
    # Configuration for entanglement simulation
    config = SimulationConfig()
    config.grid_size = (32, 32, 32)
    config.grid_bounds = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
    config.include_entanglement = True
    config.entanglement_coupling = 0.1

    # Initialize simulation
    sim = Simulation(config)

    # Set up initial entanglement configuration
    # Create two entangled regions
    x, y, z = sim.get_coordinates()

    # Region 1: centered at (-1, 0, 0)
    r1 = np.sqrt((x + 1)**2 + y**2 + z**2)
    region1_mask = r1 < 0.5

    # Region 2: centered at (1, 0, 0)
    r2 = np.sqrt((x - 1)**2 + y**2 + z**2)
    region2_mask = r2 < 0.5

    # Initialize entanglement field
    entanglement_strength = 0.2
    sim.fields.entanglement[region1_mask] = entanglement_strength
    sim.fields.entanglement[region2_mask] = entanglement_strength

    # Add correlation between regions
    correlation_field = np.exp(-(r1 + r2)/2.0) * entanglement_strength
    sim.fields.entanglement += correlation_field

    # Run simulation to evolve entanglement field
    print("Running entanglement simulation...")
    sim.run()

    # Analyze entanglement
    entanglement_data = sim.analyze_entanglement()

    # Visualize results
    ent_vis = EntanglementVisualizer(entanglement_data)

    # Plot entanglement field
    fig1 = ent_vis.plot_entanglement_field(
        component='E00',
        log_scale=True,
        save_path='entanglement_field.png'
    )

    # Plot entanglement network
    fig2 = ent_vis.plot_entanglement_network(
        threshold=0.05,
        save_path='entanglement_network.png'
    )

    # Plot mutual information
    region_pairs = [
        ((-1.5, -0.5, -0.5, 0.5, -0.5, 0.5),
         (0.5, 1.5, -0.5, 0.5, -0.5, 0.5))
    ]
    fig3 = ent_vis.plot_mutual_information(
        region_pairs,
        save_path='mutual_information.png'
    )

    print("Entanglement analysis complete!")
    print(f"Maximum entanglement entropy: {entanglement_data.max_entropy:.3f}")
    print(f"Total mutual information: {entanglement_data.total_mi:.3f}")

if __name__ == "__main__":
    main()
```

### Expected Output

- Visualization of entanglement field distribution
- Network plot showing entangled regions
- Mutual information analysis between regions
- Quantitative entanglement measures

### Key Learning Points

1. Setting up entanglement field initial conditions
2. Computing entanglement evolution
3. Analyzing quantum correlations
4. Visualizing entanglement networks

## Example 3: Combined Gravity-Entanglement System

This example demonstrates the coupling between gravitational and entanglement fields.

### Code

```python
#!/usr/bin/env python3
"""
Combined gravity-entanglement simulation example.
Shows back-reaction of entanglement on spacetime geometry.
"""

import numpy as np
from egqgem.simulations import Simulation, SimulationConfig
from egqgem.analysis import EnergyMomentumAnalyzer
from egqgem.visualization import FieldVisualizer, DiagnosticPlotter

def main():
    # Enhanced configuration for coupled system
    config = SimulationConfig()
    config.grid_size = (48, 48, 48)
    config.grid_bounds = [(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)]
    config.time_steps = 200
    config.dt = 0.005

    # Enable coupling between gravity and entanglement
    config.include_entanglement = True
    config.gravity_entanglement_coupling = 0.05
    config.entanglement_back_reaction = True

    # Initialize simulation
    sim = Simulation(config)

    # Set up initial conditions
    x, y, z = sim.get_coordinates()
    r = np.sqrt(x**2 + y**2 + z**2)

    # Initial gravitational field (weak field approximation)
    mass_density = 2.0 * np.exp(-r**2)  # Gaussian mass distribution
    sim.fields.metric[:, :, :, 0, 0] = -1.0 - 2.0 * mass_density / r
    sim.fields.metric[:, :, :, 1, 1] = 1.0 + 2.0 * mass_density / r
    sim.fields.metric[:, :, :, 2, 2] = 1.0 + 2.0 * mass_density / r
    sim.fields.metric[:, :, :, 3, 3] = 1.0 + 2.0 * mass_density / r

    # Initial entanglement field correlated with mass density
    sim.fields.entanglement = 0.1 * mass_density * np.random.random(r.shape)

    # Add monitoring callbacks
    def monitor_energy(step, sim_data):
        if step % 20 == 0:
            analyzer = EnergyMomentumAnalyzer(sim_data)
            energy = analyzer.compute_total_energy()
            print(f"Step {step}: Total energy = {energy:.6f}")

    sim.add_callback(monitor_energy)

    # Run coupled simulation
    print("Running coupled gravity-entanglement simulation...")
    sim.run()

    # Analyze results
    data = sim.get_data()

    # Energy-momentum analysis
    em_analyzer = EnergyMomentumAnalyzer(data)
    energy_evolution = em_analyzer.energy_evolution()
    stress_tensor = em_analyzer.compute_stress_tensor()

    # Visualize results
    visualizer = FieldVisualizer(data)
    diagnostic = DiagnosticPlotter(sim)

    # Plot final metric field
    fig1 = visualizer.plot_metric_field(
        component='g00',
        time_slice=-1,  # Final time
        save_path='final_metric.png'
    )

    # Plot entanglement back-reaction
    fig2 = visualizer.plot_entanglement_field(
        component='E00',
        time_slice=-1,
        save_path='final_entanglement.png'
    )

    # Create evolution comparison
    fig3 = visualizer.plot_field_evolution(
        field_type='metric',
        point=(0.0, 0.0, 0.0),
        components=['g00'],
        save_path='metric_evolution.png'
    )

    # Diagnostic plots
    fig4 = diagnostic.plot_energy_conservation(
        save_path='energy_conservation.png'
    )

    fig5 = diagnostic.plot_constraint_violations(
        save_path='constraint_violations.png'
    )

    # Print summary
    print("\nSimulation Summary:")
    print(f"Final total energy: {energy_evolution[-1]:.6f}")
    print(f"Energy conservation error: {abs(energy_evolution[-1] - energy_evolution[0]):.2e}")
    print(f"Maximum entanglement: {np.max(data.entanglement_field):.4f}")
    print("All plots saved successfully!")

if __name__ == "__main__":
    main()
```

### Expected Output

- Evolution of coupled gravitational and entanglement fields
- Energy conservation monitoring
- Constraint violation analysis
- Quantitative measures of back-reaction effects

### Key Learning Points

1. Setting up coupled field systems
2. Monitoring conservation laws during simulation
3. Analyzing back-reaction effects
4. Comprehensive diagnostic analysis

## Example 4: Data Analysis and Visualization

This example focuses on post-processing and analysis of simulation results.

### Code

```python
#!/usr/bin/env python3
"""
Data analysis and visualization example.
Demonstrates comprehensive analysis of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from egqgem.simulations import SimulationData
from egqgem.analysis import FieldAnalyzer, StatisticalAnalyzer
from egqgem.visualization import FieldVisualizer, InteractivePlotter
from egqgem.utilities import DataManager

def main():
    # Load existing simulation data
    # (This assumes you have run one of the previous examples)
    try:
        data = DataManager.load_simulation_data('simulation_results.h5')
    except FileNotFoundError:
        print("No simulation data found. Please run Example 1, 2, or 3 first.")
        return

    # Create analyzers
    field_analyzer = FieldAnalyzer(data.metric_field, data.coordinates)
    stats_analyzer = StatisticalAnalyzer(data)

    # Perform field analysis
    print("Analyzing field properties...")

    # Find field extrema
    maxima = field_analyzer.find_extrema('maximum')
    minima = field_analyzer.find_extrema('minimum')

    print(f"Found {len(maxima)} field maxima and {len(minima)} minima")

    # Compute field statistics
    stats = field_analyzer.compute_statistics()
    print(f"Field statistics:")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std:  {stats['std']:.6f}")
    print(f"  Min:  {stats['min']:.6f}")
    print(f"  Max:  {stats['max']:.6f}")

    # Detect coherent features
    features = field_analyzer.detect_features(threshold=0.1)
    print(f"Detected {len(features)} coherent features")

    # Power spectrum analysis
    frequencies, power_spectrum = field_analyzer.compute_power_spectrum()

    # Statistical analysis
    correlations = stats_analyzer.compute_correlations()
    fluctuations = stats_analyzer.analyze_fluctuations()

    # Create comprehensive visualizations
    visualizer = FieldVisualizer(data)

    # Multi-panel field comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Metric field
    axes[0,0].imshow(data.metric_field[:, :, data.grid_size[2]//2, 0, 0])
    axes[0,0].set_title('Metric Field g₀₀')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')

    # Entanglement field
    if hasattr(data, 'entanglement_field'):
        axes[0,1].imshow(data.entanglement_field[:, :, data.grid_size[2]//2])
        axes[0,1].set_title('Entanglement Field')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')

    # Power spectrum
    axes[1,0].loglog(frequencies, power_spectrum)
    axes[1,0].set_title('Power Spectrum')
    axes[1,0].set_xlabel('Frequency')
    axes[1,0].set_ylabel('Power')
    axes[1,0].grid(True)

    # Field evolution at center
    center_evolution = data.metric_field[
        data.grid_size[0]//2,
        data.grid_size[1]//2,
        data.grid_size[2]//2,
        0, 0, :
    ]
    axes[1,1].plot(data.time_points, center_evolution)
    axes[1,1].set_title('Field Evolution at Center')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('g₀₀')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300)
    plt.show()

    # Create interactive visualization
    print("Creating interactive visualization...")
    interactive = InteractivePlotter(data)

    # Create field explorer (saves HTML file)
    field_explorer = interactive.create_field_explorer(
        field_types=['metric', 'entanglement'],
        save_html='field_explorer.html'
    )

    print("Interactive field explorer saved as 'field_explorer.html'")

    # Export analysis results
    analysis_results = {
        'field_statistics': stats,
        'extrema_locations': {'maxima': maxima, 'minima': minima},
        'coherent_features': features,
        'power_spectrum': {'frequencies': frequencies, 'power': power_spectrum},
        'correlations': correlations,
        'fluctuation_analysis': fluctuations
    }

    DataManager.save_analysis_results(analysis_results, 'analysis_results.json')

    print("\nAnalysis complete!")
    print("Results saved to:")
    print("  - comprehensive_analysis.png")
    print("  - field_explorer.html")
    print("  - analysis_results.json")

if __name__ == "__main__":
    main()
```

### Expected Output

- Comprehensive statistical analysis of field data
- Multi-panel visualization comparing different fields
- Power spectrum analysis
- Interactive HTML visualization tool
- Exported analysis results in JSON format

### Key Learning Points

1. Loading and analyzing existing simulation data
2. Computing field statistics and detecting features
3. Creating multi-panel publication-quality plots
4. Generating interactive visualizations
5. Exporting analysis results for further use

## Running the Examples

### Prerequisites

Ensure you have the EG-QGEM framework installed:

```bash
pip install egqgem
# or if installing from source:
pip install -e .
```

### Execution

Run each example individually:

```bash
python basic_example_1.py
python basic_example_2.py
python basic_example_3.py
python basic_example_4.py
```

### Expected Runtime

- Example 1: ~2-5 minutes (depending on hardware)
- Example 2: ~3-7 minutes
- Example 3: ~5-10 minutes
- Example 4: ~1-2 minutes (analysis only)

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce grid_size in configuration
2. **Slow Performance**: Decrease time_steps or increase dt
3. **Visualization Issues**: Check matplotlib backend settings
4. **File Not Found**: Ensure previous examples ran successfully

### Performance Tips

1. Start with smaller grid sizes for testing
2. Use fewer time steps for initial exploration
3. Enable parallel processing for larger simulations
4. Monitor memory usage during execution

## Next Steps

After running these basic examples, consider:

1. Modifying parameters to explore different physics
2. Implementing custom initial conditions
3. Adding your own analysis functions
4. Exploring the advanced examples in the next section

For more sophisticated scenarios, see the [Advanced Examples](advanced_examples.md) documentation.
