# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

This guide will get you from zero to running EG-QGEM simulations in just a few minutes.

## 📋 Prerequisites

- **Python 3.9+** installed on your system
- **Git** for version control
- **8GB RAM** minimum (16GB recommended)
- **Terminal/Command Prompt** access

## ⚡ Rapid Setup

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd TOE

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Run verification script
python verify_gui.py
```

You should see output like:

```
🎉 All components verified successfully!
✅ Ready to run EG-QGEM GUI interface
```

### Step 3: Run Your First Simulation

#### Option A: Graphical Interface (Recommended)

```bash
python launch_gui.py
```

#### Option B: Command Line Interface

```bash
python research_interface.py
```

#### Option C: Interactive Notebook

```bash
jupyter notebook notebooks/EG-QGEM_Interactive_Research.ipynb
```

## 🎯 Your First Simulation

### Using the GUI Interface

1. **Launch the GUI**: `python launch_gui.py`
2. **Select Simulation Type**: Choose "Spacetime Emergence" from dropdown
3. **Set Parameters**:
   - Subsystems: 50 (default)
   - Evolution Steps: 100 (default)
   - Pattern: "local" (default)
4. **Click "Run Simulation"**
5. **Watch the Progress**: Monitor in the log panel
6. **View Results**: Real-time plots appear in the visualization panel

### Using the Command Line

```bash
# Run complete research workflow
python research_interface.py

# Run specific simulation
python simulations/spacetime_emergence.py

# Run with custom parameters
python -c "
from simulations.spacetime_emergence import run_emergence_simulation
simulator, data = run_emergence_simulation(n_subsystems=30, steps=50)
print(f'Total entanglement: {simulator.total_entanglement:.2f}')
"
```

## 📊 Understanding Your Results

### Spacetime Emergence Simulation

After running the simulation, you'll see:

1. **Entanglement Evolution Plot**: Shows how quantum entanglement grows over time
2. **Dimensional Emergence**: Displays the effective dimensionality of spacetime
3. **Geometry Metrics**: Quantifies the emergent geometric properties

**Key Metrics**:

- **Total Entanglement**: Overall quantum correlation (higher = more curved spacetime)
- **Average Entanglement**: Per-subsystem correlation strength
- **Dimensional Measure**: Effective spatial dimensions (should approach 3)

### Example Output

```
Simulation completed in 5.90 seconds
Total entanglement: 135.47
Average entanglement: 2.71
Effective dimensions: 2.95
```

## 🔧 Quick Customization

### Adjust Simulation Parameters

```python
# In Python script or notebook
from simulations.spacetime_emergence import run_emergence_simulation

# Small, fast simulation
simulator, data = run_emergence_simulation(
    n_subsystems=20,    # Fewer subsystems = faster
    steps=50,           # Fewer steps = quicker results
    pattern='local'     # Local entanglement pattern
)

# Large, detailed simulation
simulator, data = run_emergence_simulation(
    n_subsystems=100,   # More subsystems = detailed results
    steps=200,          # More steps = longer evolution
    pattern='global'    # Global entanglement pattern
)
```

### Modify GUI Parameters

1. **Launch GUI**: `python launch_gui.py`
2. **Adjust Sliders**: Use the parameter controls in the left panel
3. **Try Different Patterns**: Switch between 'local', 'global', 'random'
4. **Export Results**: Use the "Export" buttons to save data and plots

## 🎮 Interactive Exploration

### Jupyter Notebook Workflow

```bash
# Start Jupyter
jupyter notebook

# Open the research notebook
# Navigate to: notebooks/EG-QGEM_Interactive_Research.ipynb
```

The notebook includes:

- **Interactive simulations** with real-time parameter adjustment
- **Detailed explanations** of the physics
- **Visualization examples** for all simulation types
- **Research workflows** for systematic studies

### Command Line Examples

```bash
# Quick black hole simulation
python -c "
from simulations.black_hole_simulator import BlackHoleSimulator
bh = BlackHoleSimulator(mass=10, charge=0, angular_momentum=0.5)
results = bh.compute_hawking_radiation(100)
print(f'Hawking temperature: {results[\"temperature\"]:.2e} K')
"

# Generate experimental predictions
python -c "
from experiments.predictions import generate_experimental_predictions
predictions = generate_experimental_predictions()
for exp, pred in predictions.items():
    print(f'{exp}: {pred[\"description\"]}')
"
```

## 🔍 Troubleshooting

### Common Issues

**1. ImportError: No module named 'PyQt5'**

```bash
pip install PyQt5
```

**2. "No display detected" (for GUI)**

```bash
# For headless servers
export QT_QPA_PLATFORM=offscreen
python launch_gui.py
```

**3. Simulation runs slowly**

- Reduce number of subsystems (try 20-30)
- Decrease evolution steps (try 50-100)
- Use 'local' entanglement pattern

**4. Out of memory errors**

- Reduce matrix sizes in black hole simulations
- Use fewer subsystems in spacetime emergence
- Close other applications to free RAM

### Getting Help

```bash
# Check system compatibility
python verify_gui.py

# Run tests to identify issues
python -m pytest tests/ -v

# View detailed error logs
python research_interface.py --verbose
```

## 📈 Next Steps

### 1. Explore All Simulation Types

- **Spacetime Emergence**: Quantum entanglement → geometric spacetime
- **Black Hole Dynamics**: Hawking radiation and information scrambling
- **Experimental Predictions**: Testable theoretical predictions

### 2. Learn the Theory

- Read [Core Principles](../theory/core_principles.md)
- Study [Mathematical Framework](../theory/mathematical_framework.md)
- Review [Published Results](../research/published_results.md)

### 3. Advanced Usage

- Try [Advanced Examples](advanced_examples.md)
- Explore [Research Workflows](../examples/research_workflows.md)
- Contribute to [Ongoing Studies](../research/ongoing_studies.md)

### 4. Customize and Extend

- Modify simulation parameters
- Create custom analysis scripts
- Develop new visualization tools
- Contribute improvements to the codebase

## 🎉 Congratulations

You now have a working EG-QGEM research platform. You can:

- ✅ Run quantum gravity simulations
- ✅ Visualize emergent spacetime geometry
- ✅ Generate theoretical predictions
- ✅ Analyze results with professional tools

**Ready to explore the quantum origins of spacetime?** 🌌

---

**Estimated Time**: 5-10 minutes
**Difficulty**: Beginner
**Next Tutorial**: [Basic Simulations](basic_simulations.md)
