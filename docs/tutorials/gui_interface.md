# GUI Interface Tutorial

## 🖥️ Overview

The EG-QGEM Graphical User Interface provides an intuitive way to set up, run, and analyze simulations without writing code. This tutorial covers all aspects of using the GUI effectively.

## 🚀 Launching the GUI

### Quick Launch

```bash
# From the TOE directory
python launch_gui.py
```

**Expected Output:**

```
🌌 EG-QGEM Research Platform
🚀 Launching GUI interface...
✅ GUI loaded successfully!
🌐 Server running at: http://localhost:8080
```

### Alternative Launch Methods

```bash
# Method 1: Direct GUI module
python gui_interface.py

# Method 2: Research interface (advanced features)
python research_interface.py

# Method 3: Simple test GUI
python simple_gui_test.py
```

## 🎨 GUI Overview

### Main Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│ EG-QGEM Research Platform                          [?] [⚙️] │
├─────────────────────────────────────────────────────────────┤
│ 📊 Dashboard │ 🔬 Simulations │ 📈 Analysis │ 📚 Research  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Main Content Area                                          │
│  (Changes based on selected tab)                            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Status: Ready │ Progress: ████████░░ 80% │ Time: 00:02:15  │
└─────────────────────────────────────────────────────────────┘
```

### Navigation Tabs

1. **📊 Dashboard**: System overview and quick access
2. **🔬 Simulations**: Set up and run simulations
3. **📈 Analysis**: Visualize and analyze results
4. **📚 Research**: Advanced research tools

## 📊 Dashboard Tab

### System Status Panel

The dashboard provides real-time system information:

- **CPU Usage**: Current processor utilization
- **Memory Usage**: RAM consumption and availability
- **GPU Status**: Graphics card availability and usage
- **Storage**: Available disk space for results

### Quick Start Panel

**Recent Simulations**

- List of recently run simulations
- One-click access to results
- Status indicators (Running, Completed, Failed)

**Template Simulations**

- Pre-configured simulation templates
- One-click launch for common scenarios
- Educational examples

### Performance Metrics

**Real-time Charts**

- Simulation performance over time
- Resource utilization graphs
- Error rate monitoring

## 🔬 Simulations Tab

### Simulation Setup Workflow

#### Step 1: Choose Simulation Type

**Available Templates:**

| Template | Description | Complexity | Time |
|----------|-------------|------------|------|
| 🌌 **Spacetime Emergence** | Basic emergence from entanglement | Beginner | 5 min |
| 🕳️ **Black Hole Formation** | Schwarzschild black hole | Intermediate | 15 min |
| 🌊 **Gravitational Waves** | Binary merger simulation | Advanced | 2 hours |
| 🌍 **Cosmological Evolution** | Large-scale structure | Expert | 8 hours |

#### Step 2: Configure Parameters

**Basic Parameters Panel:**

```
Grid Configuration:
┌─────────────────────────────────────┐
│ Grid Size: [64] x [64] x [64]       │
│ Spatial Extent: [10.0] (Planck)    │
│ Boundary: [Periodic ▼]             │
└─────────────────────────────────────┘

Time Evolution:
┌─────────────────────────────────────┐
│ Time Step: [0.01] (Planck)         │
│ Total Time: [10.0] (Planck)        │
│ Adaptive: [☑️] Enable              │
└─────────────────────────────────────┘

Physics Parameters:
┌─────────────────────────────────────┐
│ Coupling Strength: [1.0]           │
│ Decoherence Rate: [0.01]           │
│ Quantum Field Mass: [0.1]          │
└─────────────────────────────────────┘
```

**Advanced Parameters (Expandable):**

- Numerical solver settings
- Boundary condition details
- Output format options
- Performance optimization

#### Step 3: Initial Conditions

**Visual Initial Condition Editor:**

```
Initial State Configuration:
┌─────────────────────────────────────┐
│  Entanglement Distribution:         │
│  ○ Uniform                          │
│  ● Gaussian Blob                    │
│  ○ Multiple Sources                 │
│  ○ Custom Function                  │
│                                     │
│  Center: X:[5.0] Y:[5.0] Z:[5.0]   │
│  Width:  [2.0]                     │
│  Amplitude: [1.0]                  │
│                                     │
│  [3D Preview] [Load from File]     │
└─────────────────────────────────────┘

Spacetime Metric:
┌─────────────────────────────────────┐
│  ● Flat (Minkowski)                │
│  ○ Schwarzschild                   │
│  ○ Kerr (Rotating)                 │
│  ○ Custom                          │
│                                     │
│  [Metric Preview]                  │
└─────────────────────────────────────┘
```

#### Step 4: Output Configuration

```
Output Settings:
┌─────────────────────────────────────┐
│ Output Directory: [results/sim_001] │
│ Save Interval: [10] steps          │
│ Data Format: [HDF5 ▼]              │
│ Compression: [☑️] Enable           │
│                                     │
│ Save Components:                    │
│ ☑️ Entanglement Field              │
│ ☑️ Metric Tensor                   │
│ ☑️ Curvature Data                  │
│ ☑️ Energy Density                  │
│ ☑️ Diagnostics                     │
│                                     │
│ Visualization:                      │
│ ☑️ Generate Plots                  │
│ ☑️ Create Animation                │
│ ☑️ 3D Renderings                   │
└─────────────────────────────────────┘
```

### Running Simulations

#### Simulation Control Panel

```
Simulation Control:
┌─────────────────────────────────────┐
│ Status: [Ready to Start]           │
│                                     │
│ [▶️ Start] [⏸️ Pause] [⏹️ Stop]      │
│                                     │
│ Progress: ████████░░ 80%           │
│ Step: 800/1000                     │
│ Elapsed: 00:02:15                  │
│ Remaining: 00:00:34                │
│                                     │
│ CPU Usage: 85%                     │
│ Memory: 4.2/16 GB                  │
│ GPU: 45% (if available)            │
└─────────────────────────────────────┘
```

#### Real-Time Monitoring

**Live Plots Panel:**

- Entanglement evolution
- Energy conservation
- Constraint violations
- Performance metrics

**Real-Time 3D Visualization:**

- Interactive 3D viewer
- Adjustable opacity and colormaps
- Slice planes and isosurfaces
- Animation controls

### Simulation Queue Management

```
Simulation Queue:
┌─────────────────────────────────────────────────────────────┐
│ # │ Name           │ Status    │ Progress │ ETA     │ Actions │
├───┼────────────────┼───────────┼──────────┼─────────┼─────────┤
│ 1 │ BlackHole_001  │ Running   │ ██████░░ │ 00:15:30│ [⏸️][⏹️] │
│ 2 │ Emergence_002  │ Queued    │ ░░░░░░░░ │ --:--:--│ [▶️][🗑️] │
│ 3 │ Cosmology_001  │ Completed │ ████████ │ Done    │ [📊][📁] │
└─────────────────────────────────────────────────────────────┘

[➕ Add Simulation] [📊 Batch Analysis] [⚙️ Queue Settings]
```

## 📈 Analysis Tab

### Results Browser

```
Results Browser:
┌─────────────────────────────────────────────────────────────┐
│ 📁 results/                                                 │
│   ├─ 📁 blackhole_20250608_074802/                         │
│   │   ├─ 📊 simulation_data.h5                             │
│   │   ├─ 📈 energy_evolution.png                           │
│   │   ├─ 🎬 animation.mp4                                  │
│   │   └─ 📝 simulation_log.txt                             │
│   ├─ 📁 emergence_20250608_075344/                         │
│   └─ 📁 cosmology_20250608_080903/                         │
│                                                             │
│ Selected: blackhole_20250608_074802/                       │
│ [📊 Load Data] [📈 Quick Plot] [🔍 Detailed Analysis]      │
└─────────────────────────────────────────────────────────────┘
```

### Visualization Tools

#### 2D Plotting Interface

```
2D Plot Configuration:
┌─────────────────────────────────────┐
│ Data Source: [Entanglement Field ▼] │
│ Plot Type: [Heatmap ▼]             │
│ Time Slice: ████████░░ 800/1000    │
│ Spatial Slice: Z = [16]            │
│                                     │
│ Colormap: [viridis ▼]              │
│ Scale: [Linear ▼]                  │
│ Range: Auto [☑️] Min:[0] Max:[1]   │
│                                     │
│ [Update Plot] [Save Image]         │
└─────────────────────────────────────┘
```

#### 3D Visualization Interface

```
3D Visualization:
┌─────────────────────────────────────┐
│  🎮 Interactive 3D Viewer           │
│  ┌─────────────────────────────┐    │
│  │                             │    │
│  │     [3D Spacetime View]     │    │
│  │                             │    │
│  └─────────────────────────────┘    │
│                                     │
│  Controls:                          │
│  🔄 Rotate: [Mouse Drag]           │
│  🔍 Zoom: [Mouse Wheel]            │
│  📷 Camera: [Reset] [Top] [Side]   │
│                                     │
│  Rendering:                         │
│  Opacity: ████████░░ 80%           │
│  Quality: [High ▼]                 │
│  Lighting: [☑️] Enable             │
│                                     │
│  [Screenshot] [Record Video]        │
└─────────────────────────────────────┘
```

### Data Analysis Tools

#### Statistical Analysis Panel

```
Statistical Analysis:
┌─────────────────────────────────────┐
│ Dataset: [Current Simulation ▼]    │
│                                     │
│ Basic Statistics:                   │
│ • Mean Entanglement: 0.234         │
│ • Standard Deviation: 0.089        │
│ • Min/Max: 0.001 / 0.845           │
│ • Total Energy: -1.234e-03         │
│                                     │
│ Correlation Analysis:               │
│ • Entanglement-Curvature: 0.89     │
│ • Energy Conservation: 99.7%       │
│                                     │
│ Time Series Analysis:               │
│ • Trend: [Decreasing ▼]           │
│ • Periodicity: None detected       │
│ • Stability: Good                   │
│                                     │
│ [Export Statistics] [Generate Report]│
└─────────────────────────────────────┘
```

#### Comparison Tools

```
Simulation Comparison:
┌─────────────────────────────────────────────────────────────┐
│ Simulation A: [blackhole_001 ▼]                            │
│ Simulation B: [blackhole_002 ▼]                            │
│                                                             │
│ Comparison Metrics:                                         │
│ ┌─────────────────┬─────────────────┬─────────────────┐    │
│ │ Metric          │ Simulation A    │ Simulation B    │    │
│ ├─────────────────┼─────────────────┼─────────────────┤    │
│ │ Final Energy    │ -1.234e-03     │ -1.235e-03     │    │
│ │ Max Curvature   │ 2.45e-02       │ 2.47e-02       │    │
│ │ Runtime         │ 00:15:30       │ 00:16:45       │    │
│ │ Memory Usage    │ 4.2 GB         │ 4.1 GB         │    │
│ └─────────────────┴─────────────────┴─────────────────┘    │
│                                                             │
│ [Side-by-Side Plot] [Difference Plot] [Export Comparison]  │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Research Tab

### Literature Integration

```
Research Assistant:
┌─────────────────────────────────────┐
│ 📚 Literature Database              │
│ ┌─────────────────────────────────┐ │
│ │ Search: [quantum gravity]       │ │
│ │ [🔍 Search] [Advanced ▼]       │ │
│ └─────────────────────────────────┘ │
│                                     │
│ Recent Papers:                      │
│ • "Entanglement and Spacetime"     │
│ • "Quantum Gravity Experiments"    │
│ • "AdS/CFT Correspondence"         │
│                                     │
│ Recommendations:                    │
│ Based on your current research...   │
│ • Similar studies                   │
│ • Relevant methodologies           │
│ • Citation suggestions             │
│                                     │
│ [Bibliography Manager]             │
└─────────────────────────────────────┘
```

### Experimental Predictions

```
Prediction Generator:
┌─────────────────────────────────────┐
│ Experiment Type: [Tabletop ▼]      │
│ Observable: [Force Deviation ▼]    │
│                                     │
│ Parameters:                         │
│ • Test Mass: [1e-3] kg             │
│ • Source Mass: [100] kg            │
│ • Separation: [0.1] m              │
│                                     │
│ EG-QGEM Prediction:                │
│ • Signal: 1.2e-15 N                │
│ • Uncertainty: ±2.3e-16 N          │
│ • Detection Confidence: High       │
│                                     │
│ [Generate Full Report]             │
│ [Export to Experimentalists]       │
└─────────────────────────────────────┘
```

### Collaboration Tools

```
Collaboration Hub:
┌─────────────────────────────────────┐
│ 👥 Team Members (3 online)         │
│ • Dr. Alice Smith (Theory)         │
│ • Prof. Bob Johnson (Experiments)  │
│ • Dr. Carol Wilson (Computing)     │
│                                     │
│ 💬 Recent Activity:                │
│ • Alice shared new parameters      │
│ • Bob uploaded experimental data   │
│ • Carol optimized simulation code  │
│                                     │
│ 📊 Shared Resources:               │
│ • Parameter sets                   │
│ • Simulation templates             │
│ • Analysis scripts                 │
│ • Research notes                   │
│                                     │
│ [Start Video Conference]           │
│ [Share Current Results]            │
└─────────────────────────────────────┘
```

## ⚙️ Advanced GUI Features

### Custom Scripting Interface

```
Script Editor:
┌─────────────────────────────────────────────────────────────┐
│ File: custom_analysis.py                            [Save]  │
├─────────────────────────────────────────────────────────────┤
│  1  # Custom analysis script                               │
│  2  import numpy as np                                     │
│  3  from gui_interface import get_current_simulation       │
│  4                                                         │
│  5  def analyze_entanglement_patterns():                   │
│  6      sim_data = get_current_simulation()                │
│  7      # Your custom analysis here                       │
│  8      return results                                     │
│  9                                                         │
│ 10  # Execute automatically when simulation completes     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ [▶️ Run Script] [🐛 Debug] [📚 Help] [💾 Save Template]    │
└─────────────────────────────────────────────────────────────┘
```

### Plugin System

```
Plugin Manager:
┌─────────────────────────────────────────────────────────────┐
│ Installed Plugins:                                          │
│ ☑️ Advanced Visualization Suite v1.2                      │
│ ☑️ Machine Learning Integration v0.8                      │
│ ☑️ Experimental Data Import v1.0                          │
│ ☐ Quantum Circuit Simulator v0.5                          │
│                                                             │
│ Available Plugins:                                          │
│ • Statistical Analysis Extension                           │
│ • Collaboration Tools Pro                                  │
│ • High-Performance Computing Interface                     │
│                                                             │
│ [Browse Plugin Store] [Install from File] [Create Plugin] │
└─────────────────────────────────────────────────────────────┘
```

### Automation and Workflows

```
Workflow Designer:
┌─────────────────────────────────────────────────────────────┐
│ Workflow: "Parameter Study Automation"                     │
│                                                             │
│ ┌─────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐         │
│ │Start│→│Set Params│→│Run Sim  │→│Analyze  │→ ...      │
│ └─────┘  └──────────┘  └─────────┘  └──────────┘         │
│                                                             │
│ Trigger: [Manual ▼] [Scheduled ▼] [Event-based ▼]        │
│ Notification: [Email ▼] [Slack ▼] [None ▼]               │
│                                                             │
│ [Save Workflow] [Run Now] [Schedule] [Export]             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration and Settings

### General Settings

```
GUI Configuration:
┌─────────────────────────────────────┐
│ Appearance:                         │
│ • Theme: [Dark ▼]                  │
│ • Font Size: [Medium ▼]            │
│ • Language: [English ▼]            │
│                                     │
│ Performance:                        │
│ • Auto-save: [Every 5 min ▼]      │
│ • Memory limit: [8 GB]             │
│ • CPU cores: [Auto ▼]             │
│                                     │
│ Notifications:                      │
│ ☑️ Simulation completion           │
│ ☑️ Error alerts                    │
│ ☐ Progress updates                 │
│                                     │
│ [Apply] [Reset to Defaults]        │
└─────────────────────────────────────┘
```

### Advanced Settings

```
Advanced Configuration:
┌─────────────────────────────────────┐
│ Numerical Settings:                 │
│ • Default tolerance: [1e-8]        │
│ • Max iterations: [1000]           │
│ • Solver method: [Auto ▼]         │
│                                     │
│ File Management:                    │
│ • Auto-cleanup: [☑️] 30 days      │
│ • Compression: [☑️] Enable        │
│ • Backup location: [Browse...]     │
│                                     │
│ External Tools:                     │
│ • Python path: [Auto-detect]       │
│ • CUDA toolkit: [/usr/local/cuda]  │
│ • MPI installation: [Detected]     │
│                                     │
│ [Test Configuration] [Import/Export]│
└─────────────────────────────────────┘
```

## 🐛 Troubleshooting GUI Issues

### Common Problems and Solutions

#### Problem 1: GUI Won't Start

**Symptoms:**

```
Error: Failed to launch GUI
ModuleNotFoundError: No module named 'tkinter'
```

**Solution:**

```bash
# Install tkinter (Ubuntu/Debian)
sudo apt-get install python3-tkinter

# Install tkinter (macOS with Homebrew)
brew install python-tk

# Verify installation
python -c "import tkinter; print('GUI support available')"
```

#### Problem 2: Slow Performance

**Symptoms:**

- GUI feels sluggish
- Plots take long time to update
- High CPU usage

**Solutions:**

1. **Reduce Plot Update Frequency:**

   ```
   Settings → Performance → Plot Update: Every 10 steps
   ```

2. **Lower 3D Visualization Quality:**

   ```
   Analysis → 3D Viewer → Quality: Medium
   ```

3. **Disable Real-time Monitoring:**

   ```
   Simulations → Real-time Plots: Disable
   ```

#### Problem 3: Memory Issues

**Symptoms:**

```
MemoryError: Unable to allocate array for plot
```

**Solutions:**

1. **Increase Memory Limit:**

   ```
   Settings → Performance → Memory Limit: 16 GB
   ```

2. **Enable Data Streaming:**

   ```
   Settings → Advanced → Streaming Mode: Enable
   ```

3. **Use Data Downsampling:**

   ```
   Analysis → Data Options → Downsample: 2x
   ```

### Getting Help

**Built-in Help System:**

- Press `F1` for context-sensitive help
- Click `?` button for feature explanations
- Access tutorial videos from Help menu

**Online Resources:**

- GUI documentation: [gui_interface.md](../api/gui_interface.md)
- Video tutorials: Available in Help → Tutorials
- Community forum: Discussion and troubleshooting

**Support Options:**

- Bug reports: Use Help → Report Bug
- Feature requests: Help → Request Feature
- Direct support: <help@egqgem.org>

## 🎯 Best Practices

### Efficient Workflow

1. **Start Small**: Begin with small grid sizes for testing
2. **Use Templates**: Leverage pre-configured simulation templates
3. **Monitor Resources**: Keep an eye on CPU and memory usage
4. **Save Frequently**: Enable auto-save for long simulations
5. **Organize Results**: Use descriptive names and folder structure

### Performance Optimization

1. **Hardware Utilization**:
   - Enable GPU acceleration if available
   - Use all CPU cores for parallel processing
   - Ensure sufficient RAM for your grid size

2. **Simulation Settings**:
   - Start with coarse grids, refine as needed
   - Use adaptive time stepping
   - Enable compression for large datasets

3. **Visualization Efficiency**:
   - Reduce plot update frequency for long simulations
   - Use downsampling for large datasets
   - Cache visualization data when possible

The GUI interface makes EG-QGEM accessible to researchers without extensive programming experience while providing powerful tools for advanced users. Take advantage of the tutorials and help system to master these capabilities!
