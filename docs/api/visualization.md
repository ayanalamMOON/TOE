# Visualization API Reference

This document provides comprehensive API reference for the EG-QGEM visualization and plotting framework.

## Overview

The visualization module provides tools for plotting and analyzing EG-QGEM simulation results, including field visualizations, entanglement maps, and diagnostic plots.

## Core Visualization Classes

### FieldVisualizer

Main class for visualizing gravitational and entanglement fields.

```python
class FieldVisualizer:
    """
    Primary visualization class for EG-QGEM fields.
    """

    def __init__(self, simulation_data: SimulationData,
                 config: VisualizationConfig = None):
        """
        Initialize field visualizer.

        Parameters:
        -----------
        simulation_data : SimulationData
            Simulation results to visualize
        config : VisualizationConfig, optional
            Visualization configuration settings
        """
```

#### Methods

##### plot_metric_field()

```python
def plot_metric_field(self, component: str = 'g00',
                     time_slice: float = None,
                     slice_type: str = 'xy',
                     colormap: str = 'viridis',
                     save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot spacetime metric field components.

    Parameters:
    -----------
    component : str
        Metric component to plot ('g00', 'g11', 'g22', 'g33', 'g01', etc.)
    time_slice : float, optional
        Time value for slice (defaults to final time)
    slice_type : str
        Slice orientation ('xy', 'xz', 'yz', 'radial')
    colormap : str
        Matplotlib colormap name
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_entanglement_field()

```python
def plot_entanglement_field(self, component: str = 'E00',
                           time_slice: float = None,
                           slice_type: str = 'xy',
                           log_scale: bool = False,
                           save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot entanglement tensor field components.

    Parameters:
    -----------
    component : str
        Entanglement tensor component ('E00', 'E11', etc.)
    time_slice : float, optional
        Time value for slice
    slice_type : str
        Slice orientation
    log_scale : bool
        Use logarithmic color scale
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_field_evolution()

```python
def plot_field_evolution(self, field_type: str,
                        point: Tuple[float, float, float],
                        components: List[str] = None,
                        save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot time evolution of field at specific spatial point.

    Parameters:
    -----------
    field_type : str
        Field type ('metric', 'entanglement', 'curvature')
    point : tuple
        Spatial coordinates (x, y, z)
    components : list, optional
        Field components to plot
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### create_animation()

```python
def create_animation(self, field_type: str,
                    component: str,
                    slice_type: str = 'xy',
                    fps: int = 10,
                    save_path: str = None) -> matplotlib.animation.Animation:
    """
    Create animated visualization of field evolution.

    Parameters:
    -----------
    field_type : str
        Field type to animate
    component : str
        Field component
    slice_type : str
        Slice orientation
    fps : int
        Frames per second
    save_path : str, optional
        Path to save animation

    Returns:
    --------
    matplotlib.animation.Animation
        Animation object
    """
```

### EntanglementVisualizer

Specialized class for entanglement-specific visualizations.

```python
class EntanglementVisualizer:
    """
    Specialized visualizer for quantum entanglement phenomena.
    """

    def __init__(self, entanglement_data: EntanglementData):
        """
        Initialize entanglement visualizer.

        Parameters:
        -----------
        entanglement_data : EntanglementData
            Entanglement analysis results
        """
```

#### Methods

##### plot_entanglement_network()

```python
def plot_entanglement_network(self, threshold: float = 0.1,
                             layout: str = 'spring',
                             node_size_scale: float = 1.0,
                             save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot entanglement network between spatial regions.

    Parameters:
    -----------
    threshold : float
        Minimum entanglement strength to display
    layout : str
        Network layout algorithm ('spring', 'circular', 'random')
    node_size_scale : float
        Scaling factor for node sizes
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_entanglement_spectrum()

```python
def plot_entanglement_spectrum(self, region: Tuple[float, ...] = None,
                              time_range: Tuple[float, float] = None,
                              save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot entanglement entropy spectrum.

    Parameters:
    -----------
    region : tuple, optional
        Spatial region coordinates
    time_range : tuple, optional
        Time range (start, end)
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_mutual_information()

```python
def plot_mutual_information(self, region_pairs: List[Tuple],
                           time_slice: float = None,
                           save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot mutual information between spatial regions.

    Parameters:
    -----------
    region_pairs : list
        List of region coordinate pairs
    time_slice : float, optional
        Time value for analysis
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

### DiagnosticPlotter

Class for creating diagnostic and analysis plots.

```python
class DiagnosticPlotter:
    """
    Create diagnostic plots for simulation analysis.
    """

    def __init__(self, simulation: Simulation):
        """
        Initialize diagnostic plotter.

        Parameters:
        -----------
        simulation : Simulation
            Simulation object with results
        """
```

#### Methods

##### plot_energy_conservation()

```python
def plot_energy_conservation(self, save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot energy conservation diagnostics.

    Parameters:
    -----------
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_constraint_violations()

```python
def plot_constraint_violations(self, constraint_types: List[str] = None,
                              save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot constraint violation diagnostics.

    Parameters:
    -----------
    constraint_types : list, optional
        Types of constraints to plot
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

##### plot_convergence_analysis()

```python
def plot_convergence_analysis(self, resolution_series: List[int],
                             save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot numerical convergence analysis.

    Parameters:
    -----------
    resolution_series : list
        List of grid resolutions used
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
```

## 3D Visualization Classes

### Volume3DVisualizer

Class for 3D volumetric visualizations.

```python
class Volume3DVisualizer:
    """
    3D volumetric visualization of fields.
    """

    def __init__(self, field_data: np.ndarray,
                 coordinates: Tuple[np.ndarray, ...]):
        """
        Initialize 3D volume visualizer.

        Parameters:
        -----------
        field_data : np.ndarray
            3D field data array
        coordinates : tuple
            Coordinate arrays (x, y, z)
        """
```

#### Methods

##### create_isosurface()

```python
def create_isosurface(self, isovalue: float,
                     opacity: float = 0.7,
                     color: str = 'blue',
                     save_path: str = None) -> Any:
    """
    Create isosurface visualization.

    Parameters:
    -----------
    isovalue : float
        Isosurface value
    opacity : float
        Surface opacity (0-1)
    color : str
        Surface color
    save_path : str, optional
        Path to save visualization

    Returns:
    --------
    plotly or mayavi object
        3D visualization object
    """
```

##### create_volume_render()

```python
def create_volume_render(self, transfer_function: Dict = None,
                        camera_position: Tuple[float, float, float] = None,
                        save_path: str = None) -> Any:
    """
    Create volume rendering visualization.

    Parameters:
    -----------
    transfer_function : dict, optional
        Color and opacity transfer function
    camera_position : tuple, optional
        Camera position (x, y, z)
    save_path : str, optional
        Path to save visualization

    Returns:
    --------
    plotly or mayavi object
        3D visualization object
    """
```

## Interactive Visualization

### InteractivePlotter

Class for creating interactive visualizations.

```python
class InteractivePlotter:
    """
    Create interactive web-based visualizations.
    """

    def __init__(self, simulation_data: SimulationData):
        """
        Initialize interactive plotter.

        Parameters:
        -----------
        simulation_data : SimulationData
            Simulation results to visualize
        """
```

#### Methods

##### create_dashboard()

```python
def create_dashboard(self, port: int = 8050,
                    debug: bool = False) -> None:
    """
    Create interactive dashboard for data exploration.

    Parameters:
    -----------
    port : int
        Port for web server
    debug : bool
        Enable debug mode
    """
```

##### create_field_explorer()

```python
def create_field_explorer(self, field_types: List[str],
                         save_html: str = None) -> Any:
    """
    Create interactive field exploration tool.

    Parameters:
    -----------
    field_types : list
        Available field types
    save_html : str, optional
        Path to save HTML file

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure object
    """
```

## Utility Functions

### Color and Style Utilities

```python
def create_custom_colormap(colors: List[str],
                          n_colors: int = 256) -> matplotlib.colors.Colormap:
    """
    Create custom colormap from color list.

    Parameters:
    -----------
    colors : list
        List of color specifications
    n_colors : int
        Number of colors in final colormap

    Returns:
    --------
    matplotlib.colors.Colormap
        Custom colormap object
    """

def setup_publication_style() -> None:
    """
    Set up matplotlib style for publication-quality figures.
    """

def get_field_units(field_type: str) -> str:
    """
    Get appropriate units for field type.

    Parameters:
    -----------
    field_type : str
        Type of field

    Returns:
    --------
    str
        Unit string
    """
```

### Export Utilities

```python
def export_figure_series(figures: List[matplotlib.figure.Figure],
                        base_path: str,
                        format: str = 'png',
                        dpi: int = 300) -> None:
    """
    Export series of figures to files.

    Parameters:
    -----------
    figures : list
        List of figure objects
    base_path : str
        Base path for output files
    format : str
        Output format ('png', 'pdf', 'svg')
    dpi : int
        Resolution for raster formats
    """

def create_publication_report(simulation: Simulation,
                            output_path: str,
                            include_animations: bool = False) -> None:
    """
    Generate comprehensive publication-ready report.

    Parameters:
    -----------
    simulation : Simulation
        Simulation object with results
    output_path : str
        Output directory path
    include_animations : bool
        Include animated visualizations
    """
```

## Configuration Classes

### VisualizationConfig

```python
class VisualizationConfig:
    """
    Configuration settings for visualizations.
    """

    def __init__(self):
        # Figure settings
        self.figure_size = (10, 8)
        self.dpi = 300
        self.font_size = 12

        # Color settings
        self.default_colormap = 'viridis'
        self.background_color = 'white'
        self.grid_alpha = 0.3

        # 3D settings
        self.volume_resolution = 128
        self.isosurface_smoothing = True

        # Animation settings
        self.animation_fps = 30
        self.animation_bitrate = 1800

        # Export settings
        self.export_format = 'png'
        self.export_dpi = 300
```

## Example Usage

### Basic Field Visualization

```python
from egqgem.visualization import FieldVisualizer
import numpy as np

# Load simulation data
sim_data = SimulationData.load('simulation_results.h5')

# Create visualizer
vis = FieldVisualizer(sim_data)

# Plot metric field
fig1 = vis.plot_metric_field(component='g00', time_slice=1.0)

# Plot entanglement field
fig2 = vis.plot_entanglement_field(component='E00', log_scale=True)

# Create animation
anim = vis.create_animation('entanglement', 'E00', fps=15)
```

### 3D Visualization

```python
from egqgem.visualization import Volume3DVisualizer

# Extract 3D field data
field_3d = sim_data.get_field_slice('entanglement', time=1.0)
coords = sim_data.get_coordinates()

# Create 3D visualizer
vis_3d = Volume3DVisualizer(field_3d, coords)

# Create isosurface
isosurface = vis_3d.create_isosurface(isovalue=0.1, opacity=0.5)

# Create volume rendering
volume = vis_3d.create_volume_render()
```

### Interactive Dashboard

```python
from egqgem.visualization import InteractivePlotter

# Create interactive plotter
interactive = InteractivePlotter(sim_data)

# Launch dashboard
interactive.create_dashboard(port=8050)
```

## Notes

- All visualization functions return matplotlib or plotly objects that can be further customized
- 3D visualizations require either plotly or mayavi backend
- Interactive features require dash and plotly dependencies
- Large datasets may require downsampling for smooth interactive performance
- All figures can be exported in various formats (PNG, PDF, SVG, etc.)
