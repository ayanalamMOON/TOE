# Simulation Framework API Reference

## 🎯 Overview

The EG-QGEM simulation framework provides a comprehensive set of classes and functions for running quantum-gravitational simulations. This document covers the complete API for simulation setup, execution, and analysis.

## 🏗️ Core Simulation Classes

### SpacetimeEmergenceSimulator

Primary class for studying spacetime emergence from entanglement.

```python
class SpacetimeEmergenceSimulator:
    """
    Simulate the emergence of spacetime geometry from quantum entanglement networks.

    This simulator implements the core EG-QGEM dynamics, evolving both the
    entanglement field and spacetime metric according to the modified Einstein
    field equations.
    """

    def __init__(self, grid_size=(64, 64, 64), spatial_extent=10.0,
                 time_step=0.01, boundary_conditions='periodic'):
        """
        Initialize spacetime emergence simulator.

        Parameters
        ----------
        grid_size : tuple of int, default=(64, 64, 64)
            Number of grid points in each spatial dimension
        spatial_extent : float, default=10.0
            Physical size of simulation box in Planck units
        time_step : float, default=0.01
            Time step for evolution in Planck units
        boundary_conditions : str, default='periodic'
            Boundary conditions: 'periodic', 'absorbing', 'reflecting'

        Attributes
        ----------
        grid : SpacetimeGrid
            Computational grid for spatial discretization
        metric : MetricTensor
            Spacetime metric g_μν
        entanglement_field : EntanglementField
            Quantum entanglement density field E_μν
        evolution_engine : FieldEvolutionEngine
            Numerical evolution engine
        """
```

#### Methods

```python
def set_initial_conditions(self, entanglement_field=None,
                         metric_perturbations=None,
                         matter_distribution=None):
    """
    Set initial conditions for the simulation.

    Parameters
    ----------
    entanglement_field : EntanglementField, optional
        Initial entanglement distribution
    metric_perturbations : MetricPerturbations, optional
        Initial spacetime metric deviations from flat space
    matter_distribution : MatterField, optional
        Initial matter/energy distribution

    Examples
    --------
    >>> sim = SpacetimeEmergenceSimulator()
    >>> initial_entanglement = EntanglementField.create_gaussian_distribution(
    ...     center=(5.0, 5.0, 5.0), width=2.0, amplitude=1.0
    ... )
    >>> sim.set_initial_conditions(entanglement_field=initial_entanglement)
    """

def evolve(self, num_steps=1000, save_interval=10,
           output_directory='results/', progress_callback=None):
    """
    Evolve the system through time.

    Parameters
    ----------
    num_steps : int, default=1000
        Number of time steps to evolve
    save_interval : int, default=10
        Save data every N steps
    output_directory : str, default='results/'
        Directory to save simulation results
    progress_callback : callable, optional
        Function called after each step with (step, time, state)

    Returns
    -------
    SimulationResults
        Object containing evolution history and final state

    Examples
    --------
    >>> results = sim.evolve(num_steps=500, save_interval=20)
    >>> print(f"Final energy: {results.final_energy}")
    """

def get_current_state(self):
    """
    Get the current simulation state.

    Returns
    -------
    dict
        Dictionary containing current field values and diagnostics
    """

def add_matter_source(self, matter_field, coupling_strength=1.0):
    """
    Add matter source to the simulation.

    Parameters
    ----------
    matter_field : MatterField
        Matter/energy distribution
    coupling_strength : float, default=1.0
        Coupling strength between matter and entanglement
    """

def enable_adaptive_timestep(self, tolerance=1e-6, max_timestep_factor=2.0):
    """
    Enable adaptive time stepping based on local error estimates.

    Parameters
    ----------
    tolerance : float, default=1e-6
        Error tolerance for time step adaptation
    max_timestep_factor : float, default=2.0
        Maximum factor by which time step can increase
    """
```

### BlackHoleSimulator

Specialized simulator for black hole dynamics.

```python
class BlackHoleSimulator:
    """
    Simulate black hole formation, evolution, and Hawking radiation
    through entanglement dynamics.
    """

    def __init__(self, grid_size=(128, 128, 128), schwarzschild_radius=2.0,
                 initial_mass=1.0, spin=0.0, charge=0.0):
        """
        Initialize black hole simulator.

        Parameters
        ----------
        grid_size : tuple of int
            Spatial resolution (higher resolution needed near horizon)
        schwarzschild_radius : float
            Schwarzschild radius in simulation units
        initial_mass : float
            Initial black hole mass
        spin : float or array-like
            Angular momentum parameter (0 ≤ |a| ≤ M)
        charge : float
            Electric charge (for Reissner-Nordström black holes)
        """

def setup_initial_black_hole(self, formation_mechanism='gravitational_collapse'):
    """
    Set up initial black hole configuration.

    Parameters
    ----------
    formation_mechanism : str
        'gravitational_collapse', 'primordial', 'merger_remnant'
    """

def simulate_hawking_radiation(self, evolution_time=100.0,
                              radiation_extraction_radius=10.0):
    """
    Simulate Hawking radiation emission.

    Parameters
    ----------
    evolution_time : float
        Total evolution time
    radiation_extraction_radius : float
        Radius at which to extract radiation

    Returns
    -------
    HawkingRadiationData
        Radiation spectrum and particle emission data
    """

def calculate_horizon_properties(self):
    """
    Calculate black hole horizon properties.

    Returns
    -------
    dict
        Horizon area, surface gravity, temperature, entropy
    """
```

### CosmologicalSimulator

Large-scale cosmological structure formation.

```python
class CosmologicalSimulator:
    """
    Simulate cosmological structure formation through entanglement dynamics.
    """

    def __init__(self, box_size=100.0, grid_size=(256, 256, 256),
                 initial_redshift=30.0, final_redshift=0.0,
                 cosmology='Planck2018'):
        """
        Initialize cosmological simulator.

        Parameters
        ----------
        box_size : float
            Comoving box size in Mpc/h
        grid_size : tuple of int
            Number of grid points per dimension
        initial_redshift : float
            Starting redshift for simulation
        final_redshift : float
            Final redshift
        cosmology : str or dict
            Cosmological parameters ('Planck2018', 'WMAP9', or custom dict)
        """

def set_initial_conditions(self, power_spectrum='planck2018',
                          perturbation_amplitude=1e-5):
    """
    Set cosmological initial conditions.

    Parameters
    ----------
    power_spectrum : str or callable
        Initial power spectrum of density fluctuations
    perturbation_amplitude : float
        Amplitude of initial density perturbations
    """

def evolve_with_expansion(self, output_redshifts=None,
                         include_nonlinear=True):
    """
    Evolve with cosmic expansion.

    Parameters
    ----------
    output_redshifts : array-like, optional
        Redshifts at which to save output
    include_nonlinear : bool, default=True
        Include nonlinear structure formation

    Returns
    -------
    CosmologicalResults
        Structure formation history and final state
    """
```

## 📊 Data Structures

### SimulationResults

Container for simulation output data.

```python
class SimulationResults:
    """
    Container for simulation results and analysis tools.
    """

    def __init__(self, simulation_type, parameters, evolution_data):
        """
        Initialize results container.

        Attributes
        ----------
        simulation_type : str
            Type of simulation that generated these results
        parameters : dict
            Simulation parameters used
        evolution_data : dict
            Time series data from simulation
        """

    # Core data access
    @property
    def time_array(self):
        """Array of time values for evolution."""

    @property
    def entanglement_evolution(self):
        """Time evolution of entanglement field."""

    @property
    def metric_evolution(self):
        """Time evolution of spacetime metric."""

    # Analysis methods
    def calculate_energy_conservation(self):
        """Calculate energy conservation throughout evolution."""

    def extract_gravitational_waves(self, extraction_radius=50.0):
        """Extract gravitational wave signal."""

    def compute_entanglement_entropy(self, region=None):
        """Compute entanglement entropy for specified region."""

    def analyze_phase_transitions(self):
        """Identify and characterize phase transitions."""

    # Visualization
    def plot_energy_evolution(self, ax=None):
        """Plot energy conservation over time."""

    def plot_entanglement_distribution(self, time_slice=-1, ax=None):
        """Plot spatial entanglement distribution."""

    def create_animation(self, field='entanglement', fps=10):
        """Create animation of field evolution."""

    # Data export
    def save_to_hdf5(self, filename):
        """Save results to HDF5 file."""

    def export_to_numpy(self, filename_base):
        """Export data arrays to NumPy format."""
```

### EntanglementField

Quantum entanglement field representation.

```python
class EntanglementField:
    """
    Quantum entanglement field E_μν with geometric interpretation.
    """

    def __init__(self, grid_shape, spatial_dimensions=3):
        """
        Initialize entanglement field.

        Parameters
        ----------
        grid_shape : tuple
            Shape of computational grid
        spatial_dimensions : int, default=3
            Number of spatial dimensions
        """

    @classmethod
    def create_gaussian_distribution(cls, center, width, amplitude=1.0):
        """
        Create Gaussian entanglement distribution.

        Parameters
        ----------
        center : array-like
            Center of Gaussian distribution
        width : float
            Width parameter (standard deviation)
        amplitude : float, default=1.0
            Peak amplitude

        Returns
        -------
        EntanglementField
            Field with Gaussian distribution
        """

    @classmethod
    def create_from_quantum_state(cls, quantum_state, partition_scheme):
        """
        Create entanglement field from quantum state.

        Parameters
        ----------
        quantum_state : QuantumState
            Quantum state to analyze
        partition_scheme : str
            How to partition system for entanglement calculation

        Returns
        -------
        EntanglementField
            Entanglement field derived from quantum state
        """

    def evolve_step(self, dt, metric_tensor, matter_coupling=0.0):
        """
        Evolve entanglement field by one time step.

        Parameters
        ----------
        dt : float
            Time step
        metric_tensor : MetricTensor
            Current spacetime metric
        matter_coupling : float, default=0.0
            Coupling to matter fields
        """

    def calculate_entanglement_entropy(self, region_mask):
        """
        Calculate entanglement entropy for specified region.

        Parameters
        ----------
        region_mask : ndarray
            Boolean mask defining spatial region

        Returns
        -------
        float
            von Neumann entanglement entropy
        """

    def compute_mutual_information(self, region_a, region_b):
        """
        Compute mutual information between regions.

        Parameters
        ----------
        region_a, region_b : ndarray
            Boolean masks defining regions

        Returns
        -------
        float
            Mutual information I(A:B)
        """
```

### MetricTensor

Spacetime metric representation and operations.

```python
class MetricTensor:
    """
    Spacetime metric tensor g_μν with geometric calculations.
    """

    def __init__(self, grid_shape, metric_signature=(-1, 1, 1, 1)):
        """
        Initialize metric tensor.

        Parameters
        ----------
        grid_shape : tuple
            Shape of spatial grid
        metric_signature : tuple, default=(-1, 1, 1, 1)
            Metric signature (mostly plus convention)
        """

    @classmethod
    def create_minkowski(cls, grid_shape):
        """Create flat Minkowski spacetime metric."""

    @classmethod
    def create_schwarzschild(cls, grid_shape, mass, center=None):
        """
        Create Schwarzschild black hole metric.

        Parameters
        ----------
        grid_shape : tuple
            Spatial grid shape
        mass : float
            Black hole mass
        center : array-like, optional
            Center of black hole
        """

    def calculate_christoffel_symbols(self):
        """
        Calculate Christoffel symbols Γ^μ_νρ.

        Returns
        -------
        ndarray
            Christoffel symbols array
        """

    def calculate_riemann_tensor(self):
        """
        Calculate Riemann curvature tensor R^μ_νρσ.

        Returns
        -------
        ndarray
            Riemann tensor components
        """

    def calculate_ricci_tensor(self):
        """
        Calculate Ricci tensor R_μν.

        Returns
        -------
        ndarray
            Ricci tensor components
        """

    def calculate_ricci_scalar(self):
        """
        Calculate Ricci scalar R.

        Returns
        -------
        ndarray
            Ricci scalar at each grid point
        """

    def calculate_einstein_tensor(self):
        """
        Calculate Einstein tensor G_μν.

        Returns
        -------
        ndarray
            Einstein tensor components
        """

    def geodesic_equation(self, initial_position, initial_velocity):
        """
        Solve geodesic equation for particle motion.

        Parameters
        ----------
        initial_position : array-like
            Initial 4-position
        initial_velocity : array-like
            Initial 4-velocity

        Returns
        -------
        Geodesic
            Geodesic trajectory object
        """
```

## ⚙️ Configuration and Parameters

### SimulationParameters

Configuration management for simulations.

```python
class SimulationParameters:
    """
    Centralized parameter management for simulations.
    """

    def __init__(self, parameter_dict=None):
        """
        Initialize with parameters.

        Parameters
        ----------
        parameter_dict : dict, optional
            Dictionary of parameter values
        """

    @classmethod
    def load_from_file(cls, filename):
        """Load parameters from JSON or YAML file."""

    @classmethod
    def get_default_spacetime_emergence(cls):
        """Get default parameters for spacetime emergence simulation."""

    @classmethod
    def get_default_black_hole(cls):
        """Get default parameters for black hole simulation."""

    @classmethod
    def get_default_cosmological(cls):
        """Get default parameters for cosmological simulation."""

    def validate(self):
        """Validate parameter consistency and physics constraints."""

    def save_to_file(self, filename):
        """Save parameters to file."""

    def get_summary(self):
        """Get human-readable parameter summary."""
```

## 🔧 Utility Functions

### Grid and Discretization

```python
def create_spacetime_grid(spatial_extent, grid_size, dimensions=3):
    """
    Create computational grid for spacetime simulations.

    Parameters
    ----------
    spatial_extent : float or array-like
        Physical size of simulation domain
    grid_size : int or tuple
        Number of grid points
    dimensions : int, default=3
        Number of spatial dimensions

    Returns
    -------
    SpacetimeGrid
        Computational grid object
    """

def setup_finite_difference_operators(grid, order=4):
    """
    Set up finite difference operators for derivatives.

    Parameters
    ----------
    grid : SpacetimeGrid
        Computational grid
    order : int, default=4
        Order of finite difference accuracy

    Returns
    -------
    dict
        Dictionary of finite difference operators
    """
```

### Initial Conditions

```python
def generate_vacuum_fluctuations(grid, field_mass=0.0, temperature=0.0):
    """
    Generate quantum vacuum fluctuations.

    Parameters
    ----------
    grid : SpacetimeGrid
        Computational grid
    field_mass : float, default=0.0
        Field mass parameter
    temperature : float, default=0.0
        Temperature for thermal fluctuations

    Returns
    -------
    QuantumField
        Vacuum fluctuation field
    """

def create_matter_distribution(distribution_type, parameters, grid):
    """
    Create matter/energy distribution.

    Parameters
    ----------
    distribution_type : str
        Type of distribution ('gaussian', 'uniform', 'delta', 'custom')
    parameters : dict
        Distribution parameters
    grid : SpacetimeGrid
        Computational grid

    Returns
    -------
    MatterField
        Matter distribution field
    """
```

### Analysis Functions

```python
def calculate_total_energy(metric, entanglement_field, matter_field=None):
    """
    Calculate total energy of the system.

    Parameters
    ----------
    metric : MetricTensor
        Spacetime metric
    entanglement_field : EntanglementField
        Entanglement field
    matter_field : MatterField, optional
        Matter field contribution

    Returns
    -------
    float
        Total energy
    """

def extract_gravitational_waves(metric_evolution, extraction_radius,
                               observer_angles=(0, 0)):
    """
    Extract gravitational wave signal from metric evolution.

    Parameters
    ----------
    metric_evolution : array-like
        Time series of metric tensor
    extraction_radius : float
        Radius for wave extraction
    observer_angles : tuple, default=(0, 0)
        Observer angles (theta, phi)

    Returns
    -------
    tuple
        (h_plus, h_cross) gravitational wave polarizations
    """

def measure_entanglement_scaling(entanglement_field, region_sizes):
    """
    Measure entanglement scaling with region size.

    Parameters
    ----------
    entanglement_field : EntanglementField
        Entanglement field to analyze
    region_sizes : array-like
        Range of region sizes to test

    Returns
    -------
    dict
        Scaling analysis results
    """
```

## 🚀 Performance and Optimization

### Parallel Execution

```python
def run_simulation_parallel(simulation_function, parameter_sets,
                           num_processes=None):
    """
    Run multiple simulations in parallel.

    Parameters
    ----------
    simulation_function : callable
        Function that runs a single simulation
    parameter_sets : list
        List of parameter dictionaries
    num_processes : int, optional
        Number of parallel processes (default: number of CPU cores)

    Returns
    -------
    list
        Results from all simulations
    """

class GPUAccelerator:
    """
    GPU acceleration for computationally intensive operations.
    """

    def __init__(self, device='auto'):
        """Initialize GPU accelerator."""

    def accelerate_field_evolution(self, evolution_function):
        """Accelerate field evolution using GPU."""

    def accelerate_matrix_operations(self, matrix_function):
        """Accelerate matrix operations using GPU."""
```

### Memory Management

```python
class MemoryManager:
    """
    Manage memory usage for large simulations.
    """

    def __init__(self, max_memory_gb=8):
        """Initialize memory manager."""

    def enable_streaming(self, chunk_size=1000):
        """Enable data streaming for large datasets."""

    def setup_checkpointing(self, checkpoint_interval=100):
        """Set up simulation checkpointing."""

    def optimize_for_system(self):
        """Optimize memory usage for current system."""
```

## 🎯 Usage Examples

### Basic Spacetime Emergence

```python
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator
from theory.entanglement_dynamics import EntanglementField

# Create simulator
sim = SpacetimeEmergenceSimulator(
    grid_size=(64, 64, 64),
    spatial_extent=10.0,
    time_step=0.01
)

# Set initial conditions
initial_entanglement = EntanglementField.create_gaussian_distribution(
    center=(5.0, 5.0, 5.0),
    width=2.0,
    amplitude=1.0
)

sim.set_initial_conditions(entanglement_field=initial_entanglement)

# Run simulation
results = sim.evolve(num_steps=1000, save_interval=50)

# Analyze results
energy_conservation = results.calculate_energy_conservation()
print(f"Energy conservation: {energy_conservation:.6f}")

# Visualize
results.plot_energy_evolution()
results.create_animation(field='entanglement')
```

### Parameter Study

```python
from simulations.parameter_studies import ParameterSweep
import numpy as np

# Define parameter ranges
parameter_ranges = {
    'entanglement_coupling': np.logspace(-2, 1, 10),
    'decoherence_rate': np.logspace(-4, -1, 8),
    'grid_size': [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
}

# Create parameter sweep
sweep = ParameterSweep(
    simulation_function=run_emergence_simulation,
    parameter_ranges=parameter_ranges,
    observables=['final_energy', 'max_curvature', 'entanglement_entropy']
)

# Run parameter sweep
results = sweep.run_parallel(num_cores=8)

# Analyze results
optimal_params = sweep.find_optimal_parameters(
    objective='minimize_energy_variance'
)
```

This API provides comprehensive access to EG-QGEM's simulation capabilities, from basic usage to advanced research applications.
