# Utilities API Reference

This document provides comprehensive API reference for the EG-QGEM utility functions and helper tools.

## Overview

The utilities module contains helper functions, data processing tools, mathematical utilities, and convenience functions used throughout the EG-QGEM framework.

## Mathematical Utilities

### Tensor Operations

```python
import numpy as np
from typing import Tuple, List, Union

def christoffel_symbols(metric: np.ndarray,
                       coordinates: np.ndarray) -> np.ndarray:
    """
    Compute Christoffel symbols from metric tensor.

    Parameters:
    -----------
    metric : np.ndarray
        4D metric tensor g_μν with shape (..., 4, 4)
    coordinates : np.ndarray
        Coordinate grid points

    Returns:
    --------
    np.ndarray
        Christoffel symbols Γ^μ_νρ with shape (..., 4, 4, 4)
    """

def riemann_tensor(christoffel: np.ndarray) -> np.ndarray:
    """
    Compute Riemann curvature tensor from Christoffel symbols.

    Parameters:
    -----------
    christoffel : np.ndarray
        Christoffel symbols Γ^μ_νρ

    Returns:
    --------
    np.ndarray
        Riemann tensor R^μ_νρσ with shape (..., 4, 4, 4, 4)
    """

def ricci_tensor(riemann: np.ndarray) -> np.ndarray:
    """
    Compute Ricci tensor from Riemann tensor.

    Parameters:
    -----------
    riemann : np.ndarray
        Riemann curvature tensor

    Returns:
    --------
    np.ndarray
        Ricci tensor R_μν with shape (..., 4, 4)
    """

def ricci_scalar(ricci: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """
    Compute Ricci scalar from Ricci tensor and metric.

    Parameters:
    -----------
    ricci : np.ndarray
        Ricci tensor R_μν
    metric : np.ndarray
        Metric tensor g_μν

    Returns:
    --------
    np.ndarray
        Ricci scalar R
    """

def einstein_tensor(ricci: np.ndarray,
                   ricci_scalar: np.ndarray,
                   metric: np.ndarray) -> np.ndarray:
    """
    Compute Einstein tensor.

    Parameters:
    -----------
    ricci : np.ndarray
        Ricci tensor R_μν
    ricci_scalar : np.ndarray
        Ricci scalar R
    metric : np.ndarray
        Metric tensor g_μν

    Returns:
    --------
    np.ndarray
        Einstein tensor G_μν
    """
```

### Differential Operations

```python
def covariant_derivative(tensor: np.ndarray,
                        christoffel: np.ndarray,
                        direction: int) -> np.ndarray:
    """
    Compute covariant derivative of tensor field.

    Parameters:
    -----------
    tensor : np.ndarray
        Input tensor field
    christoffel : np.ndarray
        Christoffel symbols
    direction : int
        Direction index for derivative

    Returns:
    --------
    np.ndarray
        Covariant derivative
    """

def lie_derivative(tensor: np.ndarray,
                  vector_field: np.ndarray,
                  coordinates: np.ndarray) -> np.ndarray:
    """
    Compute Lie derivative of tensor along vector field.

    Parameters:
    -----------
    tensor : np.ndarray
        Input tensor field
    vector_field : np.ndarray
        Vector field for Lie derivative
    coordinates : np.ndarray
        Coordinate system

    Returns:
    --------
    np.ndarray
        Lie derivative
    """

def gradient(field: np.ndarray,
            coordinates: np.ndarray,
            metric: np.ndarray = None) -> np.ndarray:
    """
    Compute gradient of scalar field.

    Parameters:
    -----------
    field : np.ndarray
        Scalar field
    coordinates : np.ndarray
        Coordinate grid
    metric : np.ndarray, optional
        Metric tensor (for covariant gradient)

    Returns:
    --------
    np.ndarray
        Gradient vector field
    """

def divergence(vector_field: np.ndarray,
              coordinates: np.ndarray,
              metric: np.ndarray) -> np.ndarray:
    """
    Compute divergence of vector field.

    Parameters:
    -----------
    vector_field : np.ndarray
        Vector field
    coordinates : np.ndarray
        Coordinate grid
    metric : np.ndarray
        Metric tensor

    Returns:
    --------
    np.ndarray
        Divergence scalar field
    """

def laplacian(field: np.ndarray,
             coordinates: np.ndarray,
             metric: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian of scalar field.

    Parameters:
    -----------
    field : np.ndarray
        Scalar field
    coordinates : np.ndarray
        Coordinate grid
    metric : np.ndarray
        Metric tensor

    Returns:
    --------
    np.ndarray
        Laplacian scalar field
    """
```

### Numerical Integration

```python
def integrate_field(field: np.ndarray,
                   coordinates: np.ndarray,
                   region: Tuple[float, ...] = None,
                   method: str = 'simpson') -> float:
    """
    Integrate field over spatial region.

    Parameters:
    -----------
    field : np.ndarray
        Field to integrate
    coordinates : np.ndarray
        Coordinate grid
    region : tuple, optional
        Integration boundaries
    method : str
        Integration method ('simpson', 'trapz', 'gaussian')

    Returns:
    --------
    float
        Integrated value
    """

def surface_integral(field: np.ndarray,
                    surface_coords: np.ndarray,
                    normal_vectors: np.ndarray) -> float:
    """
    Compute surface integral of vector field.

    Parameters:
    -----------
    field : np.ndarray
        Vector field
    surface_coords : np.ndarray
        Surface coordinate grid
    normal_vectors : np.ndarray
        Surface normal vectors

    Returns:
    --------
    float
        Surface integral value
    """

def line_integral(field: np.ndarray,
                 path_coords: np.ndarray,
                 tangent_vectors: np.ndarray) -> float:
    """
    Compute line integral of vector field.

    Parameters:
    -----------
    field : np.ndarray
        Vector field
    path_coords : np.ndarray
        Path coordinate points
    tangent_vectors : np.ndarray
        Path tangent vectors

    Returns:
    --------
    float
        Line integral value
    """
```

## Data Processing Utilities

### Field Analysis

```python
class FieldAnalyzer:
    """
    Utility class for field analysis operations.
    """

    def __init__(self, field_data: np.ndarray,
                 coordinates: np.ndarray):
        """
        Initialize field analyzer.

        Parameters:
        -----------
        field_data : np.ndarray
            Field data to analyze
        coordinates : np.ndarray
            Coordinate grid
        """
        self.field_data = field_data
        self.coordinates = coordinates

    def find_extrema(self, field_type: str = 'maximum') -> List[Tuple]:
        """
        Find extrema (maxima/minima) in field.

        Parameters:
        -----------
        field_type : str
            Type of extrema ('maximum', 'minimum', 'both')

        Returns:
        --------
        list
            List of extrema coordinates
        """

    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute basic field statistics.

        Returns:
        --------
        dict
            Statistics dictionary with mean, std, min, max, etc.
        """

    def detect_features(self, threshold: float = None) -> List[Dict]:
        """
        Detect coherent features in field.

        Parameters:
        -----------
        threshold : float, optional
            Feature detection threshold

        Returns:
        --------
        list
            List of detected features with properties
        """

    def compute_power_spectrum(self, dimensions: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of field.

        Parameters:
        -----------
        dimensions : list, optional
            Dimensions for FFT

        Returns:
        --------
        tuple
            (frequencies, power_spectrum)
        """
```

### Data Filtering

```python
def smooth_field(field: np.ndarray,
                kernel_size: Union[int, Tuple[int, ...]] = 3,
                method: str = 'gaussian') -> np.ndarray:
    """
    Apply smoothing filter to field data.

    Parameters:
    -----------
    field : np.ndarray
        Input field data
    kernel_size : int or tuple
        Size of smoothing kernel
    method : str
        Smoothing method ('gaussian', 'uniform', 'median')

    Returns:
    --------
    np.ndarray
        Smoothed field data
    """

def denoise_field(field: np.ndarray,
                 noise_level: float = 0.1,
                 method: str = 'wavelet') -> np.ndarray:
    """
    Remove noise from field data.

    Parameters:
    -----------
    field : np.ndarray
        Noisy field data
    noise_level : float
        Estimated noise level
    method : str
        Denoising method ('wavelet', 'bilateral', 'tv')

    Returns:
    --------
    np.ndarray
        Denoised field data
    """

def resample_field(field: np.ndarray,
                  old_coords: np.ndarray,
                  new_coords: np.ndarray,
                  method: str = 'cubic') -> np.ndarray:
    """
    Resample field data onto new coordinate grid.

    Parameters:
    -----------
    field : np.ndarray
        Original field data
    old_coords : np.ndarray
        Original coordinate grid
    new_coords : np.ndarray
        Target coordinate grid
    method : str
        Interpolation method ('cubic', 'linear', 'nearest')

    Returns:
    --------
    np.ndarray
        Resampled field data
    """
```

### Data Validation

```python
def validate_field_data(field: np.ndarray,
                       coordinates: np.ndarray,
                       field_type: str) -> Dict[str, Union[bool, str]]:
    """
    Validate field data for consistency and physical constraints.

    Parameters:
    -----------
    field : np.ndarray
        Field data to validate
    coordinates : np.ndarray
        Coordinate grid
    field_type : str
        Type of field ('metric', 'entanglement', 'scalar')

    Returns:
    --------
    dict
        Validation results with status and messages
    """

def check_symmetries(tensor: np.ndarray,
                    symmetry_type: str) -> bool:
    """
    Check tensor symmetries.

    Parameters:
    -----------
    tensor : np.ndarray
        Tensor to check
    symmetry_type : str
        Symmetry type ('symmetric', 'antisymmetric', 'hermitian')

    Returns:
    --------
    bool
        True if tensor has specified symmetry
    """

def verify_conservation_laws(fields: Dict[str, np.ndarray],
                            coordinates: np.ndarray) -> Dict[str, float]:
    """
    Verify conservation law violations.

    Parameters:
    -----------
    fields : dict
        Dictionary of field arrays
    coordinates : np.ndarray
        Coordinate grid

    Returns:
    --------
    dict
        Conservation violation measures
    """
```

## File I/O Utilities

### Data Export/Import

```python
class DataManager:
    """
    Utility class for data file management.
    """

    @staticmethod
    def save_simulation_data(data: 'SimulationData',
                           filepath: str,
                           format: str = 'hdf5') -> None:
        """
        Save simulation data to file.

        Parameters:
        -----------
        data : SimulationData
            Simulation data object
        filepath : str
            Output file path
        format : str
            File format ('hdf5', 'npz', 'pickle')
        """

    @staticmethod
    def load_simulation_data(filepath: str,
                           format: str = 'auto') -> 'SimulationData':
        """
        Load simulation data from file.

        Parameters:
        -----------
        filepath : str
            Input file path
        format : str
            File format ('auto', 'hdf5', 'npz', 'pickle')

        Returns:
        --------
        SimulationData
            Loaded simulation data
        """

    @staticmethod
    def export_field_data(field: np.ndarray,
                         coordinates: np.ndarray,
                         filepath: str,
                         metadata: Dict = None) -> None:
        """
        Export field data with coordinates.

        Parameters:
        -----------
        field : np.ndarray
            Field data array
        coordinates : np.ndarray
            Coordinate grid
        filepath : str
            Output file path
        metadata : dict, optional
            Additional metadata
        """

    @staticmethod
    def import_field_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Import field data with coordinates.

        Parameters:
        -----------
        filepath : str
            Input file path

        Returns:
        --------
        tuple
            (field_data, coordinates, metadata)
        """
```

### Configuration Management

```python
class ConfigManager:
    """
    Configuration file management utilities.
    """

    @staticmethod
    def load_config(filepath: str) -> Dict:
        """
        Load configuration from file.

        Parameters:
        -----------
        filepath : str
            Configuration file path

        Returns:
        --------
        dict
            Configuration parameters
        """

    @staticmethod
    def save_config(config: Dict, filepath: str) -> None:
        """
        Save configuration to file.

        Parameters:
        -----------
        config : dict
            Configuration parameters
        filepath : str
            Output file path
        """

    @staticmethod
    def validate_config(config: Dict,
                       schema: Dict = None) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.

        Parameters:
        -----------
        config : dict
            Configuration to validate
        schema : dict, optional
            Configuration schema

        Returns:
        --------
        tuple
            (is_valid, error_messages)
        """

    @staticmethod
    def merge_configs(base_config: Dict,
                     override_config: Dict) -> Dict:
        """
        Merge configuration dictionaries.

        Parameters:
        -----------
        base_config : dict
            Base configuration
        override_config : dict
            Override parameters

        Returns:
        --------
        dict
            Merged configuration
        """
```

## Coordinate System Utilities

### Coordinate Transformations

```python
def cartesian_to_spherical(x: np.ndarray,
                          y: np.ndarray,
                          z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to spherical coordinates.

    Parameters:
    -----------
    x, y, z : np.ndarray
        Cartesian coordinates

    Returns:
    --------
    tuple
        (r, theta, phi) spherical coordinates
    """

def spherical_to_cartesian(r: np.ndarray,
                          theta: np.ndarray,
                          phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical to Cartesian coordinates.

    Parameters:
    -----------
    r, theta, phi : np.ndarray
        Spherical coordinates

    Returns:
    --------
    tuple
        (x, y, z) Cartesian coordinates
    """

def cartesian_to_cylindrical(x: np.ndarray,
                           y: np.ndarray,
                           z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to cylindrical coordinates.

    Parameters:
    -----------
    x, y, z : np.ndarray
        Cartesian coordinates

    Returns:
    --------
    tuple
        (rho, phi, z) cylindrical coordinates
    """

def transform_tensor_components(tensor: np.ndarray,
                              jacobian: np.ndarray,
                              tensor_type: str) -> np.ndarray:
    """
    Transform tensor components between coordinate systems.

    Parameters:
    -----------
    tensor : np.ndarray
        Tensor components in original coordinates
    jacobian : np.ndarray
        Jacobian matrix of coordinate transformation
    tensor_type : str
        Tensor type ('contravariant', 'covariant', 'mixed')

    Returns:
    --------
    np.ndarray
        Transformed tensor components
    """
```

### Grid Generation

```python
def create_uniform_grid(bounds: Tuple[Tuple[float, float], ...],
                       resolution: Tuple[int, ...]) -> np.ndarray:
    """
    Create uniform coordinate grid.

    Parameters:
    -----------
    bounds : tuple
        Coordinate bounds for each dimension
    resolution : tuple
        Grid resolution for each dimension

    Returns:
    --------
    np.ndarray
        Coordinate grid arrays
    """

def create_adaptive_grid(field: np.ndarray,
                        bounds: Tuple[Tuple[float, float], ...],
                        refinement_criterion: str = 'gradient') -> np.ndarray:
    """
    Create adaptive mesh based on field properties.

    Parameters:
    -----------
    field : np.ndarray
        Field for adaptive refinement
    bounds : tuple
        Coordinate bounds
    refinement_criterion : str
        Refinement criterion ('gradient', 'curvature', 'error')

    Returns:
    --------
    np.ndarray
        Adaptive coordinate grid
    """

def create_spherical_grid(r_range: Tuple[float, float],
                         theta_range: Tuple[float, float] = (0, np.pi),
                         phi_range: Tuple[float, float] = (0, 2*np.pi),
                         resolution: Tuple[int, int, int] = (50, 50, 50)) -> Tuple[np.ndarray, ...]:
    """
    Create spherical coordinate grid.

    Parameters:
    -----------
    r_range : tuple
        Radial coordinate range
    theta_range : tuple
        Polar angle range
    phi_range : tuple
        Azimuthal angle range
    resolution : tuple
        Grid resolution (r, theta, phi)

    Returns:
    --------
    tuple
        (r, theta, phi) coordinate grids
    """
```

## Performance Utilities

### Memory Management

```python
def estimate_memory_usage(grid_shape: Tuple[int, ...],
                         n_fields: int,
                         dtype: type = np.float64) -> float:
    """
    Estimate memory usage for simulation.

    Parameters:
    -----------
    grid_shape : tuple
        Shape of computational grid
    n_fields : int
        Number of field variables
    dtype : type
        Data type for fields

    Returns:
    --------
    float
        Estimated memory usage in GB
    """

def optimize_memory_layout(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """
    Optimize memory layout for cache efficiency.

    Parameters:
    -----------
    arrays : list
        List of arrays to optimize

    Returns:
    --------
    list
        Memory-optimized arrays
    """

def chunk_large_arrays(array: np.ndarray,
                      chunk_size: int) -> List[np.ndarray]:
    """
    Split large array into manageable chunks.

    Parameters:
    -----------
    array : np.ndarray
        Large array to chunk
    chunk_size : int
        Size of each chunk

    Returns:
    --------
    list
        List of array chunks
    """
```

### Parallel Processing Utilities

```python
def parallelize_field_operation(operation: callable,
                               field: np.ndarray,
                               n_processes: int = None,
                               chunk_overlap: int = 0) -> np.ndarray:
    """
    Parallelize operation on field data.

    Parameters:
    -----------
    operation : callable
        Operation to apply to field
    field : np.ndarray
        Input field data
    n_processes : int, optional
        Number of parallel processes
    chunk_overlap : int
        Overlap between chunks for boundary conditions

    Returns:
    --------
    np.ndarray
        Result of parallel operation
    """

def distribute_computation(task_list: List,
                          worker_function: callable,
                          n_workers: int = None) -> List:
    """
    Distribute computation tasks across workers.

    Parameters:
    -----------
    task_list : list
        List of tasks to distribute
    worker_function : callable
        Function to execute tasks
    n_workers : int, optional
        Number of worker processes

    Returns:
    --------
    list
        Results from all workers
    """
```

## Error Handling and Logging

### Custom Exceptions

```python
class EGQGEMError(Exception):
    """Base exception for EG-QGEM framework."""
    pass

class NumericalError(EGQGEMError):
    """Exception for numerical computation errors."""
    pass

class ConvergenceError(EGQGEMError):
    """Exception for convergence failures."""
    pass

class ValidationError(EGQGEMError):
    """Exception for data validation failures."""
    pass

class ConfigurationError(EGQGEMError):
    """Exception for configuration errors."""
    pass
```

### Logging Utilities

```python
import logging
from typing import Optional

def setup_logger(name: str,
                level: str = 'INFO',
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with appropriate formatting.

    Parameters:
    -----------
    name : str
        Logger name
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Log file path

    Returns:
    --------
    logging.Logger
        Configured logger object
    """

def log_simulation_progress(logger: logging.Logger,
                          current_step: int,
                          total_steps: int,
                          start_time: float) -> None:
    """
    Log simulation progress information.

    Parameters:
    -----------
    logger : logging.Logger
        Logger object
    current_step : int
        Current simulation step
    total_steps : int
        Total number of steps
    start_time : float
        Simulation start time
    """

def log_performance_metrics(logger: logging.Logger,
                          metrics: Dict[str, float]) -> None:
    """
    Log performance metrics.

    Parameters:
    -----------
    logger : logging.Logger
        Logger object
    metrics : dict
        Performance metrics dictionary
    """
```

## Example Usage

### Basic Mathematical Operations

```python
from egqgem.utilities import christoffel_symbols, ricci_tensor
import numpy as np

# Compute curvature from metric
metric = np.random.random((10, 10, 10, 4, 4))
coords = np.linspace(-1, 1, 10)

# Calculate Christoffel symbols
christoffel = christoffel_symbols(metric, coords)

# Calculate Ricci tensor
ricci = ricci_tensor(christoffel_to_riemann(christoffel))
```

### Field Analysis

```python
from egqgem.utilities import FieldAnalyzer

# Analyze field data
field_data = np.random.random((100, 100, 100))
coords = np.meshgrid(*[np.linspace(-5, 5, 100) for _ in range(3)])

analyzer = FieldAnalyzer(field_data, coords)

# Find extrema
maxima = analyzer.find_extrema('maximum')

# Compute statistics
stats = analyzer.compute_statistics()

# Detect features
features = analyzer.detect_features(threshold=0.5)
```

### Data Management

```python
from egqgem.utilities import DataManager

# Save simulation data
DataManager.save_simulation_data(sim_data, 'results.h5')

# Load simulation data
loaded_data = DataManager.load_simulation_data('results.h5')

# Export field for external analysis
DataManager.export_field_data(field, coords, 'field_export.npz')
```

## Notes

- All utility functions include comprehensive error checking and validation
- Mathematical operations use optimized NumPy routines where possible
- Parallel processing utilities automatically detect available CPU cores
- Memory management functions help prevent out-of-memory errors
- Configuration management supports JSON, YAML, and Python formats
- Logging utilities provide structured output for debugging and monitoring
