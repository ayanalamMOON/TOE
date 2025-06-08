# Core Modules API Reference

## 📚 Overview

This document provides comprehensive API documentation for all core EG-QGEM modules. Each module is documented with class definitions, method signatures, parameters, and usage examples.

## 🔬 Theory Modules

### constants.py

#### PhysicalConstants Class

```python
class PhysicalConstants:
    """Physical constants and unit conversions for EG-QGEM calculations."""

    # Fundamental Constants
    c = 2.99792458e8           # Speed of light (m/s)
    G = 6.67430e-11            # Gravitational constant (m³/kg/s²)
    hbar = 1.054571817e-34     # Reduced Planck constant (J·s)
    k_B = 1.380649e-23         # Boltzmann constant (J/K)

    # Derived Constants
    l_planck = 1.616255e-35    # Planck length (m)
    t_planck = 5.391247e-44    # Planck time (s)
    m_planck = 2.176434e-8     # Planck mass (kg)

    @classmethod
    def schwarzschild_radius(cls, mass):
        """Calculate Schwarzschild radius for given mass."""
        return 2 * cls.G * mass / cls.c**2

    @classmethod
    def planck_units(cls, quantity, value):
        """Convert to Planck units."""
        # Implementation details...
```

**Usage Example**:

```python
from theory.constants import PhysicalConstants as PC

# Calculate black hole radius
mass_solar = 1.989e30  # kg
r_s = PC.schwarzschild_radius(mass_solar)
print(f"Solar mass black hole radius: {r_s:.2e} m")

# Convert to Planck units
length_planck = PC.planck_units('length', 1.0)  # 1 meter in Planck units
```

### entanglement_tensor.py

#### EntanglementTensor Class

```python
class EntanglementTensor:
    """Geometric representation of quantum entanglement in spacetime."""

    def __init__(self, metric_signature=(-1, 1, 1, 1)):
        """
        Initialize entanglement tensor calculator.

        Parameters
        ----------
        metric_signature : tuple
            Spacetime metric signature (default: (-1,1,1,1))
        """
        self.signature = metric_signature
        self.dimension = len(metric_signature)

    def compute_entanglement_density(self, state, region_A, region_B):
        """
        Compute entanglement density between two spacetime regions.

        Parameters
        ----------
        state : QuantumState
            Global quantum state
        region_A : SpacetimeRegion
            First spacetime region
        region_B : SpacetimeRegion
            Second spacetime region

        Returns
        -------
        np.ndarray
            Entanglement density tensor E_μν
        """
        # Implementation details...

    def entanglement_entropy(self, reduced_state):
        """
        Calculate von Neumann entanglement entropy.

        Parameters
        ----------
        reduced_state : np.ndarray
            Reduced density matrix

        Returns
        -------
        float
            Entanglement entropy S = -Tr(ρ log ρ)
        """
        eigenvals = np.linalg.eigvals(reduced_state)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        return -np.sum(eigenvals * np.log(eigenvals))

    def mutual_information(self, state, region_A, region_B):
        """Calculate mutual information I(A:B) = S(A) + S(B) - S(AB)."""
        # Implementation details...
```

**Usage Example**:

```python
from theory.entanglement_tensor import EntanglementTensor
import numpy as np

# Create entanglement calculator
ent_calc = EntanglementTensor()

# Calculate entanglement between regions
state = create_quantum_state(n_qubits=10)
region_A = SpacetimeRegion(coordinates=[0, 1, 2])
region_B = SpacetimeRegion(coordinates=[3, 4, 5])

E_tensor = ent_calc.compute_entanglement_density(state, region_A, region_B)
print(f"Entanglement tensor shape: {E_tensor.shape}")
```

### modified_einstein.py

#### ModifiedEinstein Class

```python
class ModifiedEinstein:
    """Solver for Einstein field equations with entanglement contributions."""

    def __init__(self, coupling_constant=1.0):
        """
        Initialize field equation solver.

        Parameters
        ----------
        coupling_constant : float
            Coupling strength between entanglement and geometry
        """
        self.kappa = coupling_constant
        self.constants = PhysicalConstants()

    def field_equations(self, metric, stress_energy, entanglement_tensor):
        """
        Solve modified Einstein field equations: G_μν = κ(T_μν + E_μν)

        Parameters
        ----------
        metric : np.ndarray
            Spacetime metric tensor g_μν
        stress_energy : np.ndarray
            Stress-energy tensor T_μν
        entanglement_tensor : np.ndarray
            Entanglement density tensor E_μν

        Returns
        -------
        np.ndarray
            Einstein tensor G_μν
        """
        # Calculate Christoffel symbols
        christoffel = self.compute_christoffel_symbols(metric)

        # Calculate Riemann tensor
        riemann = self.compute_riemann_tensor(metric, christoffel)

        # Calculate Ricci tensor and scalar
        ricci_tensor = self.compute_ricci_tensor(riemann)
        ricci_scalar = self.compute_ricci_scalar(metric, ricci_tensor)

        # Einstein tensor
        einstein_tensor = ricci_tensor - 0.5 * metric * ricci_scalar

        return einstein_tensor

    def solve_for_metric(self, stress_energy, entanglement_tensor,
                        initial_metric=None, tolerance=1e-10):
        """
        Iteratively solve for metric given matter and entanglement sources.

        Parameters
        ----------
        stress_energy : np.ndarray
            Matter stress-energy tensor
        entanglement_tensor : np.ndarray
            Entanglement source tensor
        initial_metric : np.ndarray, optional
            Starting guess for metric (default: Minkowski)
        tolerance : float
            Convergence criterion

        Returns
        -------
        np.ndarray
            Solution metric tensor
        """
        # Implementation of iterative solution...
```

**Usage Example**:

```python
from theory.modified_einstein import ModifiedEinstein
import numpy as np

# Create field equation solver
solver = ModifiedEinstein(coupling_constant=1.0)

# Define source terms
T_matter = create_matter_tensor()        # Stress-energy from matter
E_entanglement = create_entanglement_tensor()  # Entanglement contribution

# Solve for spacetime metric
metric_solution = solver.solve_for_metric(T_matter, E_entanglement)
print(f"Metric determinant: {np.linalg.det(metric_solution)}")
```

## 🌌 Simulation Modules

### spacetime_emergence.py

#### SpacetimeEmergenceSimulator Class

```python
class SpacetimeEmergenceSimulator:
    """Simulate emergence of spacetime geometry from quantum entanglement."""

    def __init__(self, n_subsystems, dimension=3, entanglement_strength=1.0):
        """
        Initialize spacetime emergence simulation.

        Parameters
        ----------
        n_subsystems : int
            Number of quantum subsystems
        dimension : int
            Target spatial dimension (default: 3)
        entanglement_strength : float
            Overall entanglement coupling strength
        """
        self.n_subsystems = n_subsystems
        self.dimension = dimension
        self.strength = entanglement_strength
        self.entanglement_matrix = None
        self.evolution_history = []

    def initialize_quantum_state(self, pattern='random'):
        """
        Initialize quantum state with specified entanglement pattern.

        Parameters
        ----------
        pattern : str
            Entanglement pattern: 'random', 'local', 'global', 'cluster'

        Returns
        -------
        np.ndarray
            Initial quantum state vector
        """
        if pattern == 'random':
            # Random entangled state
            state = np.random.complex128(2**self.n_subsystems)
            state /= np.linalg.norm(state)
        elif pattern == 'local':
            # Nearest-neighbor entanglement
            state = self.create_local_entanglement()
        elif pattern == 'global':
            # All-to-all entanglement
            state = self.create_global_entanglement()
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return state

    def evolve_system(self, steps=100, dt=0.1):
        """
        Evolve the quantum system and track geometric emergence.

        Parameters
        ----------
        steps : int
            Number of evolution steps
        dt : float
            Time step size

        Returns
        -------
        dict
            Evolution data including entanglement and geometry metrics
        """
        evolution_data = {
            'time': [],
            'total_entanglement': [],
            'average_entanglement': [],
            'dimensional_measure': [],
            'geometric_entropy': []
        }

        for step in range(steps):
            # Time evolution
            self.time_step(dt)

            # Calculate metrics
            t = step * dt
            total_ent = self.calculate_total_entanglement()
            avg_ent = total_ent / self.n_subsystems
            dim_measure = self.calculate_dimensional_measure()
            geom_entropy = self.calculate_geometric_entropy()

            # Store data
            evolution_data['time'].append(t)
            evolution_data['total_entanglement'].append(total_ent)
            evolution_data['average_entanglement'].append(avg_ent)
            evolution_data['dimensional_measure'].append(dim_measure)
            evolution_data['geometric_entropy'].append(geom_entropy)

        return evolution_data

    def get_simulation_summary(self):
        """
        Generate summary statistics for the simulation.

        Returns
        -------
        dict
            Summary including final metrics and system properties
        """
        return {
            'n_subsystems': self.n_subsystems,
            'target_dimension': self.dimension,
            'total_entanglement': self.calculate_total_entanglement(),
            'average_entanglement': self.calculate_average_entanglement(),
            'effective_dimension': self.calculate_dimensional_measure(),
            'entanglement_distribution': self.analyze_entanglement_distribution()
        }
```

**Usage Example**:

```python
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator

# Create simulator
simulator = SpacetimeEmergenceSimulator(
    n_subsystems=50,
    dimension=3,
    entanglement_strength=1.0
)

# Initialize quantum state
initial_state = simulator.initialize_quantum_state(pattern='local')

# Run evolution
evolution_data = simulator.evolve_system(steps=100, dt=0.1)

# Get summary
summary = simulator.get_simulation_summary()
print(f"Final entanglement: {summary['total_entanglement']:.2f}")
print(f"Effective dimension: {summary['effective_dimension']:.2f}")
```

### black_hole_simulator.py

#### BlackHoleSimulator Class

```python
class BlackHoleSimulator:
    """Simulate black hole physics including Hawking radiation and information scrambling."""

    def __init__(self, mass, charge=0, angular_momentum=0):
        """
        Initialize black hole simulator.

        Parameters
        ----------
        mass : float
            Black hole mass (in solar masses)
        charge : float
            Electric charge (default: 0)
        angular_momentum : float
            Dimensionless spin parameter a/M (default: 0)
        """
        self.mass = mass * 1.989e30  # Convert to kg
        self.charge = charge
        self.spin = angular_momentum
        self.constants = PhysicalConstants()

        # Derived properties
        self.schwarzschild_radius = self.constants.schwarzschild_radius(self.mass)
        self.hawking_temperature = self.calculate_hawking_temperature()

    def calculate_hawking_temperature(self):
        """
        Calculate Hawking temperature: T = ħc³/(8πGMk_B)

        Returns
        -------
        float
            Hawking temperature in Kelvin
        """
        return (self.constants.hbar * self.constants.c**3) / \
               (8 * np.pi * self.constants.G * self.mass * self.constants.k_B)

    def compute_hawking_radiation(self, time_points):
        """
        Simulate Hawking radiation emission over time.

        Parameters
        ----------
        time_points : int
            Number of time points to simulate

        Returns
        -------
        dict
            Radiation data including spectrum and total flux
        """
        times = np.linspace(0, 1e12, time_points)  # 1e12 seconds

        # Hawking radiation spectrum (simplified)
        frequencies = np.logspace(10, 20, 100)  # Hz
        spectrum = self.hawking_spectrum(frequencies)

        # Time evolution of mass due to evaporation
        mass_evolution = self.mass_evaporation_curve(times)

        return {
            'times': times,
            'mass_evolution': mass_evolution,
            'frequencies': frequencies,
            'spectrum': spectrum,
            'temperature': self.hawking_temperature,
            'luminosity': self.hawking_luminosity()
        }

    def compute_information_scrambling(self, n_qubits=6):
        """
        Simulate quantum information scrambling in black hole.

        Parameters
        ----------
        n_qubits : int
            Number of qubits to simulate (default: 6 for performance)

        Returns
        -------
        dict
            Scrambling data including entanglement entropy evolution
        """
        # Reduced system size for performance
        hilbert_dim = 2**n_qubits

        # Create random Hamiltonian (chaotic system)
        H = np.random.randn(hilbert_dim, hilbert_dim) + \
            1j * np.random.randn(hilbert_dim, hilbert_dim)
        H = (H + H.conj().T) / 2  # Make Hermitian

        # Diagonalize for efficient time evolution
        eigenvals, eigenvecs = np.linalg.eigh(H)

        # Time evolution
        times = np.linspace(0, 10, 50)  # Reduced time points
        scrambling_data = {
            'times': times,
            'entanglement_entropy': [],
            'mutual_information': [],
            'scrambling_rate': []
        }

        # Initial state (product state)
        initial_state = np.zeros(hilbert_dim)
        initial_state[0] = 1.0

        for t in times:
            # Efficient time evolution using diagonalization
            U_t = eigenvecs @ np.diag(np.exp(-1j * eigenvals * t)) @ eigenvecs.conj().T
            state_t = U_t @ initial_state

            # Calculate entanglement entropy
            entropy = self.calculate_entanglement_entropy(state_t, n_qubits)
            scrambling_data['entanglement_entropy'].append(entropy)

        return scrambling_data
```

**Usage Example**:

```python
from simulations.black_hole_simulator import BlackHoleSimulator

# Create black hole (10 solar masses, spinning)
bh = BlackHoleSimulator(mass=10, charge=0, angular_momentum=0.5)

# Compute Hawking radiation
radiation_data = bh.compute_hawking_radiation(time_points=100)
print(f"Hawking temperature: {radiation_data['temperature']:.2e} K")

# Simulate information scrambling
scrambling_data = bh.compute_information_scrambling(n_qubits=6)
print(f"Max entanglement entropy: {max(scrambling_data['entanglement_entropy']):.2f}")
```

## 🧪 Experiment Modules

### predictions.py

#### Key Functions

```python
def generate_experimental_predictions():
    """
    Generate comprehensive experimental predictions from EG-QGEM theory.

    Returns
    -------
    dict
        Experimental predictions organized by category
    """
    predictions = {
        'gravitational_decoherence': {
            'description': 'Quantum superpositions decohere in gravitational fields',
            'observable': 'Decoherence rate vs. gravitational potential',
            'prediction': 'Γ ∝ GM/rc²',
            'test_systems': ['Atomic interferometry', 'Superconducting qubits'],
            'sensitivity': 1e-18  # Fractional measurement precision needed
        },

        'modified_dispersion': {
            'description': 'Particle dispersion modified by entanglement fields',
            'observable': 'Energy-momentum relation deviations',
            'prediction': 'E² = p²c² + m²c⁴ + δE_ent',
            'test_systems': ['High-energy cosmic rays', 'Neutrino oscillations'],
            'sensitivity': 1e-15
        },

        'entanglement_shadows': {
            'description': 'Regions of reduced entanglement create observable effects',
            'observable': 'Gravitational lensing anomalies',
            'prediction': 'Shadow patterns in CMB and galaxy surveys',
            'test_systems': ['CMB observations', 'Weak lensing surveys'],
            'sensitivity': 1e-6
        }
    }

    return predictions

class DecoherenceExperiment:
    """Design and simulate gravitational decoherence experiments."""

    def __init__(self, mass_source, size_source, test_particle_mass):
        """
        Initialize decoherence experiment.

        Parameters
        ----------
        mass_source : float
            Mass of gravitational source (kg)
        size_source : float
            Characteristic size of source (m)
        test_particle_mass : float
            Mass of test particle (kg)
        """
        self.M = mass_source
        self.R = size_source
        self.m = test_particle_mass
        self.constants = PhysicalConstants()

    def calculate_decoherence_rate(self, distance):
        """
        Calculate gravitational decoherence rate at given distance.

        Parameters
        ----------
        distance : float
            Distance from gravitational source (m)

        Returns
        -------
        float
            Decoherence rate Γ (Hz)
        """
        # EG-QGEM prediction: Γ ∝ GM/rc²
        gravitational_potential = self.constants.G * self.M / distance
        decoherence_rate = (self.m * gravitational_potential) / \
                          (self.constants.hbar * self.constants.c**2)

        return decoherence_rate

    def simulate_experiment(self, distances, measurement_time):
        """
        Simulate full decoherence experiment.

        Parameters
        ----------
        distances : array_like
            Array of distances to test (m)
        measurement_time : float
            Total measurement time (s)

        Returns
        -------
        dict
            Experimental results including decoherence curves
        """
        results = {
            'distances': distances,
            'decoherence_rates': [],
            'coherence_times': [],
            'visibility': []
        }

        for d in distances:
            gamma = self.calculate_decoherence_rate(d)
            coherence_time = 1 / gamma if gamma > 0 else np.inf
            visibility = np.exp(-gamma * measurement_time)

            results['decoherence_rates'].append(gamma)
            results['coherence_times'].append(coherence_time)
            results['visibility'].append(visibility)

        return results
```

**Usage Example**:

```python
from experiments.predictions import generate_experimental_predictions, DecoherenceExperiment

# Get all theoretical predictions
predictions = generate_experimental_predictions()
for exp_name, pred in predictions.items():
    print(f"{exp_name}: {pred['description']}")

# Design specific decoherence experiment
experiment = DecoherenceExperiment(
    mass_source=1000,      # 1 kg source mass
    size_source=0.1,       # 10 cm source size
    test_particle_mass=1e-27  # Atomic mass
)

# Simulate experiment
distances = np.logspace(-3, 0, 20)  # 1 mm to 1 m
results = experiment.simulate_experiment(distances, measurement_time=1.0)

print(f"Decoherence rate at 1 cm: {results['decoherence_rates'][5]:.2e} Hz")
```

## 🎨 Visualization Modules

### plotting.py

#### Visualization Classes

```python
class SpacetimeVisualizer:
    """Create visualizations for spacetime emergence simulations."""

    def __init__(self, style='scientific'):
        """
        Initialize visualizer with specified style.

        Parameters
        ----------
        style : str
            Plot style: 'scientific', 'presentation', 'publication'
        """
        self.style = style
        self.setup_style()

    def plot_entanglement_evolution(self, evolution_data, save_path=None):
        """
        Create plot showing entanglement evolution over time.

        Parameters
        ----------
        evolution_data : dict
            Evolution data from SpacetimeEmergenceSimulator
        save_path : str, optional
            Path to save plot (if None, display only)

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Total entanglement vs time
        axes[0,0].plot(evolution_data['time'], evolution_data['total_entanglement'])
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Total Entanglement')
        axes[0,0].set_title('Entanglement Growth')

        # Dimensional emergence
        axes[0,1].plot(evolution_data['time'], evolution_data['dimensional_measure'])
        axes[0,1].axhline(y=3, color='r', linestyle='--', label='Target dimension')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Effective Dimension')
        axes[0,1].set_title('Dimensional Emergence')
        axes[0,1].legend()

        # Average entanglement per subsystem
        axes[1,0].plot(evolution_data['time'], evolution_data['average_entanglement'])
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Average Entanglement')
        axes[1,0].set_title('Entanglement per Subsystem')

        # Geometric entropy
        axes[1,1].plot(evolution_data['time'], evolution_data['geometric_entropy'])
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Geometric Entropy')
        axes[1,1].set_title('Information Content')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

class BlackHoleVisualizer:
    """Visualize black hole simulation results."""

    def plot_hawking_radiation(self, radiation_data, save_path=None):
        """Create comprehensive Hawking radiation visualization."""
        # Implementation details...

    def plot_information_scrambling(self, scrambling_data, save_path=None):
        """Visualize quantum information scrambling dynamics."""
        # Implementation details...

class ExperimentVisualizer:
    """Visualize experimental predictions and results."""

    def plot_decoherence_predictions(self, experiment_results, save_path=None):
        """Create plots for gravitational decoherence experiments."""
        # Implementation details...
```

**Usage Example**:

```python
from visualization.plotting import SpacetimeVisualizer, BlackHoleVisualizer
from simulations.spacetime_emergence import run_emergence_simulation

# Run simulation
simulator, evolution_data = run_emergence_simulation(n_subsystems=50, steps=100)

# Create visualizations
viz = SpacetimeVisualizer(style='publication')
fig = viz.plot_entanglement_evolution(evolution_data, save_path='emergence.png')

# Display results
plt.show()
```

---

This API reference provides complete documentation for all core EG-QGEM modules. Each class and function includes detailed parameter descriptions, return values, and practical usage examples to facilitate research and development.
