"""
Synthetic Dataset Generator for EG-QGEM Research
===============================================

This module generates realistic synthetic datasets for testing and demonstration
of the EG-QGEM framework, including spacetime emergence data, black hole
dynamics, experimental signatures, and cosmological observations.
"""

import numpy as np
import pandas as pd
import h5py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import networkx as nx

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory.constants import CONSTANTS
from simulations.spacetime_emergence import SpacetimeEmergenceSimulator
from simulations.black_hole_simulator import BlackHoleSimulator

class SyntheticDataGenerator:
    """
    Generate synthetic datasets for EG-QGEM research.
    """

    def __init__(self, output_dir: str = "data/synthetic"):
        """Initialize the synthetic data generator."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories for different data types
        self.subdirs = {
            'spacetime': os.path.join(self.output_dir, 'spacetime_emergence'),
            'black_holes': os.path.join(self.output_dir, 'black_hole_dynamics'),
            'experiments': os.path.join(self.output_dir, 'experimental_data'),
            'cosmology': os.path.join(self.output_dir, 'cosmological_observations'),
            'entanglement': os.path.join(self.output_dir, 'entanglement_networks'),
            'gravitational_waves': os.path.join(self.output_dir, 'gravitational_waves')
        }

        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)

    def generate_spacetime_emergence_dataset(self,
                                           n_samples: int = 1000,
                                           grid_size: Tuple[int, int, int] = (32, 32, 32),
                                           time_steps: int = 100) -> Dict[str, Any]:
        """
        Generate synthetic spacetime emergence data.

        Args:
            n_samples: Number of emergence scenarios
            grid_size: Spatial grid dimensions
            time_steps: Number of temporal evolution steps

        Returns:
            Dictionary containing emergence datasets
        """
        print(f"🌌 Generating spacetime emergence dataset ({n_samples} samples)...")

        datasets = {
            'emergence_scenarios': [],
            'metric_tensors': [],
            'entanglement_densities': [],
            'curvature_scalars': [],
            'emergence_timescales': [],
            'critical_parameters': []
        }

        for i in range(n_samples):
            # Generate random initial conditions
            entanglement_coupling = np.random.uniform(1e-5, 1e-2)
            initial_entropy = np.random.uniform(0.1, 0.9)
            symmetry_breaking = np.random.choice(['spontaneous', 'explicit', 'none'])

            # Simulate spacetime emergence
            x, y, z = np.meshgrid(
                np.linspace(-1, 1, grid_size[0]),
                np.linspace(-1, 1, grid_size[1]),
                np.linspace(-1, 1, grid_size[2])
            )

            # Generate metric tensor evolution
            metric_evolution = np.zeros((time_steps, *grid_size, 4, 4))
            entanglement_evolution = np.zeros((time_steps, *grid_size))
            curvature_evolution = np.zeros((time_steps, *grid_size))

            for t in range(time_steps):
                time_factor = t / time_steps

                # Evolving metric components
                for mu in range(4):
                    for nu in range(4):
                        if mu == nu:
                            if mu == 0:  # Time component
                                metric_evolution[t, :, :, :, mu, nu] = -(1 +
                                    entanglement_coupling * np.sin(2*np.pi*time_factor) *
                                    np.exp(-(x**2 + y**2 + z**2)))
                            else:  # Spatial components
                                metric_evolution[t, :, :, :, mu, nu] = (1 +
                                    entanglement_coupling * np.cos(2*np.pi*time_factor) *
                                    np.exp(-(x**2 + y**2 + z**2)))

                # Entanglement density evolution
                entanglement_evolution[t] = initial_entropy * np.exp(
                    -entanglement_coupling * time_factor * (x**2 + y**2 + z**2)
                ) * (1 + 0.1 * np.random.normal(0, 1, grid_size))

                # Curvature scalar
                curvature_evolution[t] = entanglement_coupling * np.sin(
                    2*np.pi*time_factor
                ) * np.exp(-(x**2 + y**2 + z**2))

            # Calculate emergence timescale
            entanglement_avg = np.mean(entanglement_evolution, axis=(1, 2, 3))
            emergence_time = np.argmax(np.gradient(entanglement_avg)) if len(entanglement_avg) > 1 else time_steps//2

            datasets['emergence_scenarios'].append({
                'scenario_id': f'emergence_{i:04d}',
                'initial_conditions': {
                    'entanglement_coupling': entanglement_coupling,
                    'initial_entropy': initial_entropy,
                    'symmetry_breaking': symmetry_breaking
                },
                'grid_size': grid_size,
                'time_steps': time_steps
            })

            datasets['metric_tensors'].append(metric_evolution)
            datasets['entanglement_densities'].append(entanglement_evolution)
            datasets['curvature_scalars'].append(curvature_evolution)
            datasets['emergence_timescales'].append(emergence_time)
            datasets['critical_parameters'].append({
                'coupling_strength': entanglement_coupling,
                'symmetry_parameter': np.random.uniform(-1, 1),
                'dimensionality': len(grid_size)
            })

        # Save dataset
        filename = os.path.join(self.subdirs['spacetime'], 'emergence_dataset.h5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('metric_tensors', data=np.array(datasets['metric_tensors']))
            f.create_dataset('entanglement_densities', data=np.array(datasets['entanglement_densities']))
            f.create_dataset('curvature_scalars', data=np.array(datasets['curvature_scalars']))
            f.create_dataset('emergence_timescales', data=np.array(datasets['emergence_timescales']))

            # Save metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['n_samples'] = n_samples
            metadata_group.attrs['grid_size'] = grid_size
            metadata_group.attrs['time_steps'] = time_steps
            metadata_group.attrs['generation_time'] = str(datetime.now())

        # Save scenario metadata as JSON
        with open(os.path.join(self.subdirs['spacetime'], 'scenarios_metadata.json'), 'w') as f:
            json.dump({
                'scenarios': datasets['emergence_scenarios'],
                'critical_parameters': datasets['critical_parameters']
            }, f, indent=2)

        print(f"✅ Spacetime emergence dataset saved to {filename}")
        return datasets

    def generate_black_hole_dataset(self,
                                  n_black_holes: int = 500,
                                  time_duration: float = 1000.0) -> Dict[str, Any]:
        """
        Generate synthetic black hole dynamics data.

        Args:
            n_black_holes: Number of black hole scenarios
            time_duration: Simulation time in Planck units

        Returns:
            Dictionary containing black hole datasets
        """
        print(f"🕳️ Generating black hole dataset ({n_black_holes} black holes)...")

        datasets = {
            'black_hole_parameters': [],
            'hawking_radiation': [],
            'information_scrambling': [],
            'entanglement_entropy': [],
            'horizon_dynamics': []
        }

        time_points = np.linspace(0, time_duration, 1000)

        for i in range(n_black_holes):
            # Random black hole parameters
            mass = np.random.lognormal(np.log(10), 1)  # Solar masses
            spin = np.random.uniform(0, 0.998)  # Dimensionless spin
            charge = np.random.uniform(0, 0.5)  # Dimensionless charge

            # Schwarzschild radius
            r_s = 2 * CONSTANTS['G'] * mass * CONSTANTS['M_sun'] / CONSTANTS['c']**2

            # Hawking temperature
            T_hawking = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * mass * CONSTANTS['M_sun'] * CONSTANTS['k_B'])

            # Generate time-dependent quantities
            hawking_flux = T_hawking**4 * (1 + 0.1 * np.sin(time_points / 100) +
                                         0.05 * np.random.normal(0, 1, len(time_points)))

            # Information scrambling time
            scrambling_time = r_s / CONSTANTS['c'] * np.log(mass)
            scrambling_efficiency = 1 - np.exp(-time_points / scrambling_time)

            # Entanglement entropy (Page curve)
            max_entropy = mass  # Proportional to area
            page_time = scrambling_time * 2
            entanglement_entropy = np.where(
                time_points < page_time,
                max_entropy * time_points / page_time,
                max_entropy * (2 - time_points / page_time)
            )
            entanglement_entropy = np.maximum(entanglement_entropy, 0)

            # Horizon dynamics
            horizon_radius = r_s * (1 + 0.01 * spin * np.sin(time_points / 50))

            datasets['black_hole_parameters'].append({
                'bh_id': f'bh_{i:04d}',
                'mass_solar': mass,
                'spin': spin,
                'charge': charge,
                'schwarzschild_radius': r_s,
                'hawking_temperature': T_hawking
            })

            datasets['hawking_radiation'].append(hawking_flux)
            datasets['information_scrambling'].append(scrambling_efficiency)
            datasets['entanglement_entropy'].append(entanglement_entropy)
            datasets['horizon_dynamics'].append(horizon_radius)

        # Save dataset
        filename = os.path.join(self.subdirs['black_holes'], 'black_hole_dataset.h5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('time_points', data=time_points)
            f.create_dataset('hawking_radiation', data=np.array(datasets['hawking_radiation']))
            f.create_dataset('information_scrambling', data=np.array(datasets['information_scrambling']))
            f.create_dataset('entanglement_entropy', data=np.array(datasets['entanglement_entropy']))
            f.create_dataset('horizon_dynamics', data=np.array(datasets['horizon_dynamics']))

            # Metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['n_black_holes'] = n_black_holes
            metadata_group.attrs['time_duration'] = time_duration
            metadata_group.attrs['generation_time'] = str(datetime.now())

        # Save parameters as JSON
        with open(os.path.join(self.subdirs['black_holes'], 'bh_parameters.json'), 'w') as f:
            json.dump({'black_holes': datasets['black_hole_parameters']}, f, indent=2)

        print(f"✅ Black hole dataset saved to {filename}")
        return datasets

    def generate_experimental_dataset(self,
                                    n_experiments: int = 200,
                                    experiment_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate synthetic experimental data.

        Args:
            n_experiments: Number of experimental runs
            experiment_types: Types of experiments to simulate

        Returns:
            Dictionary containing experimental datasets
        """
        if experiment_types is None:
            experiment_types = ['gravitational_decoherence', 'precision_tests',
                              'cosmological_signatures', 'lab_experiments']

        print(f"🔬 Generating experimental dataset ({n_experiments} experiments)...")

        datasets = {
            'experiment_metadata': [],
            'measurements': [],
            'uncertainties': [],
            'theoretical_predictions': [],
            'residuals': []
        }

        for i in range(n_experiments):
            exp_type = np.random.choice(experiment_types)

            if exp_type == 'gravitational_decoherence':
                # Simulate decoherence rate measurements
                true_coupling = np.random.uniform(1e-6, 1e-3)
                mass_scale = np.random.uniform(1e-15, 1e-12)  # kg

                n_points = 50
                masses = np.logspace(np.log10(mass_scale/10), np.log10(mass_scale*10), n_points)

                # Theoretical prediction
                decoherence_rate = true_coupling * masses / CONSTANTS['hbar']

                # Add experimental noise
                measurement_error = 0.1 * decoherence_rate
                measured_rate = decoherence_rate + np.random.normal(0, measurement_error)

                datasets['experiment_metadata'].append({
                    'exp_id': f'exp_{i:04d}',
                    'type': exp_type,
                    'coupling_strength': true_coupling,
                    'mass_scale': mass_scale,
                    'n_measurements': n_points
                })

                datasets['measurements'].append(measured_rate)
                datasets['uncertainties'].append(measurement_error)
                datasets['theoretical_predictions'].append(decoherence_rate)
                datasets['residuals'].append(measured_rate - decoherence_rate)

            elif exp_type == 'precision_tests':
                # Simulate precision tests of general relativity
                n_points = 30
                distances = np.logspace(6, 12, n_points)  # meters

                # Einstein prediction
                einstein_prediction = 1 / distances**2

                # EG-QGEM correction
                entanglement_correction = np.random.uniform(1e-8, 1e-6)
                egqgem_prediction = einstein_prediction * (1 + entanglement_correction / distances)

                # Measurements with uncertainty
                measurement_precision = 1e-8
                measured_values = egqgem_prediction + np.random.normal(0, measurement_precision, n_points)

                datasets['experiment_metadata'].append({
                    'exp_id': f'exp_{i:04d}',
                    'type': exp_type,
                    'entanglement_correction': entanglement_correction,
                    'measurement_precision': measurement_precision,
                    'n_measurements': n_points
                })

                datasets['measurements'].append(measured_values)
                datasets['uncertainties'].append(np.full(n_points, measurement_precision))
                datasets['theoretical_predictions'].append(egqgem_prediction)
                datasets['residuals'].append(measured_values - egqgem_prediction)

        # Save dataset
        filename = os.path.join(self.subdirs['experiments'], 'experimental_dataset.h5')
        with h5py.File(filename, 'w') as f:
            # Save measurements (variable length arrays)
            measurements_group = f.create_group('measurements')
            uncertainties_group = f.create_group('uncertainties')
            predictions_group = f.create_group('theoretical_predictions')
            residuals_group = f.create_group('residuals')

            for i, (meas, unc, pred, res) in enumerate(zip(
                datasets['measurements'], datasets['uncertainties'],
                datasets['theoretical_predictions'], datasets['residuals']
            )):
                measurements_group.create_dataset(f'exp_{i:04d}', data=meas)
                uncertainties_group.create_dataset(f'exp_{i:04d}', data=unc)
                predictions_group.create_dataset(f'exp_{i:04d}', data=pred)
                residuals_group.create_dataset(f'exp_{i:04d}', data=res)

            # Metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['n_experiments'] = n_experiments
            metadata_group.attrs['experiment_types'] = experiment_types
            metadata_group.attrs['generation_time'] = str(datetime.now())

        # Save metadata as JSON
        with open(os.path.join(self.subdirs['experiments'], 'experiment_metadata.json'), 'w') as f:
            json.dump({'experiments': datasets['experiment_metadata']}, f, indent=2)

        print(f"✅ Experimental dataset saved to {filename}")
        return datasets

    def generate_entanglement_networks(self,
                                     n_networks: int = 100,
                                     network_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Generate synthetic entanglement network data.

        Args:
            n_networks: Number of entanglement networks
            network_sizes: Range of network sizes to generate

        Returns:
            Dictionary containing network datasets
        """
        if network_sizes is None:
            network_sizes = [10, 20, 50, 100, 200]

        print(f"🕸️ Generating entanglement networks ({n_networks} networks)...")

        datasets = {
            'network_metadata': [],
            'adjacency_matrices': [],
            'entanglement_weights': [],
            'geometric_properties': [],
            'topological_measures': []
        }

        for i in range(n_networks):
            n_nodes = np.random.choice(network_sizes)
            network_type = np.random.choice(['small_world', 'scale_free', 'random', 'lattice'])

            # Generate base network topology
            if network_type == 'small_world':
                G = nx.watts_strogatz_graph(n_nodes, 4, 0.3)
            elif network_type == 'scale_free':
                G = nx.barabasi_albert_graph(n_nodes, 3)
            elif network_type == 'random':
                G = nx.erdos_renyi_graph(n_nodes, 0.1)
            else:  # lattice
                side_length = int(np.sqrt(n_nodes))
                G = nx.grid_2d_graph(side_length, side_length)
                G = nx.convert_node_labels_to_integers(G)

            # Generate entanglement weights
            adjacency = nx.adjacency_matrix(G).toarray()
            entanglement_weights = np.zeros_like(adjacency, dtype=float)

            for edge in G.edges():
                # Entanglement strength based on distance and random factors
                weight = np.random.exponential(0.5) * np.random.uniform(0.1, 1.0)
                entanglement_weights[edge[0], edge[1]] = weight
                entanglement_weights[edge[1], edge[0]] = weight

            # Calculate network properties
            clustering = nx.average_clustering(G)
            path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else np.inf
            density = nx.density(G)

            # Geometric properties (assign positions and calculate geometric measures)
            pos = nx.spring_layout(G, dim=3)
            positions = np.array([pos[node] for node in G.nodes()])

            # Calculate entanglement entropy
            eigenvals = np.linalg.eigvals(entanglement_weights)
            eigenvals = eigenvals[eigenvals > 1e-12]
            entanglement_entropy = -np.sum(eigenvals * np.log(eigenvals)) if len(eigenvals) > 0 else 0

            datasets['network_metadata'].append({
                'network_id': f'net_{i:04d}',
                'n_nodes': n_nodes,
                'n_edges': G.number_of_edges(),
                'network_type': network_type,
                'is_connected': nx.is_connected(G)
            })

            datasets['adjacency_matrices'].append(adjacency)
            datasets['entanglement_weights'].append(entanglement_weights)
            datasets['geometric_properties'].append({
                'positions': positions.tolist(),
                'spatial_dimension': 3
            })
            datasets['topological_measures'].append({
                'clustering_coefficient': clustering,
                'average_path_length': path_length,
                'network_density': density,
                'entanglement_entropy': entanglement_entropy
            })

        # Save networks
        filename = os.path.join(self.subdirs['entanglement'], 'entanglement_networks.h5')
        with h5py.File(filename, 'w') as f:
            # Save adjacency matrices and weights
            adj_group = f.create_group('adjacency_matrices')
            weights_group = f.create_group('entanglement_weights')

            for i, (adj, weights) in enumerate(zip(datasets['adjacency_matrices'],
                                                  datasets['entanglement_weights'])):
                adj_group.create_dataset(f'net_{i:04d}', data=adj)
                weights_group.create_dataset(f'net_{i:04d}', data=weights)

            # Metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['n_networks'] = n_networks
            metadata_group.attrs['network_sizes'] = network_sizes
            metadata_group.attrs['generation_time'] = str(datetime.now())

        # Save detailed metadata as JSON
        with open(os.path.join(self.subdirs['entanglement'], 'network_metadata.json'), 'w') as f:
            json.dump({
                'networks': datasets['network_metadata'],
                'geometric_properties': datasets['geometric_properties'],
                'topological_measures': datasets['topological_measures']
            }, f, indent=2)

        print(f"✅ Entanglement networks saved to {filename}")
        return datasets

    def generate_gravitational_wave_dataset(self,
                                          n_signals: int = 300,
                                          duration: float = 4.0,
                                          sample_rate: float = 4096.0) -> Dict[str, Any]:
        """
        Generate synthetic gravitational wave data with EG-QGEM modifications.

        Args:
            n_signals: Number of GW signals
            duration: Signal duration in seconds
            sample_rate: Sampling rate in Hz

        Returns:
            Dictionary containing GW datasets
        """
        print(f"🌊 Generating gravitational wave dataset ({n_signals} signals)...")

        t = np.arange(0, duration, 1/sample_rate)
        datasets = {
            'signal_metadata': [],
            'waveforms_h_plus': [],
            'waveforms_h_cross': [],
            'entanglement_signatures': [],
            'noise_realizations': []
        }

        for i in range(n_signals):
            # Random binary parameters
            m1 = np.random.uniform(5, 50)  # Solar masses
            m2 = np.random.uniform(5, 50)
            distance = np.random.uniform(100, 1000)  # Mpc
            spin1 = np.random.uniform(0, 0.9)
            spin2 = np.random.uniform(0, 0.9)

            # Entanglement modification parameters
            entanglement_coupling = np.random.uniform(1e-6, 1e-3)
            modification_frequency = np.random.uniform(10, 1000)  # Hz

            # Generate basic inspiral waveform (simplified)
            chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            frequency_start = 20  # Hz
            frequency_end = 500   # Hz

            # Frequency evolution
            tau = 5 * CONSTANTS['c']**5 / (256 * CONSTANTS['G']**(5/3) * (np.pi * chirp_mass * CONSTANTS['M_sun'])**(5/3))
            t_coalesce = duration * 0.8  # Coalescence time
            f_t = (frequency_start**(-8/3) + (8*np.pi**(8/3)*CONSTANTS['G']**(5/3)*chirp_mass**(5/3)*CONSTANTS['M_sun']**(5/3)/5/CONSTANTS['c']**5) * (t_coalesce - t))**(-3/8)
            f_t = np.maximum(f_t, frequency_start)
            f_t = np.minimum(f_t, frequency_end)

            # Phase evolution
            phase = 2 * np.pi * np.cumsum(f_t) / sample_rate

            # Basic strain amplitude
            amplitude = (CONSTANTS['G'] * chirp_mass * CONSTANTS['M_sun'] / CONSTANTS['c']**2)**(5/6) / distance / 1e6 / 3.086e22  # Mpc to m
            amplitude *= (np.pi * f_t * CONSTANTS['G'] * chirp_mass * CONSTANTS['M_sun'] / CONSTANTS['c']**3)**(-7/6)

            # Standard GR waveform
            h_plus_gr = amplitude * np.cos(phase)
            h_cross_gr = amplitude * np.sin(phase)

            # EG-QGEM entanglement modification
            entanglement_phase = entanglement_coupling * np.sin(2 * np.pi * modification_frequency * t)
            h_plus = h_plus_gr * (1 + entanglement_phase)
            h_cross = h_cross_gr * (1 + entanglement_phase)

            # Entanglement signature (separate from main signal)
            entanglement_signature = entanglement_coupling * amplitude * np.sin(
                2 * np.pi * modification_frequency * t + np.pi/4
            )

            # Add noise
            strain_noise_level = 1e-23  # Typical advanced LIGO sensitivity
            noise_h_plus = np.random.normal(0, strain_noise_level, len(t))
            noise_h_cross = np.random.normal(0, strain_noise_level, len(t))

            datasets['signal_metadata'].append({
                'signal_id': f'gw_{i:04d}',
                'mass1': m1,
                'mass2': m2,
                'distance_mpc': distance,
                'spin1': spin1,
                'spin2': spin2,
                'chirp_mass': chirp_mass,
                'entanglement_coupling': entanglement_coupling,
                'modification_frequency': modification_frequency,
                'snr_estimate': np.sqrt(np.sum(amplitude**2)) / strain_noise_level
            })

            datasets['waveforms_h_plus'].append(h_plus)
            datasets['waveforms_h_cross'].append(h_cross)
            datasets['entanglement_signatures'].append(entanglement_signature)
            datasets['noise_realizations'].append({
                'h_plus_noise': noise_h_plus,
                'h_cross_noise': noise_h_cross
            })

        # Save dataset
        filename = os.path.join(self.subdirs['gravitational_waves'], 'gw_dataset.h5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('time_array', data=t)
            f.create_dataset('waveforms_h_plus', data=np.array(datasets['waveforms_h_plus']))
            f.create_dataset('waveforms_h_cross', data=np.array(datasets['waveforms_h_cross']))
            f.create_dataset('entanglement_signatures', data=np.array(datasets['entanglement_signatures']))

            # Noise data
            noise_group = f.create_group('noise_realizations')
            for i, noise_data in enumerate(datasets['noise_realizations']):
                noise_subgroup = noise_group.create_group(f'gw_{i:04d}')
                noise_subgroup.create_dataset('h_plus_noise', data=noise_data['h_plus_noise'])
                noise_subgroup.create_dataset('h_cross_noise', data=noise_data['h_cross_noise'])

            # Metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['n_signals'] = n_signals
            metadata_group.attrs['duration'] = duration
            metadata_group.attrs['sample_rate'] = sample_rate
            metadata_group.attrs['generation_time'] = str(datetime.now())

        # Save signal metadata as JSON
        with open(os.path.join(self.subdirs['gravitational_waves'], 'gw_metadata.json'), 'w') as f:
            json.dump({'signals': datasets['signal_metadata']}, f, indent=2)

        print(f"✅ Gravitational wave dataset saved to {filename}")
        return datasets

    def create_dataset_connections(self) -> Dict[str, Any]:
        """
        Create connections and cross-references between different datasets.

        Returns:
            Dictionary containing dataset connections and relationships
        """
        print("🔗 Creating dataset connections...")

        connections = {
            'spacetime_to_black_holes': [],
            'black_holes_to_experiments': [],
            'experiments_to_gw': [],
            'entanglement_to_all': [],
            'cross_validation_sets': []
        }

        # Load existing dataset metadata
        metadata_files = {
            'spacetime': os.path.join(self.subdirs['spacetime'], 'scenarios_metadata.json'),
            'black_holes': os.path.join(self.subdirs['black_holes'], 'bh_parameters.json'),
            'experiments': os.path.join(self.subdirs['experiments'], 'experiment_metadata.json'),
            'entanglement': os.path.join(self.subdirs['entanglement'], 'network_metadata.json'),
            'gw': os.path.join(self.subdirs['gravitational_waves'], 'gw_metadata.json')
        }

        loaded_metadata = {}
        for key, filename in metadata_files.items():
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_metadata[key] = json.load(f)

        # Create spacetime-to-black-hole connections
        if 'spacetime' in loaded_metadata and 'black_holes' in loaded_metadata:
            spacetime_scenarios = loaded_metadata['spacetime'].get('scenarios', [])
            black_holes = loaded_metadata['black_holes'].get('black_holes', [])

            for scenario in spacetime_scenarios[:50]:  # Connect first 50
                # Find black holes with similar entanglement coupling
                scenario_coupling = scenario['initial_conditions']['entanglement_coupling']

                suitable_bhs = []
                for bh in black_holes:
                    # Use mass as proxy for entanglement effects
                    bh_coupling_proxy = 1e-4 / bh['mass_solar']  # Smaller mass = stronger quantum effects
                    if abs(np.log10(scenario_coupling) - np.log10(bh_coupling_proxy)) < 1.0:
                        suitable_bhs.append(bh['bh_id'])

                if suitable_bhs:
                    connections['spacetime_to_black_holes'].append({
                        'spacetime_scenario': scenario['scenario_id'],
                        'connected_black_holes': suitable_bhs[:3],  # Max 3 connections
                        'connection_type': 'entanglement_coupling_similarity'
                    })

        # Create black-hole-to-experiment connections
        if 'black_holes' in loaded_metadata and 'experiments' in loaded_metadata:
            black_holes = loaded_metadata['black_holes'].get('black_holes', [])
            experiments = loaded_metadata['experiments'].get('experiments', [])

            for bh in black_holes[:30]:  # Connect first 30
                # Find experiments that could probe this black hole's physics
                relevant_experiments = []

                for exp in experiments:
                    if exp['type'] == 'gravitational_decoherence':
                        # Check if experiment sensitivity overlaps with BH parameters
                        if exp.get('coupling_strength', 0) * 100 > 1e-4 / bh['mass_solar']:
                            relevant_experiments.append(exp['exp_id'])
                    elif exp['type'] == 'precision_tests':
                        # All precision tests relevant to black holes
                        relevant_experiments.append(exp['exp_id'])

                if relevant_experiments:
                    connections['black_holes_to_experiments'].append({
                        'black_hole': bh['bh_id'],
                        'relevant_experiments': relevant_experiments[:5],
                        'connection_type': 'parameter_range_overlap'
                    })

        # Create experiment-to-GW connections
        if 'experiments' in loaded_metadata and 'gw' in loaded_metadata:
            experiments = loaded_metadata['experiments'].get('experiments', [])
            gw_signals = loaded_metadata['gw'].get('signals', [])

            for exp in experiments:
                if exp['type'] in ['precision_tests', 'gravitational_decoherence']:
                    # Find GW signals with similar entanglement parameters
                    exp_coupling = exp.get('coupling_strength', exp.get('entanglement_correction', 1e-6))

                    matching_gw = []
                    for gw in gw_signals:
                        if abs(np.log10(exp_coupling) - np.log10(gw['entanglement_coupling'])) < 0.5:
                            matching_gw.append(gw['signal_id'])

                    if matching_gw:
                        connections['experiments_to_gw'].append({
                            'experiment': exp['exp_id'],
                            'connected_gw_signals': matching_gw[:3],
                            'connection_type': 'entanglement_parameter_match'
                        })

        # Create entanglement network connections to all datasets
        if 'entanglement' in loaded_metadata:
            networks = loaded_metadata['entanglement'].get('networks', [])

            for net in networks[:20]:  # Connect first 20 networks
                network_connections = {
                    'network': net['network_id'],
                    'connected_datasets': {}
                }

                # Connect to spacetime scenarios based on network size
                if 'spacetime' in loaded_metadata:
                    scenarios = loaded_metadata['spacetime'].get('scenarios', [])
                    # Larger networks correspond to more complex spacetime emergence
                    suitable_scenarios = [s['scenario_id'] for s in scenarios
                                        if s['initial_conditions']['initial_entropy'] > 0.5
                                        and net['n_nodes'] > 50]
                    network_connections['connected_datasets']['spacetime'] = suitable_scenarios[:2]

                # Connect to black holes based on connectivity
                if 'black_holes' in loaded_metadata:
                    black_holes = loaded_metadata['black_holes'].get('black_holes', [])
                    # Highly connected networks correspond to rapidly spinning black holes
                    network_density = net['n_edges'] / (net['n_nodes'] * (net['n_nodes'] - 1) / 2)
                    suitable_bhs = [bh['bh_id'] for bh in black_holes
                                  if bh['spin'] > network_density and bh['mass_solar'] < 20]
                    network_connections['connected_datasets']['black_holes'] = suitable_bhs[:2]

                connections['entanglement_to_all'].append(network_connections)

        # Create cross-validation sets
        all_dataset_types = list(loaded_metadata.keys())
        n_validation_sets = 5

        for i in range(n_validation_sets):
            validation_set = {
                'validation_set_id': f'cross_val_{i}',
                'datasets': {}
            }

            for dataset_type in all_dataset_types:
                if dataset_type in loaded_metadata:
                    # Select random subset for validation
                    if dataset_type == 'spacetime':
                        items = loaded_metadata[dataset_type].get('scenarios', [])
                    elif dataset_type == 'black_holes':
                        items = loaded_metadata[dataset_type].get('black_holes', [])
                    elif dataset_type == 'experiments':
                        items = loaded_metadata[dataset_type].get('experiments', [])
                    elif dataset_type == 'entanglement':
                        items = loaded_metadata[dataset_type].get('networks', [])
                    elif dataset_type == 'gw':
                        items = loaded_metadata[dataset_type].get('signals', [])
                    else:
                        items = []

                    if items:
                        n_select = min(10, len(items) // n_validation_sets)
                        start_idx = i * n_select
                        end_idx = start_idx + n_select
                        selected_items = [item[list(item.keys())[0]] for item in items[start_idx:end_idx]]
                        validation_set['datasets'][dataset_type] = selected_items

            connections['cross_validation_sets'].append(validation_set)

        # Save connections
        connections_file = os.path.join(self.output_dir, 'dataset_connections.json')
        with open(connections_file, 'w') as f:
            json.dump(connections, f, indent=2)

        print(f"✅ Dataset connections saved to {connections_file}")
        return connections

    def generate_comprehensive_dataset(self) -> Dict[str, Any]:
        """
        Generate all synthetic datasets and their connections.

        Returns:
            Dictionary containing all generated datasets and connections
        """
        print("🚀 Generating comprehensive EG-QGEM synthetic dataset...")
        print("=" * 60)

        # Generate individual datasets
        spacetime_data = self.generate_spacetime_emergence_dataset(n_samples=200)
        black_hole_data = self.generate_black_hole_dataset(n_black_holes=150)
        experimental_data = self.generate_experimental_dataset(n_experiments=100)
        entanglement_data = self.generate_entanglement_networks(n_networks=80)
        gw_data = self.generate_gravitational_wave_dataset(n_signals=120)

        # Create connections
        connections = self.create_dataset_connections()

        # Generate summary report
        summary = {
            'generation_time': str(datetime.now()),
            'total_datasets': 5,
            'datasets_summary': {
                'spacetime_emergence': {
                    'n_samples': len(spacetime_data['emergence_scenarios']),
                    'data_size_mb': self._estimate_data_size(spacetime_data),
                    'description': 'Spacetime emergence from entanglement networks'
                },
                'black_hole_dynamics': {
                    'n_samples': len(black_hole_data['black_hole_parameters']),
                    'data_size_mb': self._estimate_data_size(black_hole_data),
                    'description': 'Black hole information dynamics and Hawking radiation'
                },
                'experimental_data': {
                    'n_samples': len(experimental_data['experiment_metadata']),
                    'data_size_mb': self._estimate_data_size(experimental_data),
                    'description': 'Laboratory and precision test measurements'
                },
                'entanglement_networks': {
                    'n_samples': len(entanglement_data['network_metadata']),
                    'data_size_mb': self._estimate_data_size(entanglement_data),
                    'description': 'Quantum entanglement network topologies'
                },
                'gravitational_waves': {
                    'n_samples': len(gw_data['signal_metadata']),
                    'data_size_mb': self._estimate_data_size(gw_data),
                    'description': 'GW signals with entanglement modifications'
                }
            },
            'connections_summary': {
                'spacetime_to_black_holes': len(connections['spacetime_to_black_holes']),
                'black_holes_to_experiments': len(connections['black_holes_to_experiments']),
                'experiments_to_gw': len(connections['experiments_to_gw']),
                'entanglement_connections': len(connections['entanglement_to_all']),
                'cross_validation_sets': len(connections['cross_validation_sets'])
            }
        }

        # Save summary
        summary_file = os.path.join(self.output_dir, 'dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("=" * 60)
        print("🎉 Comprehensive synthetic dataset generation complete!")
        print(f"📊 Generated {summary['total_datasets']} interconnected datasets")
        print(f"📁 Data saved to: {self.output_dir}")
        print(f"📄 Summary report: {summary_file}")

        return {
            'spacetime_data': spacetime_data,
            'black_hole_data': black_hole_data,
            'experimental_data': experimental_data,
            'entanglement_data': entanglement_data,
            'gw_data': gw_data,
            'connections': connections,
            'summary': summary
        }

    def _estimate_data_size(self, dataset: Dict[str, Any]) -> float:
        """Estimate dataset size in MB."""
        total_size = 0
        for key, value in dataset.items():
            if isinstance(value, list):
                if len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        total_size += sum(arr.nbytes for arr in value)
                    elif isinstance(value[0], dict):
                        total_size += len(str(value)) * 8  # Rough estimate
            elif isinstance(value, np.ndarray):
                total_size += value.nbytes

        return total_size / (1024 * 1024)  # Convert to MB


if __name__ == "__main__":
    # Generate comprehensive synthetic dataset
    generator = SyntheticDataGenerator()
    all_data = generator.generate_comprehensive_dataset()

    print("\n🔍 Dataset generation summary:")
    for dataset_name, info in all_data['summary']['datasets_summary'].items():
        print(f"  • {dataset_name}: {info['n_samples']} samples ({info['data_size_mb']:.1f} MB)")

    print(f"\n🔗 Generated {sum(all_data['summary']['connections_summary'].values())} cross-dataset connections")
