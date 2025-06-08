"""
Accelerator Physics Module for EG-QGEM Theory
Models particle accelerator physics with EG-QGEM modifications

This module provides comprehensive accelerator physics modeling including:
- Beam dynamics with entanglement effects
- Synchrotron radiation modifications
- Collective effects in entangled beams
- Beam-beam interactions with quantum correlations
- Luminosity calculations with EG-QGEM corrections
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import optimize, integrate, special
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import EG-QGEM theory components
from theory.constants import *
from theory.entanglement_tensor import EntanglementTensor
from theory.modified_einstein import ModifiedEinsteinSolver

@dataclass
class BeamParameters:
    """Parameters describing particle beam"""
    energy: float           # GeV
    current: float          # Amperes
    emittance_x: float     # m·rad (horizontal)
    emittance_y: float     # m·rad (vertical)
    beta_x: float          # m (horizontal beta function)
    beta_y: float          # m (vertical beta function)
    sigma_z: float         # m (bunch length)
    num_bunches: int       # number of bunches
    particles_per_bunch: float  # particles per bunch
    entanglement_fraction: float = 0.0  # fraction of entangled particles

@dataclass
class AcceleratorLattice:
    """Accelerator lattice description"""
    circumference: float   # m
    dipole_field: float   # Tesla
    rf_frequency: float   # Hz
    rf_voltage: float     # Volts
    betatron_tune_x: float
    betatron_tune_y: float
    momentum_compaction: float

class EGQGEMAcceleratorPhysics:
    """Main accelerator physics class with EG-QGEM modifications"""

    def __init__(self):
        self.entanglement_tensor = EntanglementTensor()
        self.modified_einstein = ModifiedEinsteinSolver()

        # EG-QGEM accelerator parameters
        self.entanglement_coupling = 1e-18  # GeV^-2
        self.coherence_length = 1e-6        # meters
        self.decoherence_rate = 1e6         # Hz

        # Physical constants in SI units
        self.c = 299792458                  # m/s
        self.e = 1.602176634e-19           # Coulombs
        self.m_e = 9.1093837015e-31        # kg
        self.r_e = 2.8179403262e-15        # m (classical electron radius)

    def calculate_synchrotron_radiation(self, beam: BeamParameters,
                                      lattice: AcceleratorLattice) -> Dict[str, float]:
        """
        Calculate synchrotron radiation with EG-QGEM modifications

        Args:
            beam: Beam parameters
            lattice: Accelerator lattice

        Returns:
            Dictionary with radiation parameters
        """
        # Convert energy to SI units
        energy_si = beam.energy * 1.602176634e-10  # Joules
        gamma = energy_si / (self.m_e * self.c**2)
        beta = np.sqrt(1 - 1/gamma**2)

        # Bending radius
        momentum = gamma * self.m_e * self.c
        rho = momentum / (self.e * lattice.dipole_field)

        # Standard synchrotron radiation power
        standard_power = (2 * self.r_e * self.c * self.e**2 * lattice.dipole_field**2 *
                         gamma**4) / (3 * (4 * np.pi * 8.854187817e-12))

        # EG-QGEM modification due to entanglement
        entanglement_factor = (1 + self.entanglement_coupling *
                             beam.entanglement_fraction * beam.energy**2)

        eg_qgem_power = standard_power * entanglement_factor

        # Critical photon energy
        critical_energy = (3 * gamma**3 * 1.054571817e-34 * self.c) / (2 * rho)
        critical_energy_ev = critical_energy / 1.602176634e-19

        # Radiation damping times
        tau_x = (3 * rho * self.m_e * self.c) / (2 * self.r_e * self.c * gamma**3)
        tau_y = tau_x  # Assume equal for simplicity
        tau_s = tau_x / 2

        # EG-QGEM damping modifications
        entanglement_damping_factor = 1 + 0.1 * beam.entanglement_fraction
        tau_x_eg = tau_x / entanglement_damping_factor
        tau_y_eg = tau_y / entanglement_damping_factor
        tau_s_eg = tau_s / entanglement_damping_factor

        # Quantum excitation
        quantum_excitation = (55 * np.sqrt(3) * self.r_e * 1.054571817e-34 * self.c *
                            gamma**5) / (96 * rho**2 * self.m_e)

        return {
            'standard_power': standard_power,
            'eg_qgem_power': eg_qgem_power,
            'power_enhancement': entanglement_factor,
            'critical_energy_ev': critical_energy_ev,
            'damping_time_x': tau_x_eg,
            'damping_time_y': tau_y_eg,
            'damping_time_s': tau_s_eg,
            'quantum_excitation': quantum_excitation,
            'bending_radius': rho
        }

    def calculate_beam_lifetime(self, beam: BeamParameters,
                               lattice: AcceleratorLattice) -> Dict[str, float]:
        """Calculate beam lifetime with EG-QGEM effects"""

        # Touschek lifetime (intrabeam scattering)
        gamma = beam.energy / 0.000511  # electron rest mass in GeV

        # Momentum acceptance
        momentum_acceptance = 0.01  # 1% (typical)

        # Beam cross-section
        sigma_x = np.sqrt(beam.emittance_x * beam.beta_x)
        sigma_y = np.sqrt(beam.emittance_y * beam.beta_y)

        # Particle density
        density = (beam.particles_per_bunch /
                  (np.sqrt(2*np.pi)**3 * sigma_x * sigma_y * beam.sigma_z))

        # Standard Touschek scattering rate
        standard_rate = (8 * np.pi * self.r_e**2 * self.c * density *
                        np.log(1/momentum_acceptance)) / (3 * gamma**2)

        # EG-QGEM modification
        # Entanglement reduces scattering due to quantum correlations
        entanglement_suppression = 1 - 0.3 * beam.entanglement_fraction
        eg_qgem_rate = standard_rate * entanglement_suppression

        # Convert to lifetime
        touschek_lifetime = 1 / eg_qgem_rate if eg_qgem_rate > 0 else np.inf

        # Gas scattering lifetime
        # Assume vacuum pressure of 1e-9 Torr
        pressure = 1e-9 * 133.322  # Pa
        temperature = 300  # K
        gas_density = pressure / (1.380649e-23 * temperature)  # molecules/m³

        # Cross-section for gas scattering
        gas_cross_section = 2e-22  # m² (typical for residual gas)
        gas_lifetime = 1 / (gas_density * gas_cross_section * self.c)

        # Beam-gas bremsstrahlung
        bremsstrahlung_cross_section = 1e-25  # m²
        bremsstrahlung_lifetime = 1 / (gas_density * bremsstrahlung_cross_section * self.c)

        # Quantum lifetime (spontaneous emission)
        synchrotron_data = self.calculate_synchrotron_radiation(beam, lattice)
        quantum_lifetime = beam.energy / (synchrotron_data['eg_qgem_power'] / self.e)

        # Combined lifetime
        combined_rate = (1/touschek_lifetime + 1/gas_lifetime +
                        1/bremsstrahlung_lifetime + 1/quantum_lifetime)
        combined_lifetime = 1 / combined_rate

        return {
            'touschek_lifetime': touschek_lifetime,
            'gas_scattering_lifetime': gas_lifetime,
            'bremsstrahlung_lifetime': bremsstrahlung_lifetime,
            'quantum_lifetime': quantum_lifetime,
            'combined_lifetime': combined_lifetime,
            'entanglement_benefit': standard_rate / eg_qgem_rate if eg_qgem_rate > 0 else 1.0
        }

    def calculate_luminosity(self, beam1: BeamParameters, beam2: BeamParameters,
                           crossing_angle: float = 0.0) -> Dict[str, float]:
        """
        Calculate luminosity with EG-QGEM corrections

        Args:
            beam1: First beam parameters
            beam2: Second beam parameters
            crossing_angle: Crossing angle in radians

        Returns:
            Dictionary with luminosity calculations
        """
        # Revolution frequencies
        f_rev1 = self.c / 27000  # Assume LHC circumference (27 km)
        f_rev2 = f_rev1

        # Beam sizes at interaction point
        sigma_x1 = np.sqrt(beam1.emittance_x * beam1.beta_x)
        sigma_y1 = np.sqrt(beam1.emittance_y * beam1.beta_y)
        sigma_x2 = np.sqrt(beam2.emittance_x * beam2.beta_x)
        sigma_y2 = np.sqrt(beam2.emittance_y * beam2.beta_y)

        # Combined beam sizes
        sigma_x = np.sqrt(sigma_x1**2 + sigma_x2**2)
        sigma_y = np.sqrt(sigma_y1**2 + sigma_y2**2)

        # Geometric reduction factor for crossing angle
        if crossing_angle > 0:
            sigma_z = min(beam1.sigma_z, beam2.sigma_z)
            geometric_factor = 1 / np.sqrt(1 + (crossing_angle * sigma_z / (2 * sigma_x))**2)
        else:
            geometric_factor = 1.0

        # Standard luminosity formula
        standard_luminosity = (beam1.num_bunches * beam1.particles_per_bunch *
                             beam2.particles_per_bunch * f_rev1 * geometric_factor) / \
                            (4 * np.pi * sigma_x * sigma_y)

        # EG-QGEM enhancement due to quantum correlations
        # Entangled particles have enhanced interaction probability
        entanglement_enhancement = 1 + (self.entanglement_coupling *
                                       beam1.entanglement_fraction *
                                       beam2.entanglement_fraction *
                                       beam1.energy * beam2.energy)

        eg_qgem_luminosity = standard_luminosity * entanglement_enhancement

        # Beam-beam parameter
        xi_x1 = (beam2.particles_per_bunch * self.r_e * beam1.beta_x) / \
               (2 * np.pi * gamma * sigma_x * (sigma_x + sigma_y))
        xi_y1 = (beam2.particles_per_bunch * self.r_e * beam1.beta_y) / \
               (2 * np.pi * gamma * sigma_y * (sigma_x + sigma_y))

        gamma = beam1.energy / 0.000511

        return {
            'standard_luminosity': standard_luminosity,
            'eg_qgem_luminosity': eg_qgem_luminosity,
            'enhancement_factor': entanglement_enhancement,
            'geometric_factor': geometric_factor,
            'beam_beam_xi_x': xi_x1,
            'beam_beam_xi_y': xi_y1,
            'collision_frequency': f_rev1 * beam1.num_bunches
        }

    def simulate_beam_dynamics(self, beam: BeamParameters, lattice: AcceleratorLattice,
                             num_turns: int = 1000) -> Dict[str, Any]:
        """
        Simulate beam dynamics evolution with EG-QGEM effects

        Args:
            beam: Initial beam parameters
            lattice: Accelerator lattice
            num_turns: Number of turns to simulate

        Returns:
            Dictionary with evolution data
        """
        # Initialize arrays for tracking evolution
        turns = np.arange(num_turns)
        emittance_x_evolution = np.zeros(num_turns)
        emittance_y_evolution = np.zeros(num_turns)
        energy_spread_evolution = np.zeros(num_turns)
        entanglement_evolution = np.zeros(num_turns)

        # Initial conditions
        current_emittance_x = beam.emittance_x
        current_emittance_y = beam.emittance_y
        current_energy_spread = 0.001  # 0.1% initial energy spread
        current_entanglement = beam.entanglement_fraction

        # Calculate radiation damping and excitation
        radiation_data = self.calculate_synchrotron_radiation(beam, lattice)

        # Time per turn
        turn_time = lattice.circumference / self.c

        for turn in range(num_turns):
            # Radiation damping
            damping_factor_x = np.exp(-turn_time / radiation_data['damping_time_x'])
            damping_factor_y = np.exp(-turn_time / radiation_data['damping_time_y'])

            # Quantum excitation
            excitation_rate = radiation_data['quantum_excitation'] * turn_time

            # Update emittances
            current_emittance_x = (current_emittance_x * damping_factor_x +
                                 excitation_rate * np.random.normal(0, 0.1))
            current_emittance_y = (current_emittance_y * damping_factor_y +
                                 excitation_rate * np.random.normal(0, 0.1))

            # Energy spread evolution
            current_energy_spread = (current_energy_spread *
                                   np.exp(-turn_time / radiation_data['damping_time_s']) +
                                   excitation_rate * np.random.normal(0, 0.01))

            # Entanglement decoherence
            decoherence_factor = np.exp(-self.decoherence_rate * turn_time)
            current_entanglement *= decoherence_factor

            # Add some random entanglement generation
            if np.random.random() < 0.001:  # 0.1% chance per turn
                current_entanglement += 0.01 * np.random.random()
                current_entanglement = min(current_entanglement, 1.0)

            # Store values
            emittance_x_evolution[turn] = max(current_emittance_x, 0)
            emittance_y_evolution[turn] = max(current_emittance_y, 0)
            energy_spread_evolution[turn] = max(current_energy_spread, 0)
            entanglement_evolution[turn] = np.clip(current_entanglement, 0, 1)

        return {
            'turns': turns,
            'emittance_x': emittance_x_evolution,
            'emittance_y': emittance_y_evolution,
            'energy_spread': energy_spread_evolution,
            'entanglement_fraction': entanglement_evolution,
            'final_emittance_x': current_emittance_x,
            'final_emittance_y': current_emittance_y,
            'final_energy_spread': current_energy_spread,
            'final_entanglement': current_entanglement
        }

    def calculate_collective_effects(self, beam: BeamParameters,
                                   lattice: AcceleratorLattice) -> Dict[str, float]:
        """Calculate collective effects with EG-QGEM modifications"""

        # Space charge tune shift
        gamma = beam.energy / 0.000511
        beta = np.sqrt(1 - 1/gamma**2)

        # Beam current
        average_current = beam.current

        # Space charge parameter
        perveance = (average_current * self.r_e) / (4 * np.pi * 8.854187817e-12 *
                    self.m_e * self.c**3 * beta**3 * gamma**3)

        # Tune shift
        space_charge_tune_shift = -perveance / (4 * np.pi * beam.emittance_x)

        # EG-QGEM modification: entanglement reduces space charge effects
        entanglement_reduction = 1 - 0.2 * beam.entanglement_fraction
        eg_qgem_tune_shift = space_charge_tune_shift * entanglement_reduction

        # Wake field effects
        # Resistive wall wake
        pipe_radius = 0.02  # 2 cm beam pipe radius
        conductivity = 5.8e7  # S/m (copper)

        wake_strength = np.sqrt(self.c * 377 / (2 * conductivity)) / (np.pi * pipe_radius**3)

        # Loss factor for short bunches
        loss_factor = wake_strength * beam.sigma_z

        # Energy loss per turn due to wakes
        wake_energy_loss = loss_factor * (beam.particles_per_bunch * self.e)**2

        # Beam breakup threshold
        breakup_threshold = (2 * np.pi * lattice.betatron_tune_x * beam.energy * self.e) / \
                          (wake_strength * beam.beta_x * self.c)

        return {
            'space_charge_tune_shift_standard': space_charge_tune_shift,
            'space_charge_tune_shift_eg_qgem': eg_qgem_tune_shift,
            'space_charge_improvement': entanglement_reduction,
            'wake_strength': wake_strength,
            'loss_factor': loss_factor,
            'wake_energy_loss_per_turn': wake_energy_loss,
            'beam_breakup_threshold': breakup_threshold,
            'current_margin': breakup_threshold / average_current
        }

    def optimize_entanglement_production(self, beam: BeamParameters,
                                       lattice: AcceleratorLattice) -> Dict[str, Any]:
        """Optimize accelerator parameters for maximum entanglement production"""

        def entanglement_figure_of_merit(params):
            """Figure of merit for entanglement production"""
            beta_x, beta_y, rf_voltage = params

            # Update beam parameters
            test_beam = BeamParameters(
                energy=beam.energy,
                current=beam.current,
                emittance_x=beam.emittance_x,
                emittance_y=beam.emittance_y,
                beta_x=beta_x,
                beta_y=beta_y,
                sigma_z=beam.sigma_z,
                num_bunches=beam.num_bunches,
                particles_per_bunch=beam.particles_per_bunch,
                entanglement_fraction=beam.entanglement_fraction
            )

            # Update lattice
            test_lattice = AcceleratorLattice(
                circumference=lattice.circumference,
                dipole_field=lattice.dipole_field,
                rf_frequency=lattice.rf_frequency,
                rf_voltage=rf_voltage,
                betatron_tune_x=lattice.betatron_tune_x,
                betatron_tune_y=lattice.betatron_tune_y,
                momentum_compaction=lattice.momentum_compaction
            )

            # Calculate various metrics
            radiation = self.calculate_synchrotron_radiation(test_beam, test_lattice)
            lifetime = self.calculate_beam_lifetime(test_beam, test_lattice)
            collective = self.calculate_collective_effects(test_beam, test_lattice)

            # Entanglement production rate (simplified model)
            production_rate = (radiation['power_enhancement'] *
                             lifetime['entanglement_benefit'] *
                             (1 + collective['space_charge_improvement']))

            # Penalty for extreme parameters
            if beta_x < 0.1 or beta_x > 100 or beta_y < 0.1 or beta_y > 100:
                production_rate *= 0.1
            if rf_voltage < 1e6 or rf_voltage > 1e9:
                production_rate *= 0.1

            return -production_rate  # Minimize negative for maximization

        # Initial guess
        initial_params = [beam.beta_x, beam.beta_y, lattice.rf_voltage]

        # Bounds for optimization
        bounds = [(0.1, 100), (0.1, 100), (1e6, 1e9)]

        # Optimize
        result = optimize.minimize(
            entanglement_figure_of_merit,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            optimal_beta_x, optimal_beta_y, optimal_rf_voltage = result.x

            return {
                'success': True,
                'optimal_beta_x': optimal_beta_x,
                'optimal_beta_y': optimal_beta_y,
                'optimal_rf_voltage': optimal_rf_voltage,
                'improvement_factor': -result.fun / (-entanglement_figure_of_merit(initial_params)),
                'original_fom': -entanglement_figure_of_merit(initial_params),
                'optimized_fom': -result.fun
            }
        else:
            return {
                'success': False,
                'message': result.message
            }

    def calculate_beam_beam_interactions(self, beam1: BeamParameters,
                                       beam2: BeamParameters) -> Dict[str, float]:
        """Calculate beam-beam interactions with EG-QGEM effects"""

        # Standard beam-beam force
        gamma1 = beam1.energy / 0.000511
        gamma2 = beam2.energy / 0.000511

        # Beam sizes
        sigma_x1 = np.sqrt(beam1.emittance_x * beam1.beta_x)
        sigma_y1 = np.sqrt(beam1.emittance_y * beam1.beta_y)
        sigma_x2 = np.sqrt(beam2.emittance_x * beam2.beta_x)
        sigma_y2 = np.sqrt(beam2.emittance_y * beam2.beta_y)

        # Combined sizes
        sigma_x = np.sqrt(sigma_x1**2 + sigma_x2**2)
        sigma_y = np.sqrt(sigma_y1**2 + sigma_y2**2)

        # Beam-beam parameter for beam 1
        xi_x1 = (beam2.particles_per_bunch * self.r_e * beam1.beta_x) / \
               (2 * np.pi * gamma1 * sigma_x * (sigma_x + sigma_y))
        xi_y1 = (beam2.particles_per_bunch * self.r_e * beam1.beta_y) / \
               (2 * np.pi * gamma1 * sigma_y * (sigma_x + sigma_y))

        # EG-QGEM modifications due to entanglement
        # Entangled beams experience modified electromagnetic interactions
        entanglement_correlation = np.sqrt(beam1.entanglement_fraction *
                                         beam2.entanglement_fraction)

        # Modified beam-beam parameters
        xi_modification = 1 + self.entanglement_coupling * entanglement_correlation * \
                         beam1.energy * beam2.energy

        xi_x1_eg = xi_x1 * xi_modification
        xi_y1_eg = xi_y1 * xi_modification

        # Beam-beam limit
        typical_limit = 0.003  # Typical beam-beam limit
        safety_margin_x = typical_limit / abs(xi_x1_eg) if xi_x1_eg != 0 else np.inf
        safety_margin_y = typical_limit / abs(xi_y1_eg) if xi_y1_eg != 0 else np.inf

        # Tune footprint
        max_amplitude = 6  # 6 sigma particles
        amplitude_range = np.linspace(0, max_amplitude, 100)

        # Simplified tune shift with amplitude
        tune_shift_x = xi_x1_eg * np.exp(-amplitude_range**2 / 2)
        tune_shift_y = xi_y1_eg * np.exp(-amplitude_range**2 / 2)

        return {
            'xi_x_standard': xi_x1,
            'xi_y_standard': xi_y1,
            'xi_x_eg_qgem': xi_x1_eg,
            'xi_y_eg_qgem': xi_y1_eg,
            'entanglement_modification': xi_modification,
            'safety_margin_x': safety_margin_x,
            'safety_margin_y': safety_margin_y,
            'tune_footprint_x': tune_shift_x,
            'tune_footprint_y': tune_shift_y,
            'amplitude_range': amplitude_range
        }

    def generate_accelerator_plots(self, beam: BeamParameters,
                                 lattice: AcceleratorLattice,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive accelerator physics plots"""

        # Run beam dynamics simulation
        dynamics = self.simulate_beam_dynamics(beam, lattice, num_turns=1000)

        # Calculate various physics quantities
        radiation = self.calculate_synchrotron_radiation(beam, lattice)
        lifetime = self.calculate_beam_lifetime(beam, lattice)
        collective = self.calculate_collective_effects(beam, lattice)

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Emittance evolution
        axes[0, 0].plot(dynamics['turns'], dynamics['emittance_x'],
                       label='Horizontal', color='blue')
        axes[0, 0].plot(dynamics['turns'], dynamics['emittance_y'],
                       label='Vertical', color='red')
        axes[0, 0].set_xlabel('Turn Number')
        axes[0, 0].set_ylabel('Emittance (m·rad)')
        axes[0, 0].set_title('Emittance Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Energy spread evolution
        axes[0, 1].plot(dynamics['turns'], dynamics['energy_spread'], color='green')
        axes[0, 1].set_xlabel('Turn Number')
        axes[0, 1].set_ylabel('Energy Spread')
        axes[0, 1].set_title('Energy Spread Evolution')
        axes[0, 1].grid(True)

        # Entanglement evolution
        axes[1, 0].plot(dynamics['turns'], dynamics['entanglement_fraction'],
                       color='purple')
        axes[1, 0].set_xlabel('Turn Number')
        axes[1, 0].set_ylabel('Entanglement Fraction')
        axes[1, 0].set_title('Entanglement Evolution')
        axes[1, 0].grid(True)

        # Synchrotron radiation spectrum (simplified)
        photon_energies = np.logspace(-3, 2, 1000)  # keV
        critical_energy_kev = radiation['critical_energy_ev'] / 1000

        # Simplified synchrotron spectrum
        spectrum = (photon_energies / critical_energy_kev) * \
                  np.exp(-photon_energies / critical_energy_kev)

        axes[1, 1].loglog(photon_energies, spectrum, color='orange')
        axes[1, 1].axvline(critical_energy_kev, color='red', linestyle='--',
                          label=f'Critical Energy: {critical_energy_kev:.1f} keV')
        axes[1, 1].set_xlabel('Photon Energy (keV)')
        axes[1, 1].set_ylabel('Relative Intensity')
        axes[1, 1].set_title('Synchrotron Radiation Spectrum')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Beam lifetime components
        lifetime_components = [
            ('Touschek', lifetime['touschek_lifetime']),
            ('Gas Scattering', lifetime['gas_scattering_lifetime']),
            ('Bremsstrahlung', lifetime['bremsstrahlung_lifetime']),
            ('Quantum', lifetime['quantum_lifetime'])
        ]

        names = [comp[0] for comp in lifetime_components]
        values = [comp[1] / 3600 for comp in lifetime_components]  # Convert to hours

        axes[2, 0].bar(names, values, color=['blue', 'red', 'green', 'orange'])
        axes[2, 0].set_ylabel('Lifetime (hours)')
        axes[2, 0].set_title('Beam Lifetime Components')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, axis='y')

        # Phase space plot (simplified)
        theta = np.linspace(0, 2*np.pi, 1000)
        x_orbit = np.sqrt(dynamics['emittance_x'][-1] * beam.beta_x) * np.cos(theta)
        xp_orbit = np.sqrt(dynamics['emittance_x'][-1] / beam.beta_x) * np.sin(theta)

        axes[2, 1].plot(x_orbit * 1e6, xp_orbit * 1e6, color='blue')
        axes[2, 1].set_xlabel('X Position (μm)')
        axes[2, 1].set_ylabel('X Angle (μrad)')
        axes[2, 1].set_title('Horizontal Phase Space')
        axes[2, 1].grid(True)
        axes[2, 1].set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return {
            'figure': fig,
            'final_emittance_x': dynamics['final_emittance_x'],
            'final_emittance_y': dynamics['final_emittance_y'],
            'final_entanglement': dynamics['final_entanglement'],
            'radiation_power': radiation['eg_qgem_power'],
            'combined_lifetime_hours': lifetime['combined_lifetime'] / 3600
        }
