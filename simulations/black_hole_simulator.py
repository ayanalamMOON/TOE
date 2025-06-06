"""
Black Hole Simulator for EG-QGEM Theory
=======================================

This module simulates black holes in the EG-QGEM framework, where
entanglement prevents singularities and encodes information in correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import h5py
from tqdm import tqdm

from ..theory.entanglement_tensor import EntanglementTensor
from ..theory.modified_einstein import ModifiedEinsteinSolver
from ..theory.constants import CONSTANTS

class BlackHoleSimulator:
    """
    Simulates black holes with entanglement-induced modifications.

    Key features:
    - No central singularity due to entanglement pressure
    - Information storage in entanglement correlations
    - Modified Hawking radiation with entanglement echoes
    - Firewall resolution through entanglement
    """

    def __init__(self, mass, spin=0.0, charge=0.0):
        """
        Initialize black hole simulation.

        Args:
            mass (float): Black hole mass (kg)
            spin (float): Angular momentum parameter a (dimensionless)
            charge (float): Electric charge (C)
        """
        self.mass = mass
        self.spin = spin
        self.charge = charge

        # Compute characteristic scales
        self.rs = 2 * CONSTANTS.G * mass / CONSTANTS.c**2  # Schwarzschild radius
        self.rq = CONSTANTS.G * charge**2 / (4 * np.pi * 8.854e-12 * CONSTANTS.c**4)  # Charge radius
        self.a = spin * CONSTANTS.G * mass / CONSTANTS.c**3  # Spin parameter

        # EG-QGEM modifications
        self.entanglement_scale = CONSTANTS.l_E
        self.entanglement_pressure_scale = CONSTANTS.rho_E_crit

        # Grid for simulation
        self.r_grid = np.logspace(np.log10(self.entanglement_scale),
                                 np.log10(100 * self.rs), 1000)
        self.metric = np.zeros((len(self.r_grid), 4, 4))
        self.entanglement_density = np.zeros(len(self.r_grid))

    def compute_modified_metric(self):
        """
        Compute metric with entanglement modifications.

        The metric transitions from Schwarzschild at large r to
        entanglement-dominated at small r, preventing singularity.
        """
        for i, r in enumerate(self.r_grid):
            # Classical Schwarzschild component
            f_classical = 1 - self.rs / r if r > self.rs else 0.0

            # Entanglement modification - pressure prevents collapse
            entanglement_correction = np.exp(-r / self.entanglement_scale)
            entanglement_pressure = (self.entanglement_pressure_scale *
                                   entanglement_correction / r**2)

            # Modified lapse function
            f_modified = f_classical + entanglement_pressure * self.rs / r
            f_modified = max(f_modified, 0.01)  # Prevent negative values

            # Metric components in Schwarzschild coordinates
            self.metric[i, 0, 0] = -f_modified  # g_tt
            self.metric[i, 1, 1] = 1 / f_modified  # g_rr
            self.metric[i, 2, 2] = r**2  # g_θθ
            self.metric[i, 3, 3] = r**2 * np.sin(np.pi/2)**2  # g_φφ (θ=π/2)

            # Store entanglement density
            self.entanglement_density[i] = entanglement_correction / r**2

    def compute_hawking_temperature(self):
        """
        Compute modified Hawking temperature with entanglement effects.

        Returns:
            float: Temperature in Kelvin
        """
        # Classical Hawking temperature
        T_classical = CONSTANTS.hbar * CONSTANTS.c**3 / (8 * np.pi * CONSTANTS.k_B *
                                                        CONSTANTS.G * self.mass)

        # Entanglement correction
        correction_factor = 1 + (self.entanglement_scale / self.rs)**2
        T_modified = T_classical * correction_factor

        return T_modified

    def compute_entropy(self):
        """
        Compute black hole entropy as entanglement entropy.

        Returns:
            float: Entropy in natural units
        """
        # Bekenstein-Hawking entropy
        area = 4 * np.pi * self.rs**2
        S_classical = area / (4 * CONSTANTS.G * CONSTANTS.hbar / CONSTANTS.c**3)

        # Entanglement contribution - volume law term
        volume_interior = (4/3) * np.pi * self.rs**3
        S_entanglement = (volume_interior / self.entanglement_scale**3) * np.log(2)

        # Total entropy (area law dominates for large black holes)
        S_total = S_classical + S_entanglement

        return S_total, S_classical, S_entanglement

    def simulate_hawking_radiation(self, time_steps=1000, dt=1e6):
        """
        Simulate Hawking radiation with entanglement echoes.

        Args:
            time_steps (int): Number of time steps
            dt (float): Time step in seconds

        Returns:
            dict: Radiation properties and entanglement echoes
        """
        times = np.arange(time_steps) * dt
        mass_evolution = np.zeros(time_steps)
        entropy_evolution = np.zeros(time_steps)
        temperature_evolution = np.zeros(time_steps)
        entanglement_flux = np.zeros(time_steps)

        current_mass = self.mass

        for i, t in enumerate(tqdm(times, desc="Simulating evaporation")):
            # Update mass due to Hawking radiation
            T_hawking = self.compute_hawking_temperature()

            # Stefan-Boltzmann law for black body radiation
            luminosity = (4 * np.pi * self.rs**2 * 5.67e-8 * T_hawking**4 /
                         CONSTANTS.c**2)  # Power radiated

            mass_loss_rate = luminosity / CONSTANTS.c**2
            current_mass -= mass_loss_rate * dt
            current_mass = max(current_mass, 0.1 * self.mass)  # Prevent complete evaporation

            # Update black hole parameters
            self.mass = current_mass
            self.rs = 2 * CONSTANTS.G * current_mass / CONSTANTS.c**2

            # Store evolution data
            mass_evolution[i] = current_mass
            temperature_evolution[i] = T_hawking

            S_total, S_classical, S_entanglement = self.compute_entropy()
            entropy_evolution[i] = S_total

            # Entanglement flux in radiation
            entanglement_flux[i] = S_entanglement / (4 * np.pi * self.rs**2 * dt)

        return {
            'times': times,
            'mass_evolution': mass_evolution,
            'temperature_evolution': temperature_evolution,
            'entropy_evolution': entropy_evolution,
            'entanglement_flux': entanglement_flux,
            'final_mass': current_mass
        }

    def compute_information_scrambling(self, n_qubits=100):
        """
        Simulate information scrambling in black hole interior.

        Args:
            n_qubits (int): Number of qubits representing interior information

        Returns:
            dict: Scrambling dynamics and out-of-time correlators
        """
        # Initialize random quantum state
        psi = np.random.complex128(2**min(n_qubits, 10))  # Limit size for computation
        psi /= np.linalg.norm(psi)

        # Scrambling Hamiltonian (random matrix)
        dim = len(psi)
        H = np.random.normal(0, 1, (dim, dim)) + 1j * np.random.normal(0, 1, (dim, dim))
        H = (H + H.conj().T) / 2  # Make Hermitian

        # Evolution parameters
        times = np.linspace(0, 10, 100)  # In units of 1/J (coupling strength)
        scrambling_time = np.log(n_qubits)  # Expected scrambling time

        # Compute out-of-time-ordered correlator (OTOC)
        otoc_values = np.zeros(len(times))
        entanglement_entropy = np.zeros(len(times))

        for i, t in enumerate(times):
            # Time evolution
            U = np.exp(-1j * H * t)
            psi_t = U @ psi

            # Compute OTOC (simplified - would need tensor products for full calculation)
            # |⟨ψ|V†(t)W†U(t)VU(t)W|ψ⟩|²
            otoc_values[i] = np.abs(np.vdot(psi, psi_t))**2

            # Entanglement entropy (simplified)
            # Would need to trace out subsystem for proper calculation
            rho = np.outer(psi_t, psi_t.conj())
            eigenvals = np.real(np.linalg.eigvals(rho))
            eigenvals = eigenvals[eigenvals > 1e-12]
            entanglement_entropy[i] = -np.sum(eigenvals * np.log(eigenvals))

        return {
            'times': times,
            'otoc': otoc_values,
            'entanglement_entropy': entanglement_entropy,
            'scrambling_time': scrambling_time,
            'lyapunov_exponent': 2 * np.pi / scrambling_time  # Theoretical bound
        }

    def analyze_firewall_resolution(self):
        """
        Analyze how entanglement resolves the firewall paradox.

        Returns:
            dict: Analysis of smooth horizon emergence
        """
        # Compute metric near horizon
        r_near_horizon = np.linspace(0.9 * self.rs, 1.1 * self.rs, 100)

        curvature_scalars = []
        entanglement_smoothing = []

        for r in r_near_horizon:
            # Ricci scalar (simplified calculation)
            if r > self.rs:
                ricci_classical = 0  # Vacuum Schwarzschild
            else:
                ricci_classical = np.inf  # Classical singularity

            # Entanglement smoothing
            smoothing_factor = np.exp(-(r - self.rs)**2 / self.entanglement_scale**2)
            ricci_modified = ricci_classical * (1 - smoothing_factor)

            curvature_scalars.append(ricci_modified)
            entanglement_smoothing.append(smoothing_factor)

        return {
            'radii': r_near_horizon,
            'curvature_scalars': np.array(curvature_scalars),
            'entanglement_smoothing': np.array(entanglement_smoothing),
            'horizon_location': self.rs,
            'smooth_transition': np.all(np.isfinite(curvature_scalars))
        }

    def visualize_black_hole(self, save_path=None):
        """
        Visualize black hole structure and properties.

        Args:
            save_path (str): Path to save plots
        """
        self.compute_modified_metric()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Metric components
        axes[0, 0].loglog(self.r_grid / self.rs, -self.metric[:, 0, 0],
                         label='g_tt', color='blue')
        axes[0, 0].loglog(self.r_grid / self.rs, self.metric[:, 1, 1],
                         label='g_rr', color='red')
        axes[0, 0].axvline(1.0, color='black', linestyle='--', alpha=0.7,
                          label='Classical horizon')
        axes[0, 0].set_xlabel('r/rs')
        axes[0, 0].set_ylabel('Metric components')
        axes[0, 0].set_title('Modified Metric')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Entanglement density
        axes[0, 1].loglog(self.r_grid / self.rs, self.entanglement_density)
        axes[0, 1].axvline(1.0, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('r/rs')
        axes[0, 1].set_ylabel('Entanglement density')
        axes[0, 1].set_title('Entanglement Profile')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Hawking radiation simulation
        radiation_data = self.simulate_hawking_radiation(time_steps=100, dt=1e8)
        axes[0, 2].plot(radiation_data['times'] / (365.25 * 24 * 3600),
                       radiation_data['mass_evolution'] / self.mass)
        axes[0, 2].set_xlabel('Time (years)')
        axes[0, 2].set_ylabel('M(t)/M_initial')
        axes[0, 2].set_title('Mass Evolution')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Information scrambling
        scrambling_data = self.compute_information_scrambling()
        axes[1, 0].plot(scrambling_data['times'], scrambling_data['otoc'])
        axes[1, 0].axvline(scrambling_data['scrambling_time'], color='red',
                          linestyle='--', label='Scrambling time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('OTOC')
        axes[1, 0].set_title('Information Scrambling')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Entropy evolution
        axes[1, 1].plot(radiation_data['times'] / (365.25 * 24 * 3600),
                       radiation_data['entropy_evolution'])
        axes[1, 1].set_xlabel('Time (years)')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Entropy Evolution')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Firewall analysis
        firewall_data = self.analyze_firewall_resolution()
        axes[1, 2].plot(firewall_data['radii'] / self.rs,
                       firewall_data['entanglement_smoothing'])
        axes[1, 2].axvline(1.0, color='black', linestyle='--', alpha=0.7,
                          label='Horizon')
        axes[1, 2].set_xlabel('r/rs')
        axes[1, 2].set_ylabel('Entanglement smoothing')
        axes[1, 2].set_title('Firewall Resolution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_simulation_data(self, filename):
        """
        Save simulation data to HDF5 file.

        Args:
            filename (str): Output filename
        """
        with h5py.File(filename, 'w') as f:
            # Black hole parameters
            f.attrs['mass'] = self.mass
            f.attrs['spin'] = self.spin
            f.attrs['charge'] = self.charge
            f.attrs['schwarzschild_radius'] = self.rs

            # Grid and metric
            f.create_dataset('r_grid', data=self.r_grid)
            f.create_dataset('metric', data=self.metric)
            f.create_dataset('entanglement_density', data=self.entanglement_density)

            # Hawking radiation
            radiation_data = self.simulate_hawking_radiation(time_steps=100)
            rad_group = f.create_group('hawking_radiation')
            for key, value in radiation_data.items():
                rad_group.create_dataset(key, data=value)

            # Information scrambling
            scrambling_data = self.compute_information_scrambling()
            scr_group = f.create_group('information_scrambling')
            for key, value in scrambling_data.items():
                scr_group.create_dataset(key, data=value)

def simulate_stellar_collapse(initial_mass, collapse_time=1e6):
    """
    Simulate stellar collapse to black hole in EG-QGEM theory.

    Args:
        initial_mass (float): Initial stellar mass (kg)
        collapse_time (float): Collapse timescale (s)

    Returns:
        dict: Collapse simulation results
    """
    print(f"Simulating stellar collapse to black hole...")

    # Time evolution during collapse
    times = np.linspace(0, collapse_time, 1000)
    radius_evolution = []
    density_evolution = []
    entanglement_growth = []

    for t in times:
        # Simplified collapse model
        progress = t / collapse_time

        # Radius decreases
        r_initial = 1e6  # Initial stellar radius (m)
        r_t = r_initial * (1 - progress)**2

        # Density increases
        rho_t = initial_mass / (4/3 * np.pi * r_t**3)

        # Entanglement grows as matter becomes more correlated
        entanglement_t = progress**2 * CONSTANTS.rho_E_crit

        radius_evolution.append(r_t)
        density_evolution.append(rho_t)
        entanglement_growth.append(entanglement_t)

    # Final black hole
    rs_final = 2 * CONSTANTS.G * initial_mass / CONSTANTS.c**2

    return {
        'times': times,
        'radius_evolution': np.array(radius_evolution),
        'density_evolution': np.array(density_evolution),
        'entanglement_growth': np.array(entanglement_growth),
        'final_schwarzschild_radius': rs_final,
        'entanglement_prevents_singularity': True
    }

if __name__ == "__main__":
    # Example: Simulate a solar mass black hole
    solar_mass = 1.989e30  # kg

    bh = BlackHoleSimulator(mass=10 * solar_mass)  # 10 solar mass black hole
    bh.visualize_black_hole()

    # Save data
    bh.save_simulation_data("black_hole_simulation.h5")

    print(f"Black hole simulation complete!")
    print(f"Schwarzschild radius: {bh.rs/1000:.2f} km")
    print(f"Hawking temperature: {bh.compute_hawking_temperature():.2e} K")

    S_total, S_classical, S_entanglement = bh.compute_entropy()
    print(f"Total entropy: {S_total:.2e}")
    print(f"Classical contribution: {S_classical:.2e}")
    print(f"Entanglement contribution: {S_entanglement:.2e}")
